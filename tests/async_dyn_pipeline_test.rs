use audiostream::codec::error::CodecError;
use audiostream::codec::packet::CodecPacket;
use audiostream::common::audio::audio::Rational;
use audiostream::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
use audiostream::pipeline::node::dynamic_node_interface::DynNode;
use audiostream::pipeline::node::node_interface::{AsyncPipeline, NodeBuffer, NodeBufferKind};
use std::collections::VecDeque;

struct PacketEchoNode {
    q: VecDeque<NodeBuffer>,
    flushed: bool,
}

impl PacketEchoNode {
    fn new() -> Self {
        Self {
            q: VecDeque::new(),
            flushed: false,
        }
    }
}

impl DynNode for PacketEchoNode {
    fn name(&self) -> &'static str {
        "packet-echo"
    }
    fn input_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Packet
    }
    fn output_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Packet
    }
    fn push(&mut self, input: Option<NodeBuffer>) -> audiostream::codec::error::CodecResult<()> {
        match input {
            None => {
                self.flushed = true;
                Ok(())
            }
            Some(NodeBuffer::Packet(p)) => {
                // echo: prefix 0xAA
                let mut d = vec![0xAA];
                d.extend_from_slice(&p.data);
                let mut out = CodecPacket::new(d, p.time_base);
                out.pts = p.pts;
                out.duration = p.duration;
                self.q.push_back(NodeBuffer::Packet(out));
                Ok(())
            }
            Some(_) => Err(CodecError::InvalidData("expects packet")),
        }
    }
    fn pull(&mut self) -> audiostream::codec::error::CodecResult<NodeBuffer> {
        if let Some(v) = self.q.pop_front() {
            return Ok(v);
        }
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn async_dyn_pipeline_push_then_get() {
    let nodes: Vec<Box<dyn DynNode>> = vec![Box::new(PacketEchoNode::new()), Box::new(PacketEchoNode::new())];
    let mut p = AsyncDynPipeline::new(nodes).unwrap();

    let tb = Rational::new(1, 1);
    p.push_frame(NodeBuffer::Packet(CodecPacket::new(vec![1, 2, 3], tb))).unwrap();
    p.push_frame(NodeBuffer::Packet(CodecPacket::new(vec![4], tb))).unwrap();
    p.flush().unwrap();

    let mut outs = Vec::new();
    loop {
        match p.get_frame().await {
            Ok(NodeBuffer::Packet(pkt)) => outs.push(pkt.data),
            Err(CodecError::Eof) => break,
            Err(e) => panic!("unexpected err: {e:?}"),
            _ => panic!("unexpected kind"),
        }
    }
    assert_eq!(outs, vec![vec![0xAA, 0xAA, 1, 2, 3], vec![0xAA, 0xAA, 4]]);
}

struct CounterNode {
    q: VecDeque<NodeBuffer>,
    flushed: bool,
    ctr: u8,
}

impl CounterNode {
    fn new() -> Self {
        Self {
            q: VecDeque::new(),
            flushed: false,
            ctr: 0,
        }
    }
}

impl DynNode for CounterNode {
    fn name(&self) -> &'static str {
        "counter"
    }
    fn input_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Packet
    }
    fn output_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Packet
    }
    fn push(&mut self, input: Option<NodeBuffer>) -> audiostream::codec::error::CodecResult<()> {
        match input {
            None => {
                self.flushed = true;
                Ok(())
            }
            Some(NodeBuffer::Packet(p)) => {
                let mut d = vec![self.ctr];
                d.extend_from_slice(&p.data);
                self.ctr = self.ctr.wrapping_add(1);
                let mut out = CodecPacket::new(d, p.time_base);
                out.pts = p.pts;
                out.duration = p.duration;
                self.q.push_back(NodeBuffer::Packet(out));
                Ok(())
            }
            Some(_) => Err(CodecError::InvalidData("expects packet")),
        }
    }
    fn pull(&mut self) -> audiostream::codec::error::CodecResult<NodeBuffer> {
        if let Some(v) = self.q.pop_front() {
            return Ok(v);
        }
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }
    fn reset(&mut self) -> audiostream::codec::error::CodecResult<()> {
        self.q.clear();
        self.flushed = false;
        self.ctr = 0;
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn async_dyn_pipeline_reset_preserves_pending_outputs_when_not_forced() {
    let nodes: Vec<Box<dyn DynNode>> = vec![Box::new(CounterNode::new()), Box::new(CounterNode::new())];
    let mut p = AsyncDynPipeline::new(nodes).unwrap();
    let tb = Rational::new(1, 1);

    // 先 push 一帧，但不读取输出；直接 reset(force=false)。
    p.push_frame(NodeBuffer::Packet(CodecPacket::new(vec![1], tb))).unwrap();
    p.reset(false).await.unwrap();

    // reset 前的输出不应被强行丢弃（应该仍可读到）
    let out1 = p.get_frame().await.unwrap();
    let NodeBuffer::Packet(pkt1) = out1 else { panic!("unexpected kind") };
    assert_eq!(pkt1.data, vec![0, 0, 1]);

    // reset 后计数器应归零
    p.push_frame(NodeBuffer::Packet(CodecPacket::new(vec![2], tb))).unwrap();
    let out2 = p.get_frame().await.unwrap();
    let NodeBuffer::Packet(pkt2) = out2 else { panic!("unexpected kind") };
    assert_eq!(pkt2.data, vec![0, 0, 2]);
}

