//! 运行时版 Node 接口：用 `NodeBuffer` 做类型擦除，便于 `Vec<Box<dyn DynNode>>` 拼 pipeline。

use crate::codec::decoder::decoder_interface::AudioDecoder;
use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::AudioFrameView;
use crate::pipeline::node::node_interface::{NodeBuffer, NodeBufferKind};
use std::collections::VecDeque;

/// 动态 node：输入/输出都是 `NodeBuffer`，用 `CodecError::{Again,Eof}` 表达背压/结束。
pub trait DynNode: Send {
    fn name(&self) -> &'static str;
    fn input_kind(&self) -> NodeBufferKind;
    fn output_kind(&self) -> NodeBufferKind;

    /// - `Some(buf)`: 正常输入
    /// - `None`: flush（输入结束）
    fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()>;

    /// - `Ok(buf)`: 得到一个输出
    /// - `Err(Again)`: 暂无输出，需要更多输入或继续驱动
    /// - `Err(Eof)`: flush 后已无更多输出
    fn pull(&mut self) -> CodecResult<NodeBuffer>;
}

pub struct ProcessorNode<P: AudioProcessor> {
    p: P,
}

impl<P: AudioProcessor> ProcessorNode<P> {
    pub fn new(p: P) -> Self {
        Self { p }
    }
}

impl<P: AudioProcessor> DynNode for ProcessorNode<P> {
    fn name(&self) -> &'static str {
        self.p.name()
    }

    fn input_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Pcm
    }

    fn output_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Pcm
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()> {
        match input {
            None => self.p.send_frame(None),
            Some(NodeBuffer::Pcm(f)) => self.p.send_frame(Some(&f as &dyn AudioFrameView)),
            Some(_) => Err(CodecError::InvalidData("processor expects PCM input")),
        }
    }

    fn pull(&mut self) -> CodecResult<NodeBuffer> {
        self.p.receive_frame().map(NodeBuffer::Pcm)
    }
}

/// Identity 节点：把输入 buffer 原样移动到输出（零拷贝，不做任何处理）。
///
/// - 作为占位节点，便于后续插入 processor/filter。
pub struct IdentityNode {
    kind: NodeBufferKind,
    q: VecDeque<NodeBuffer>,
    flushed: bool,
}

impl IdentityNode {
    pub fn new(kind: NodeBufferKind) -> Self {
        Self {
            kind,
            q: VecDeque::new(),
            flushed: false,
        }
    }
}

impl DynNode for IdentityNode {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn input_kind(&self) -> NodeBufferKind {
        self.kind
    }

    fn output_kind(&self) -> NodeBufferKind {
        self.kind
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        match input {
            None => {
                self.flushed = true;
                Ok(())
            }
            Some(buf) => {
                if buf.kind() != self.kind {
                    return Err(CodecError::InvalidData("identity node buffer kind mismatch"));
                }
                self.q.push_back(buf);
                Ok(())
            }
        }
    }

    fn pull(&mut self) -> CodecResult<NodeBuffer> {
        if let Some(v) = self.q.pop_front() {
            return Ok(v);
        }
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }
}

pub struct EncoderNode<E: AudioEncoder> {
    e: E,
}

impl<E: AudioEncoder> EncoderNode<E> {
    pub fn new(e: E) -> Self {
        Self { e }
    }
}

impl<E: AudioEncoder> DynNode for EncoderNode<E> {
    fn name(&self) -> &'static str {
        self.e.name()
    }

    fn input_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Pcm
    }

    fn output_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Packet
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()> {
        match input {
            None => self.e.send_frame(None),
            Some(NodeBuffer::Pcm(f)) => self.e.send_frame(Some(&f as &dyn AudioFrameView)),
            Some(_) => Err(CodecError::InvalidData("encoder expects PCM input")),
        }
    }

    fn pull(&mut self) -> CodecResult<NodeBuffer> {
        self.e.receive_packet().map(NodeBuffer::Packet)
    }
}

pub struct DecoderNode<D: AudioDecoder> {
    d: D,
}

impl<D: AudioDecoder> DecoderNode<D> {
    pub fn new(d: D) -> Self {
        Self { d }
    }
}

impl<D: AudioDecoder> DynNode for DecoderNode<D> {
    fn name(&self) -> &'static str {
        self.d.name()
    }

    fn input_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Packet
    }

    fn output_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Pcm
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()> {
        match input {
            None => self.d.send_packet(None),
            Some(NodeBuffer::Packet(p)) => self.d.send_packet(Some(p)),
            Some(_) => Err(CodecError::InvalidData("decoder expects Packet input")),
        }
    }

    fn pull(&mut self) -> CodecResult<NodeBuffer> {
        self.d.receive_frame().map(NodeBuffer::Pcm)
    }
}

/// 一个简单的运行时 pipeline runner：把节点串成链，负责把上游输出推入下游。
pub struct DynPipeline {
    nodes: Vec<Box<dyn DynNode>>,
    done: Vec<bool>,
}

impl DynPipeline {
    pub fn new(nodes: Vec<Box<dyn DynNode>>) -> Result<Self, CodecError> {
        if nodes.is_empty() {
            return Err(CodecError::InvalidData("pipeline requires at least 1 node"));
        }
        // 基础校验：相邻节点 output_kind == input_kind
        for i in 0..(nodes.len() - 1) {
            let ok = nodes[i].output_kind() == nodes[i + 1].input_kind();
            if !ok {
                return Err(CodecError::InvalidData("pipeline node kind mismatch"));
            }
        }
        Ok(Self {
            done: vec![false; nodes.len()],
            nodes,
        })
    }

    /// 推入一个输入（或 flush），并尽可能把数据跑到末端，返回末端所有可用输出。
    pub fn push_and_drain(&mut self, input: Option<NodeBuffer>) -> CodecResult<Vec<NodeBuffer>> {
        self.nodes[0].push(input)?;
        self.drain_all()
    }

    /// 尽可能推进整条链，把末端所有可用输出取出来。
    pub fn drain_all(&mut self) -> CodecResult<Vec<NodeBuffer>> {
        // 从前到后：把每一段的输出尽可能推到下一段
        for i in 0..(self.nodes.len() - 1) {
            if self.done[i] {
                continue;
            }
            loop {
                match self.nodes[i].pull() {
                    Ok(buf) => {
                        self.nodes[i + 1].push(Some(buf))?;
                    }
                    Err(CodecError::Again) => break,
                    Err(CodecError::Eof) => {
                        // 上游结束，通知下游 flush
                        self.done[i] = true;
                        self.nodes[i + 1].push(None)?;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // 收集末端输出
        let last = self.nodes.len() - 1;
        let mut outs = Vec::new();
        loop {
            match self.nodes[last].pull() {
                Ok(buf) => outs.push(buf),
                Err(CodecError::Again) | Err(CodecError::Eof) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(outs)
    }
}


