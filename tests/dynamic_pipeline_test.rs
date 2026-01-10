use audiostream::pipeline::node::dynamic_node_interface::{DynNode, DynPipeline};
use audiostream::pipeline::node::node_interface::{IdentityNode, NodeBuffer, NodeBufferKind};
use audiostream::codec::error::{CodecError, CodecResult};
use audiostream::common::audio::audio::{AudioFrame, AudioFormat, ChannelLayout, SampleFormat};
use std::collections::VecDeque;

/// 一个用于测试背压的节点：
/// - `push(Some(_))` 永远返回 `Again`（模拟“输出队列未取空/需要先 pull”的背压）
/// - 但它仍会缓存输入，随后可通过 `pull()` 取回
struct AgainOnPushNode {
    kind: NodeBufferKind,
    q: VecDeque<NodeBuffer>,
    flushed: bool,
}

impl AgainOnPushNode {
    fn new(kind: NodeBufferKind) -> Self {
        Self {
            kind,
            q: VecDeque::new(),
            flushed: false,
        }
    }
}

impl DynNode for AgainOnPushNode {
    fn name(&self) -> &'static str {
        "again-on-push"
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
                Err(CodecError::Again)
            }
            Some(buf) => {
                if buf.kind() != self.kind {
                    return Err(CodecError::InvalidData("buffer kind mismatch"));
                }
                self.q.push_back(buf);
                Err(CodecError::Again)
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

fn make_pcm(nb_samples: usize) -> AudioFrame {
    let fmt = AudioFormat {
        sample_rate: 48_000,
        sample_format: SampleFormat::F32 { planar: true },
        channel_layout: ChannelLayout::stereo(),
    };
    AudioFrame::new_alloc(fmt, nb_samples).unwrap()
}

#[test]
fn dynpipeline_does_not_fail_on_again_from_push() {
    let nodes: Vec<Box<dyn DynNode>> = vec![
        Box::new(AgainOnPushNode::new(NodeBufferKind::Pcm)),
        Box::new(IdentityNode::new(NodeBufferKind::Pcm)),
    ];
    let mut p = DynPipeline::new(nodes).unwrap();

    // 推入 3 个 chunk，确保每次 push 都是 Again，但 pipeline 仍能产出
    for _ in 0..3 {
        let outs = p.push_and_drain(Some(NodeBuffer::Pcm(make_pcm(256)))).unwrap();
        assert_eq!(outs.len(), 1);
        assert!(matches!(outs[0], NodeBuffer::Pcm(_)));
    }

    // flush：不应卡死/报错
    let _ = p.push_and_drain(None).unwrap();
    loop {
        let outs = p.drain_all().unwrap();
        if outs.is_empty() {
            break;
        }
    }
}


