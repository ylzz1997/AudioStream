//! Tap / Tee 节点：把数据 **透传到下游** 的同时，**复制一份** 交给一个 `AudioSink` 处理。
//!
//! 适用场景：
//! - 在 pipeline 中间插一个旁路写盘/统计/监控，而不影响主链路的类型连接。
//!
//! 语义约定：
//! - `push(Some(frame))`：将 `frame.clone()` 交给 side-sink，再把原始 `frame` 入队等待下游 `pull()`
//! - `push(None)`：视为 stream 结束（flush），会对 side-sink 调用一次 `finalize()`
//! - `pull()`：从队列取出并输出（透传）
//!
//! 注意：
//! - side-sink 是同步 `AudioSink`，因此 `push()` 会把 side-sink 的开销计入当前节点（可能阻塞主链路）。
//!   若需要并行/异步写入，请传入带队列/后台线程的 sink（例如项目里的 async/parallel sink 或自行封装）。

use crate::codec::error::{CodecError, CodecResult};
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::{NodeBuffer, NodeBufferKind};
use crate::pipeline::node::static_node_interface::StaticNode;
use crate::pipeline::sink::audio_sink::AudioSink;
use crate::runner::error::RunnerError;
use std::collections::VecDeque;

fn map_sink_err(e: RunnerError) -> CodecError {
    CodecError::Other(format!("tap node sink error: {e}"))
}

/// 动态版 Tap 节点：可插入 `DynPipeline` / `AsyncDynPipeline`。
///
/// - `kind`：本节点接受/输出的 buffer kind（必须与相邻节点连接匹配）
/// - `sink`：旁路 sink，通常用 `AudioSink<In = NodeBuffer>`；若只想接 PCM，可传 `PcmSink(writer)`
pub struct TapNode<S>
where
    S: AudioSink<In = NodeBuffer> + Send,
{
    kind: NodeBufferKind,
    sink: S,
    q: VecDeque<NodeBuffer>,
    flushed: bool,
    finalized: bool,
}

impl<S> TapNode<S>
where
    S: AudioSink<In = NodeBuffer> + Send,
{
    pub fn new(kind: NodeBufferKind, sink: S) -> Self {
        Self {
            kind,
            sink,
            q: VecDeque::new(),
            flushed: false,
            finalized: false,
        }
    }

    /// 取回内部 sink（例如为了在外层检查统计信息）。
    ///
    /// 注意：若你在 flush 前取走 sink，将不会自动 finalize。
    pub fn into_sink(self) -> S {
        self.sink
    }
}

impl<S> DynNode for TapNode<S>
where
    S: AudioSink<In = NodeBuffer> + Send,
{
    fn name(&self) -> &'static str {
        "tap"
    }

    fn input_kind(&self) -> NodeBufferKind {
        self.kind
    }

    fn output_kind(&self) -> NodeBufferKind {
        self.kind
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("tap node already flushed"));
        }
        match input {
            None => {
                self.flushed = true;
                if !self.finalized {
                    self.sink.finalize().map_err(map_sink_err)?;
                    self.finalized = true;
                }
                Ok(())
            }
            Some(buf) => {
                if buf.kind() != self.kind {
                    return Err(CodecError::InvalidData("tap node buffer kind mismatch"));
                }
                self.sink.push(buf.clone()).map_err(map_sink_err)?;
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

    fn reset(&mut self) -> CodecResult<()> {
        // 语义：reset 只清内部队列/状态，不强行 finalize sink（让外部决定 sink 生命周期）。
        self.q.clear();
        self.flushed = false;
        self.finalized = false;
        Ok(())
    }
}

/// 静态版 Tap 节点：可插入 `Pipeline3` / `AsyncPipeline3`（要求 `T: Clone`）。
pub struct TapStaticNode<S, T>
where
    S: AudioSink<In = T>,
    T: Clone,
{
    sink: S,
    q: VecDeque<T>,
    flushed: bool,
    finalized: bool,
}

impl<S, T> TapStaticNode<S, T>
where
    S: AudioSink<In = T>,
    T: Clone,
{
    pub fn new(sink: S) -> Self {
        Self {
            sink,
            q: VecDeque::new(),
            flushed: false,
            finalized: false,
        }
    }

    pub fn into_sink(self) -> S {
        self.sink
    }
}

impl<S, T> StaticNode for TapStaticNode<S, T>
where
    S: AudioSink<In = T>,
    T: Clone,
{
    type In = T;
    type Out = T;

    fn name(&self) -> &'static str {
        "tap"
    }

    fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("tap node already flushed"));
        }
        match input {
            None => {
                self.flushed = true;
                if !self.finalized {
                    self.sink.finalize().map_err(map_sink_err)?;
                    self.finalized = true;
                }
                Ok(())
            }
            Some(v) => {
                self.sink.push(v.clone()).map_err(map_sink_err)?;
                self.q.push_back(v);
                Ok(())
            }
        }
    }

    fn pull(&mut self) -> CodecResult<Self::Out> {
        if let Some(v) = self.q.pop_front() {
            return Ok(v);
        }
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        self.q.clear();
        self.flushed = false;
        self.finalized = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::audio::audio::{AudioFormat, ChannelLayout, SampleFormat};

    struct CollectSink<T> {
        items: Vec<T>,
        finalized: bool,
    }

    impl<T> Default for CollectSink<T> {
        fn default() -> Self {
            Self {
                items: Vec::new(),
                finalized: false,
            }
        }
    }

    impl<T> AudioSink for CollectSink<T> {
        type In = T;

        fn name(&self) -> &'static str {
            "collect"
        }

        fn push(&mut self, input: Self::In) -> crate::runner::error::RunnerResult<()> {
            self.items.push(input);
            Ok(())
        }

        fn finalize(&mut self) -> crate::runner::error::RunnerResult<()> {
            self.finalized = true;
            Ok(())
        }
    }

    #[test]
    fn tap_node_passthrough_and_copy_to_sink() {
        let sink = CollectSink::<NodeBuffer>::default();
        let mut tap = TapNode::new(NodeBufferKind::Pcm, sink);

        let fmt = AudioFormat {
            sample_rate: 48_000,
            sample_format: SampleFormat::I16 { planar: false },
            channel_layout: ChannelLayout::stereo(),
        };
        let f = crate::common::audio::audio::AudioFrame::new_alloc(fmt, 480).unwrap();
        let b = NodeBuffer::Pcm(f);
        tap.push(Some(b.clone())).unwrap();

        // 下游透传
        let out = tap.pull().unwrap();
        assert_eq!(out.kind(), NodeBufferKind::Pcm);

        // sink 收到 clone
        let sink = tap.into_sink();
        assert_eq!(sink.items.len(), 1);
        assert_eq!(sink.items[0].kind(), NodeBufferKind::Pcm);
    }

    #[test]
    fn tap_node_finalize_on_flush() {
        let sink = CollectSink::<NodeBuffer>::default();
        let mut tap = TapNode::new(NodeBufferKind::Pcm, sink);
        tap.push(None).unwrap();
        let sink = tap.into_sink();
        assert!(sink.finalized);
    }
}

