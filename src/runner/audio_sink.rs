use crate::common::audio::audio::AudioFrame;
use crate::common::audio::audio::AudioFrameView;
use crate::common::io::io::AudioWriter;
use crate::pipeline::node::node_interface::NodeBuffer;
use crate::runner::error::{RunnerError, RunnerResult};
use async_trait::async_trait;

/// 音频汇（只进）：Runner 会把末端输出写入 Sink，最后调用 finalize。
pub trait AudioSink {
    type In;

    fn name(&self) -> &'static str {
        "audio-sink"
    }

    fn push(&mut self, input: Self::In) -> RunnerResult<()>;
    fn finalize(&mut self) -> RunnerResult<()>;
}

/// 异步音频汇（只进）：允许在 tokio runtime 中 `await` 写入与 finalize。
#[async_trait]
pub trait AsyncAudioSink: Send {
    type In: Send + 'static;

    fn name(&self) -> &'static str {
        "async-audio-sink"
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()>;
    async fn finalize(&mut self) -> RunnerResult<()>;
}

/// 同步 sink 自动适配为 async sink（在 async 上下文里直接同步执行）。
#[async_trait]
impl<T> AsyncAudioSink for T
where
    T: AudioSink + Send,
    T::In: Send + 'static,
{
    type In = T::In;

    fn name(&self) -> &'static str {
        AudioSink::name(self)
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        AudioSink::push(self, input)
    }

    async fn finalize(&mut self) -> RunnerResult<()> {
        AudioSink::finalize(self)
    }
}

/// 让现有的 `AudioWriter` 自动成为 `AudioSink<In = AudioFrame>`。
impl<T> AudioSink for T
where
    T: AudioWriter,
{
    type In = AudioFrame;

    fn name(&self) -> &'static str {
        "audio-writer"
    }

    fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        self.write_frame(&input as &dyn AudioFrameView)?;
        Ok(())
    }

    fn finalize(&mut self) -> RunnerResult<()> {
        Ok(AudioWriter::finalize(self)?)
    }
}

/// 把 `AudioSink<In = AudioFrame>` 适配为 `AudioSink<In = NodeBuffer>`（仅支持 PCM）。
pub struct PcmSink<S> {
    inner: S,
}

impl<S> PcmSink<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S> AudioSink for PcmSink<S>
where
    S: AudioSink<In = AudioFrame>,
{
    type In = NodeBuffer;

    fn name(&self) -> &'static str {
        "pcm-sink"
    }

    fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        match input {
            NodeBuffer::Pcm(f) => self.inner.push(f),
            NodeBuffer::Packet(_) => Err(RunnerError::InvalidData("pcm sink cannot accept Packet")),
        }
    }

    fn finalize(&mut self) -> RunnerResult<()> {
        self.inner.finalize()
    }
}


