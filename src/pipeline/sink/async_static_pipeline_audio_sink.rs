//! Static convenience wrapper for the async pipeline sink (fixed 3 processor stages).

use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::AudioFrame;
use crate::common::io::io::AudioWriter;
use super::audio_sink::AsyncAudioSink;
use crate::runner::error::RunnerResult;
use async_trait::async_trait;
use crate::pipeline::sink::async_pipeline_audio_sink::AsyncPipelineAudioSink;

/// 静态版（固定 3 段 processor）：`P1 -> P2 -> P3 -> writer`。
pub struct AsyncPipelineAudioSink3<P1, P2, P3, W>
where
    P1: AudioProcessor + Send + 'static,
    P2: AudioProcessor + Send + 'static,
    P3: AudioProcessor + Send + 'static,
    W: AudioWriter + Send + 'static,
{
    inner: AsyncPipelineAudioSink,
    _phantom: core::marker::PhantomData<(P1, P2, P3, W)>,
}

impl<P1, P2, P3, W> AsyncPipelineAudioSink3<P1, P2, P3, W>
where
    P1: AudioProcessor + Send + 'static,
    P2: AudioProcessor + Send + 'static,
    P3: AudioProcessor + Send + 'static,
    W: AudioWriter + Send + 'static,
{
    pub fn new(p1: P1, p2: P2, p3: P3, writer: W, queue_capacity: usize) -> Self {
        let processors: Vec<Box<dyn AudioProcessor>> = vec![Box::new(p1), Box::new(p2), Box::new(p3)];
        let writer: Box<dyn AudioWriter + Send> = Box::new(writer);
        let inner = AsyncPipelineAudioSink::new(processors, writer, queue_capacity);
        Self {
            inner,
            _phantom: core::marker::PhantomData,
        }
    }

    pub fn with_default_capacity(p1: P1, p2: P2, p3: P3, writer: W) -> Self {
        Self::new(p1, p2, p3, writer, 8)
    }
}

#[async_trait]
impl<P1, P2, P3, W> AsyncAudioSink for AsyncPipelineAudioSink3<P1, P2, P3, W>
where
    P1: AudioProcessor + Send + 'static,
    P2: AudioProcessor + Send + 'static,
    P3: AudioProcessor + Send + 'static,
    W: AudioWriter + Send + 'static,
{
    type In = AudioFrame;

    fn name(&self) -> &'static str {
        "async-pipeline-audio-sink3"
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        self.inner.push(input).await
    }

    async fn finalize(&mut self) -> RunnerResult<()> {
        self.inner.finalize().await
    }
}

