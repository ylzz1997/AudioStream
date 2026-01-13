//! Fully static (compile-time typed) async pipeline -> writer sink (fixed 3 stages).
//!
//! This is the "static typed" counterpart of `AsyncPipelineAudioSink`:
//! - Uses `StaticNode` with associated types `In/Out` to enforce connectivity at compile time.
//! - Allows mixing processor/encoder/decoder nodes, as long as adjacent `Out == In`.
//! - Requires the **last** node output to be `AudioFrame` (PCM), because the writer consumes PCM.
//! - Runs the 3 stages in parallel via the existing tokio static pipeline implementation.

use crate::codec::error::CodecError;
use crate::common::audio::audio::{AudioFrame, AudioFrameView};
use crate::common::io::io::AudioWriter;
use crate::pipeline::node::async_static_node_interface::AsyncPipeline3;
use crate::pipeline::node::static_node_interface::StaticNode;
use super::audio_sink::AsyncAudioSink;
use crate::runner::error::{RunnerError, RunnerResult};
use async_trait::async_trait;
use tokio::sync::mpsc;
use core::marker::PhantomData;

/// Fully static 3-stage pipeline sink: `N1 -> N2 -> N3 -> writer(PCM)`.
///
/// Compile-time guarantees:
/// - `N2::In == N1::Out`
/// - `N3::In == N2::Out`
/// - `N3::Out == AudioFrame` (PCM)
pub struct AsyncPipelineAudioSink3<N1, N2, N3, W>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out, Out = AudioFrame> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    W: AudioWriter + Send + 'static,
{
    tx: Option<<AsyncPipeline3<N1, N2, N3> as crate::pipeline::node::node_interface::AsyncPipelineEndpoint>::Producer>,
    out_task: Option<tokio::task::JoinHandle<RunnerResult<()>>>,
    writer_task: Option<tokio::task::JoinHandle<RunnerResult<()>>>,
    finalized: bool,
    _phantom: PhantomData<W>,
}

impl<N1, N2, N3, W> AsyncPipelineAudioSink3<N1, N2, N3, W>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out, Out = AudioFrame> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    W: AudioWriter + Send + 'static,
{
    /// `queue_capacity` controls the bounded queue between pipeline output and the writer worker.
    pub fn new(n1: N1, n2: N2, n3: N3, mut writer: W, queue_capacity: usize) -> Self {
        let cap = queue_capacity.max(1);

        let pipeline = AsyncPipeline3::new(n1, n2, n3);
        let (tx, mut rx) = pipeline.endpoints();

        // Bounded queue to apply backpressure from writer to pipeline output task.
        let (wtx, mut wrx) = mpsc::channel::<AudioFrame>(cap);

        // Writer worker (blocking).
        let writer_task = tokio::task::spawn_blocking(move || {
            while let Some(frame) = wrx.blocking_recv() {
                writer
                    .write_frame(&frame as &dyn AudioFrameView)
                    .map_err(RunnerError::from)?;
            }
            writer.finalize().map_err(RunnerError::from)?;
            Ok::<(), RunnerError>(())
        });

        // Drain pipeline outputs -> writer queue.
        let out_task = tokio::spawn(async move {
            loop {
                match rx.get_frame().await {
                    Ok(frame) => {
                        wtx.send(frame)
                            .await
                            .map_err(|_| RunnerError::InvalidState("writer queue closed"))?;
                    }
                    Err(CodecError::Again) => continue,
                    Err(CodecError::Eof) => break,
                    Err(e) => return Err::<(), RunnerError>(e.into()),
                }
            }
            // Close writer queue.
            drop(wtx);
            Ok::<(), RunnerError>(())
        });

        Self {
            tx: Some(tx),
            out_task: Some(out_task),
            writer_task: Some(writer_task),
            finalized: false,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<N1, N2, N3, W> AsyncAudioSink for AsyncPipelineAudioSink3<N1, N2, N3, W>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out, Out = AudioFrame> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    W: AudioWriter + Send + 'static,
{
    type In = N1::In;

    fn name(&self) -> &'static str {
        "async-static-pipeline-audio-sink3"
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        if self.finalized {
            return Err(RunnerError::InvalidState("async static pipeline sink already finalized"));
        }
        let Some(tx) = &self.tx else {
            return Err(RunnerError::InvalidState("async static pipeline sink already closed"));
        };
        tx.push_frame(input).map_err(RunnerError::from)
    }

    async fn finalize(&mut self) -> RunnerResult<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        if let Some(tx) = self.tx.take() {
            tx.flush().map_err(RunnerError::from)?;
        }

        if let Some(t) = self.out_task.take() {
            match t.await {
                Ok(r) => r?,
                Err(_) => return Err(RunnerError::InvalidState("output task join failed")),
            }
        }

        if let Some(t) = self.writer_task.take() {
            match t.await {
                Ok(r) => r?,
                Err(_) => return Err(RunnerError::InvalidState("writer task join failed")),
            }
        }

        Ok(())
    }
}

impl<N1, N2, N3, W> Drop for AsyncPipelineAudioSink3<N1, N2, N3, W>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out, Out = AudioFrame> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    W: AudioWriter + Send + 'static,
{
    fn drop(&mut self) {
        if let Some(t) = &self.out_task {
            t.abort();
        }
        if let Some(t) = &self.writer_task {
            t.abort();
        }
    }
}

