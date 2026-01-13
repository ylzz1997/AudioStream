//! Async "pipeline + sink" implementation:
//! - Each `AudioProcessor` runs in its own blocking stage (pipeline parallel).
//! - The final `AudioWriter` is the last stage (single-threaded write).
//! - All stage channels are bounded, providing backpressure to upstream.
//!
//! This is designed to be used as an `AsyncAudioSink<In=AudioFrame>`.
//!
//! Notes:
//! - Stages are implemented with `tokio::task::spawn_blocking` because most codec / IO work is blocking.
//! - Errors are recorded as strings and surfaced as `CodecError::Other(...)` to avoid `Clone` requirements.
//! - This module intentionally does NOT implement `reset()` semantics (unlike `AsyncPipeline3`).
//!   If you need reset, we can extend the message protocol similarly to pipeline nodes.

use crate::codec::error::CodecError;
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFrame, AudioFrameView};
use crate::common::io::io::AudioWriter;
use super::audio_sink::AsyncAudioSink;
use crate::runner::error::{RunnerError, RunnerResult};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

#[derive(Debug)]
enum Msg<T> {
    Data(T),
    Flush,
}

fn store_err(store: &Arc<Mutex<Option<String>>>, msg: String) {
    let mut g = store.lock().expect("err_store poisoned");
    if g.is_none() {
        *g = Some(msg);
    }
}

fn err_or_closed(store: &Arc<Mutex<Option<String>>>) -> RunnerError {
    let msg = store
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .unwrap_or_else(|| "async pipeline audio sink channel closed".to_string());
    RunnerError::Codec(CodecError::Other(msg))
}

fn send_and_drain_processor(
    p: &mut dyn AudioProcessor,
    input: Option<AudioFrame>,
    tx_next: &mpsc::Sender<Msg<AudioFrame>>,
    err_store: &Arc<Mutex<Option<String>>>,
) -> Result<(), ()> {
    // Helper: drain as much output as possible to next stage.
    fn drain(
        p: &mut dyn AudioProcessor,
        tx_next: &mpsc::Sender<Msg<AudioFrame>>,
        err_store: &Arc<Mutex<Option<String>>>,
    ) -> Result<(), ()> {
        loop {
            match p.receive_frame() {
                Ok(out) => {
                    if tx_next.blocking_send(Msg::Data(out)).is_err() {
                        store_err(err_store, "next stage channel closed".to_string());
                        return Err(());
                    }
                }
                Err(CodecError::Again) => return Ok(()),
                Err(CodecError::Eof) => return Ok(()),
                Err(e) => {
                    store_err(err_store, format!("processor {} pull error: {e}", p.name()));
                    return Err(());
                }
            }
        }
    }

    // Convert owned frame to view for send_frame.
    let input = input;
    loop {
        let view: Option<&dyn AudioFrameView> = input.as_ref().map(|f| f as &dyn AudioFrameView);
        match p.send_frame(view) {
            Ok(()) => break,
            Err(CodecError::Again) => {
                if drain(p, tx_next, err_store).is_err() {
                    return Err(());
                }
                // retry send_frame with the same input
                continue;
            }
            Err(CodecError::Eof) => {
                // Treat as "no more output from this stage"; still propagate flush downstream.
                return Ok(());
            }
            Err(e) => {
                store_err(err_store, format!("processor {} push error: {e}", p.name()));
                return Err(());
            }
        }
    }

    drain(p, tx_next, err_store)
}

fn flush_processor(
    p: &mut dyn AudioProcessor,
    tx_next: &mpsc::Sender<Msg<AudioFrame>>,
    err_store: &Arc<Mutex<Option<String>>>,
) -> Result<(), ()> {
    // flush: send_frame(None) then pull until Eof
    loop {
        match p.send_frame(None) {
            Ok(()) => break,
            Err(CodecError::Again) => {
                // Need to drain outputs first
                if send_and_drain_processor(p, None, tx_next, err_store).is_err() {
                    return Err(());
                }
            }
            Err(e) => {
                store_err(err_store, format!("processor {} flush push error: {e}", p.name()));
                return Err(());
            }
        }
    }

    loop {
        match p.receive_frame() {
            Ok(out) => {
                if tx_next.blocking_send(Msg::Data(out)).is_err() {
                    store_err(err_store, "next stage channel closed".to_string());
                    return Err(());
                }
            }
            Err(CodecError::Again) => continue,
            Err(CodecError::Eof) => break,
            Err(e) => {
                store_err(err_store, format!("processor {} flush pull error: {e}", p.name()));
                return Err(());
            }
        }
    }
    Ok(())
}

fn spawn_processor_stage(
    mut p: Box<dyn AudioProcessor>,
    mut rx: mpsc::Receiver<Msg<AudioFrame>>,
    tx_next: mpsc::Sender<Msg<AudioFrame>>,
    err_store: Arc<Mutex<Option<String>>>,
) {
    tokio::task::spawn_blocking(move || {
        let mut flushed = false;
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                Msg::Data(frame) => {
                    if flushed {
                        store_err(&err_store, format!("processor {} received data after flush", p.name()));
                        break;
                    }
                    if send_and_drain_processor(p.as_mut(), Some(frame), &tx_next, &err_store).is_err() {
                        break;
                    }
                }
                Msg::Flush => {
                    flushed = true;
                    let _ = flush_processor(p.as_mut(), &tx_next, &err_store);
                    let _ = tx_next.blocking_send(Msg::Flush);
                    break;
                }
            }
        }

        // Upstream closed without explicit flush: treat as flush once.
        if !flushed {
            let _ = flush_processor(p.as_mut(), &tx_next, &err_store);
            let _ = tx_next.blocking_send(Msg::Flush);
        }
    });
}

fn spawn_writer_stage(
    mut w: Box<dyn AudioWriter + Send>,
    mut rx: mpsc::Receiver<Msg<AudioFrame>>,
    err_store: Arc<Mutex<Option<String>>>,
) -> tokio::task::JoinHandle<RunnerResult<()>> {
    tokio::task::spawn_blocking(move || {
        let mut flushed = false;
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                Msg::Data(frame) => {
                    if flushed {
                        store_err(&err_store, "writer received data after flush".to_string());
                        break;
                    }
                    w.write_frame(&frame as &dyn AudioFrameView)
                        .map_err(RunnerError::from)?;
                }
                Msg::Flush => {
                    flushed = true;
                    break;
                }
            }
        }

        // Either got Flush, or upstream closed: finalize once.
        if !flushed {
            // upstream closed without an explicit flush
        }
        w.finalize().map_err(RunnerError::from)?;
        Ok(())
    })
}

/// 动态版：`processors (PCM->PCM)* -> writer`，每个 processor 作为一个 stage 并行跑。
///
/// - 输入：`AudioFrame`
/// - 背压：所有 stage channel 都是 bounded；当末端写入慢时，上游 `push().await` 会被阻塞（理想的 backpressure）。
pub struct AsyncPipelineAudioSink {
    in_tx: mpsc::Sender<Msg<AudioFrame>>,
    join: Option<tokio::task::JoinHandle<RunnerResult<()>>>,
    err_store: Arc<Mutex<Option<String>>>,
    finalized: bool,
}

impl AsyncPipelineAudioSink {
    /// 构造一个 pipeline sink。
    ///
    /// - `queue_capacity`: 每个 stage 之间的有界队列容量（>=1）。越大吞吐越高、延迟/内存越大。
    pub fn new(
        processors: Vec<Box<dyn AudioProcessor>>,
        writer: Box<dyn AudioWriter + Send>,
        queue_capacity: usize,
    ) -> Self {
        let cap = queue_capacity.max(1);
        let err_store = Arc::new(Mutex::new(None));

        // Build stage channels
        let (in_tx, in_rx) = mpsc::channel::<Msg<AudioFrame>>(cap);
        let mut prev_rx = in_rx;

        // For each processor, create a new stage.
        for p in processors.into_iter() {
            let (tx_next, rx_next) = mpsc::channel::<Msg<AudioFrame>>(cap);
            let rx = prev_rx;
            spawn_processor_stage(p, rx, tx_next, err_store.clone());
            prev_rx = rx_next;
        }

        // Last stage: writer
        let join = spawn_writer_stage(writer, prev_rx, err_store.clone());

        Self {
            in_tx,
            join: Some(join),
            err_store,
            finalized: false,
        }
    }

    pub fn with_default_capacity(processors: Vec<Box<dyn AudioProcessor>>, writer: Box<dyn AudioWriter + Send>) -> Self {
        Self::new(processors, writer, 8)
    }
}

#[async_trait]
impl AsyncAudioSink for AsyncPipelineAudioSink {
    type In = AudioFrame;

    fn name(&self) -> &'static str {
        "async-pipeline-audio-sink"
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        if self.finalized {
            return Err(RunnerError::Codec(CodecError::InvalidState(
                "async pipeline audio sink already finalized",
            )));
        }
        self.in_tx
            .send(Msg::Data(input))
            .await
            .map_err(|_| err_or_closed(&self.err_store))?;
        Ok(())
    }

    async fn finalize(&mut self) -> RunnerResult<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        self.in_tx
            .send(Msg::Flush)
            .await
            .map_err(|_| err_or_closed(&self.err_store))?;

        let Some(join) = self.join.take() else {
            return Ok(());
        };
        match join.await {
            Ok(r) => r,
            Err(e) => Err(RunnerError::Codec(CodecError::Other(format!(
                "async pipeline audio sink join failed: {e}"
            )))),
        }
    }
}

impl Drop for AsyncPipelineAudioSink {
    fn drop(&mut self) {
        if let Some(j) = &self.join {
            j.abort();
        }
    }
}

