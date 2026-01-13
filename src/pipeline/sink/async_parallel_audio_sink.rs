//! Async parallel fan-out sink.
//!
//! Similar to `common/io/boundwriter/parallel_audio_writer.rs`, but for async sinks:
//! - `push`: dispatches to all bound sinks concurrently and awaits all.
//! - `finalize`: finalizes all sinks concurrently and awaits all.
//! - Returns the first error (if any) but still waits for all sinks.
//!
//! Important:
//! - `In` must be `Clone` so the same input can be fanned out.

use super::audio_sink::AsyncAudioSink;
use crate::runner::error::{RunnerError, RunnerResult};
use async_trait::async_trait;

/// Fan-out each input to all bound sinks in parallel (async).
pub struct AsyncParallelAudioSink<In>
where
    In: Clone + Send + 'static,
{
    sinks: Vec<Box<dyn AsyncAudioSink<In = In> + Send>>,
}

impl<In> AsyncParallelAudioSink<In>
where
    In: Clone + Send + 'static,
{
    pub fn new() -> Self {
        Self { sinks: vec![] }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            sinks: Vec::with_capacity(cap),
        }
    }

    pub fn bind(&mut self, s: Box<dyn AsyncAudioSink<In = In> + Send>) {
        self.sinks.push(s);
    }

    pub fn extend(&mut self, ss: impl IntoIterator<Item = Box<dyn AsyncAudioSink<In = In> + Send>>) {
        self.sinks.extend(ss);
    }

    pub fn len(&self) -> usize {
        self.sinks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sinks.is_empty()
    }

    pub fn into_sinks(self) -> Vec<Box<dyn AsyncAudioSink<In = In> + Send>> {
        self.sinks
    }
}

impl<In> Default for AsyncParallelAudioSink<In>
where
    In: Clone + Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<In> AsyncAudioSink for AsyncParallelAudioSink<In>
where
    In: Clone + Send + 'static,
{
    type In = In;

    fn name(&self) -> &'static str {
        "async-parallel-audio-sink"
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        if self.sinks.is_empty() {
            return Ok(());
        }

        // Move sinks into tasks to satisfy the borrow checker (no overlapping &mut borrows across await).
        let sinks = std::mem::take(&mut self.sinks);
        let mut first_err: Option<RunnerError> = None;

        let mut handles = Vec::with_capacity(sinks.len());
        for mut s in sinks {
            let v = input.clone();
            handles.push(tokio::spawn(async move {
                let r = s.push(v).await;
                (s, r)
            }));
        }

        let mut out: Vec<Box<dyn AsyncAudioSink<In = In> + Send>> = Vec::with_capacity(handles.len());
        for h in handles {
            match h.await {
                Ok((s, r)) => {
                    if first_err.is_none() {
                        if let Err(e) = r {
                            first_err = Some(e);
                        }
                    }
                    out.push(s);
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(RunnerError::InvalidState("AsyncParallelAudioSink worker panicked"));
                    }
                    // If a worker panicked, we can't recover the sink instance. Drop it.
                }
            }
        }

        self.sinks = out;
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    async fn finalize(&mut self) -> RunnerResult<()> {
        if self.sinks.is_empty() {
            return Ok(());
        }

        let sinks = std::mem::take(&mut self.sinks);
        let mut first_err: Option<RunnerError> = None;

        let mut handles = Vec::with_capacity(sinks.len());
        for mut s in sinks {
            handles.push(tokio::spawn(async move {
                let r = s.finalize().await;
                (s, r)
            }));
        }

        let mut out: Vec<Box<dyn AsyncAudioSink<In = In> + Send>> = Vec::with_capacity(handles.len());
        for h in handles {
            match h.await {
                Ok((s, r)) => {
                    if first_err.is_none() {
                        if let Err(e) = r {
                            first_err = Some(e);
                        }
                    }
                    out.push(s);
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(RunnerError::InvalidState("AsyncParallelAudioSink worker panicked"));
                    }
                }
            }
        }

        self.sinks = out;
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

