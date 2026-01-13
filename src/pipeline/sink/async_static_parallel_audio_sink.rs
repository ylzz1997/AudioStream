//! Static (fixed-arity) async parallel fan-out sink.
//!
//! Compared with `AsyncParallelAudioSink` (Vec-based), this version keeps sinks as concrete fields,
//! avoiding allocation and dynamic dispatch at the fan-out layer.
//!
//! Semantics:
//! - `push`: dispatches to both sinks concurrently and awaits both.
//! - `finalize`: finalizes both sinks concurrently and awaits both.
//! - Returns the first error (if any) but still waits for both sinks.

use super::audio_sink::AsyncAudioSink;
use crate::runner::error::RunnerResult;
use async_trait::async_trait;

/// Fixed 2-way async parallel sink fan-out.
pub struct AsyncStaticParallelAudioSink<S1, S2, In>
where
    In: Clone + Send + 'static,
    S1: AsyncAudioSink<In = In> + Send,
    S2: AsyncAudioSink<In = In> + Send,
{
    s1: S1,
    s2: S2,
    _phantom: core::marker::PhantomData<In>,
}

impl<S1, S2, In> AsyncStaticParallelAudioSink<S1, S2, In>
where
    In: Clone + Send + 'static,
    S1: AsyncAudioSink<In = In> + Send,
    S2: AsyncAudioSink<In = In> + Send,
{
    pub fn new(s1: S1, s2: S2) -> Self {
        Self {
            s1,
            s2,
            _phantom: core::marker::PhantomData,
        }
    }

    pub fn into_inner(self) -> (S1, S2) {
        (self.s1, self.s2)
    }

    pub fn sinks(&self) -> (&S1, &S2) {
        (&self.s1, &self.s2)
    }

    pub fn sinks_mut(&mut self) -> (&mut S1, &mut S2) {
        (&mut self.s1, &mut self.s2)
    }
}

#[async_trait]
impl<S1, S2, In> AsyncAudioSink for AsyncStaticParallelAudioSink<S1, S2, In>
where
    In: Clone + Send + 'static,
    S1: AsyncAudioSink<In = In> + Send,
    S2: AsyncAudioSink<In = In> + Send,
{
    type In = In;

    fn name(&self) -> &'static str {
        "async-static-parallel-audio-sink"
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        // Borrow two distinct fields and run concurrently.
        let s1 = &mut self.s1;
        let s2 = &mut self.s2;
        let (r1, r2) = tokio::join!(s1.push(input.clone()), s2.push(input));

        // Return the first error (if any), but both futures have completed.
        r1?;
        r2?;
        Ok(())
    }

    async fn finalize(&mut self) -> RunnerResult<()> {
        let s1 = &mut self.s1;
        let s2 = &mut self.s2;
        let (r1, r2) = tokio::join!(s1.finalize(), s2.finalize());
        r1?;
        r2?;
        Ok(())
    }
}

