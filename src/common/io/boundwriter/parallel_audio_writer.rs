use crate::common::audio::audio::AudioFrameView;
use crate::common::io::io::{AudioIOError, AudioIOResult, AudioWriter};
use std::thread;

use super::MultiAudioWriter;

/// A multi-writer that fans out each frame to all bound writers in parallel.
///
/// Semantics:
/// - `write_frame`: dispatches to all writers concurrently and waits for all to finish.
///   Returns the first error (if any) but still waits all writers.
/// - `finalize`: finalizes all writers concurrently and waits for all to finish.
pub struct ParallelAudioWriter {
    writers: Vec<Box<dyn AudioWriter + Send>>,
}

impl ParallelAudioWriter {
    pub fn new() -> Self {
        Self { writers: vec![] }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            writers: Vec::with_capacity(cap),
        }
    }

    pub fn from_writers(writers: Vec<Box<dyn AudioWriter + Send>>) -> Self {
        Self { writers }
    }

    pub fn into_writers(self) -> Vec<Box<dyn AudioWriter + Send>> {
        self.writers
    }
}

impl Default for ParallelAudioWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiAudioWriter for ParallelAudioWriter {
    fn bind(&mut self, w: Box<dyn AudioWriter + Send>) {
        self.writers.push(w);
    }

    fn len(&self) -> usize {
        self.writers.len()
    }
}

impl AudioWriter for ParallelAudioWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        if self.writers.is_empty() {
            return Ok(());
        }

        // Move writers into threads to satisfy the borrow checker (no overlapping &mut borrows).
        let writers = std::mem::take(&mut self.writers);
        let mut first_err: Option<AudioIOError> = None;

        let writers = thread::scope(|s| {
            let mut handles = Vec::with_capacity(writers.len());
            for mut w in writers {
                let f = frame;
                handles.push(s.spawn(move || {
                    let r = w.write_frame(f);
                    (w, r)
                }));
            }

            let mut out: Vec<Box<dyn AudioWriter + Send>> = Vec::with_capacity(handles.len());
            for h in handles {
                match h.join() {
                    Ok((w, r)) => {
                        if first_err.is_none() {
                            if let Err(e) = r {
                                first_err = Some(e);
                            }
                        }
                        out.push(w);
                    }
                    Err(_) => {
                        if first_err.is_none() {
                            first_err = Some(AudioIOError::Format("ParallelAudioWriter worker panicked"));
                        }
                        // If a worker panicked, we can't recover the writer instance.
                        // Drop it to keep invariants sane.
                    }
                }
            }
            out
        });

        self.writers = writers;
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        if self.writers.is_empty() {
            return Ok(());
        }

        let writers = std::mem::take(&mut self.writers);
        let mut first_err: Option<AudioIOError> = None;

        let writers = thread::scope(|s| {
            let mut handles = Vec::with_capacity(writers.len());
            for mut w in writers {
                handles.push(s.spawn(move || {
                    let r = w.finalize();
                    (w, r)
                }));
            }

            let mut out: Vec<Box<dyn AudioWriter + Send>> = Vec::with_capacity(handles.len());
            for h in handles {
                match h.join() {
                    Ok((w, r)) => {
                        if first_err.is_none() {
                            if let Err(e) = r {
                                first_err = Some(e);
                            }
                        }
                        out.push(w);
                    }
                    Err(_) => {
                        if first_err.is_none() {
                            first_err = Some(AudioIOError::Format("ParallelAudioWriter worker panicked"));
                        }
                    }
                }
            }
            out
        });

        self.writers = writers;
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

