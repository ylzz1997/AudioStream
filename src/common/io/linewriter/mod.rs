//! Linear "processor chain -> writer" helpers.
//!
//! `LineAudioWriter` is a synchronous sink/writer adapter:
//! - It implements `AudioWriter` so it can be used wherever a writer is required.
//! - By the blanket impl in `runner/audio_sink.rs`, any `AudioWriter` is also an `AudioSink<In=AudioFrame>`.
//! - It supports an ordered list of `AudioProcessor` before the final `AudioWriter`.

pub mod line_audio_writer;

pub use line_audio_writer::LineAudioWriter;
