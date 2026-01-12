//! Multi-writer helpers.

pub mod multi_audio_writer;
pub mod parallel_audio_writer;

pub use multi_audio_writer::MultiAudioWriter;
pub use parallel_audio_writer::ParallelAudioWriter;

