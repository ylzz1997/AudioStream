use crate::common::audio::audio::{AudioFrame, AudioFrameView};
use crate::common::io::file::AudioFileResult;

/// 写端抽象：把 PCM 帧写入某种IO（内部可走任意 codec）。
pub trait AudioWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioFileResult<()>;
    fn finalize(&mut self) -> AudioFileResult<()>;
}

/// 读端抽象：从某种IO流式读出 PCM 帧。
pub trait AudioReader {
    fn next_frame(&mut self) -> AudioFileResult<Option<AudioFrame>>;
}