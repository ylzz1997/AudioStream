use crate::common::audio::audio::{AudioFrame, AudioFrameView};
use crate::codec::error::CodecError;

/// 写端抽象：把 PCM 帧写入某种IO（内部可走任意 codec）。
pub trait AudioWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()>;
    fn finalize(&mut self) -> AudioIOResult<()>;
}

/// 读端抽象：从某种IO流式读出 PCM 帧。
pub trait AudioReader {
    fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>>;
}

#[derive(Debug)]
pub enum AudioIOError {
    Io(std::io::Error),
    Codec(CodecError),
    Format(&'static str),
}

pub type AudioIOResult<T> = Result<T, AudioIOError>;