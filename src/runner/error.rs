use crate::codec::error::CodecError;
use crate::common::io::file::AudioFileError;
use core::fmt;

#[derive(Debug)]
pub enum RunnerError {
    Codec(CodecError),
    Io(AudioFileError),
    InvalidData(&'static str),
    InvalidState(&'static str),
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunnerError::Codec(e) => write!(f, "codec error: {e}"),
            RunnerError::Io(e) => write!(f, "io error: {e}"),
            RunnerError::InvalidData(msg) => write!(f, "invalid data: {msg}"),
            RunnerError::InvalidState(msg) => write!(f, "invalid state: {msg}"),
        }
    }
}

impl std::error::Error for RunnerError {}

impl From<CodecError> for RunnerError {
    fn from(e: CodecError) -> Self {
        RunnerError::Codec(e)
    }
}

impl From<AudioFileError> for RunnerError {
    fn from(e: AudioFileError) -> Self {
        RunnerError::Io(e)
    }
}

pub type RunnerResult<T> = Result<T, RunnerError>;


