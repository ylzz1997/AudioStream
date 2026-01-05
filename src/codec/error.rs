// Codec Error Types
use core::fmt;

/// - `Again`: 类似 `EAGAIN`，表示需要先调用对端的 receive_*（或再 send_*）推进状态机
/// - `Eof`:  类似 `EOF`，表示 flush 后已经无更多输出
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    Again,
    Eof,

    InvalidState(&'static str),
    InvalidData(&'static str),
    Unsupported(&'static str),

    Other(String),
}

impl CodecError {
    pub const fn is_again(&self) -> bool {
        matches!(self, CodecError::Again)
    }

    pub const fn is_eof(&self) -> bool {
        matches!(self, CodecError::Eof)
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::Again => write!(f, "again (EAGAIN): need to drive codec state machine"),
            CodecError::Eof => write!(f, "end of stream (EOF)"),
            CodecError::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            CodecError::InvalidData(msg) => write!(f, "invalid data: {msg}"),
            CodecError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            CodecError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for CodecError {}

pub type CodecResult<T> = Result<T, CodecError>;


