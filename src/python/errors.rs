use crate::codec::error::CodecError;
use crate::common::audio::audio::AudioError;
use crate::common::io::AudioIOError;
use crate::runner::error::RunnerError;

use pyo3::exceptions::{PyBlockingIOError, PyEOFError, PyRuntimeError, PyValueError};
use pyo3::{PyErr, Python};

pub(crate) fn map_codec_err(e: CodecError) -> PyErr {
    match e {
        CodecError::InvalidData(msg) => PyValueError::new_err(msg),
        CodecError::InvalidState(msg) => PyRuntimeError::new_err(msg),
        CodecError::Unsupported(msg) => PyRuntimeError::new_err(msg),
        CodecError::Other(msg) => PyRuntimeError::new_err(msg),
        CodecError::Again => PyRuntimeError::new_err("codec again (EAGAIN)"),
        CodecError::Eof => PyRuntimeError::new_err("codec eof (EOF)"),
    }
}

pub(crate) fn map_audio_err(e: AudioError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub(crate) fn map_runner_err(e: RunnerError) -> PyErr {
    match e {
        RunnerError::Codec(ce) => map_codec_err(ce),
        RunnerError::Io(ioe) => PyRuntimeError::new_err(ioe.to_string()),
        RunnerError::InvalidData(msg) => PyValueError::new_err(msg),
        RunnerError::InvalidState(msg) => PyRuntimeError::new_err(msg),
    }
}

pub(crate) fn pyerr_to_runner_err(e: PyErr) -> RunnerError {
    RunnerError::Codec(CodecError::Other(e.to_string()))
}

pub(crate) fn pyerr_to_codec_err(e: PyErr) -> CodecError {
    // 允许 Python 用异常表达控制流：
    // - BlockingIOError => Again（暂无输出/背压）
    // - EOFError => Eof（结束）
    Python::with_gil(|py| {
        if e.is_instance_of::<PyBlockingIOError>(py) {
            return CodecError::Again;
        }
        if e.is_instance_of::<PyEOFError>(py) {
            return CodecError::Eof;
        }
        CodecError::Other(e.to_string())
    })
}

pub(crate) fn map_file_err(e: AudioIOError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}


