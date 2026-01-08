//! FFmpeg backend 共享工具（仅在 `feature="ffmpeg"` 启用时编译）。

#[cfg(feature = "ffmpeg")]
extern crate ffmpeg_sys_next as ff;

#[cfg(feature = "ffmpeg")]
use crate::codec::error::CodecError;

#[cfg(feature = "ffmpeg")]
use libc;

#[cfg(feature = "ffmpeg")]
use std::ffi::CStr;

/// 将 FFmpeg 错误码转换为可读字符串。
#[cfg(feature = "ffmpeg")]
pub fn ff_err_to_string(err: i32) -> String {
    let mut buf = [0u8; 256];
    unsafe {
        ff::av_strerror(err, buf.as_mut_ptr() as *mut i8, buf.len());
    }
    let cstr = match CStr::from_bytes_until_nul(&buf) {
        Ok(s) => s,
        Err(_) => return format!("ffmpeg error {err}"),
    };
    cstr.to_string_lossy().into_owned()
}

/// 将 FFmpeg API 返回的错误码映射为本项目统一的 `CodecError`。
///
/// 约定：
/// - EAGAIN/EWOULDBLOCK => `Again`
/// - AVERROR_EOF        => `Eof`
/// - 其它              => `Other(<string>)`
#[cfg(feature = "ffmpeg")]
pub fn map_ff_err(err: i32) -> CodecError {
    #[cfg(unix)]
    let is_again = err == ff::AVERROR(libc::EAGAIN) || err == ff::AVERROR(libc::EWOULDBLOCK);
    #[cfg(windows)]
    let is_again = err == ff::AVERROR(libc::WSAEWOULDBLOCK);

    if is_again {
        return CodecError::Again;
    }
    if err == ff::AVERROR_EOF {
        return CodecError::Eof;
    }
    CodecError::Other(ff_err_to_string(err))
}


