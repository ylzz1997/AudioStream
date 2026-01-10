//! FFmpeg backend 共享工具（仅在 `feature="ffmpeg"` 启用时编译）。

#[cfg(feature = "ffmpeg")]
extern crate ffmpeg_sys_next as ff;

#[cfg(feature = "ffmpeg")]
use crate::codec::error::CodecError;

#[cfg(feature = "ffmpeg")]
use libc;

#[cfg(feature = "ffmpeg")]
use std::ffi::CStr;

#[cfg(feature = "ffmpeg")]
use crate::common::audio::audio::ChannelLayout;

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

/// 尝试从 FFmpeg 的 `AVChannelLayout` 提取 channel layout mask。
///
/// 说明（FFmpeg 6+）：
/// - `AVChannelLayout.order == AV_CHANNEL_ORDER_NATIVE` 时，`u.mask` 才有意义
/// - 其它 order（例如 UNSPEC / CUSTOM / AMBISONIC）可能没有 mask，或需要解析 `u.map`
///   目前我们只在能拿到 native mask 时使用它；否则退回到“按声道数的默认布局”（1/2ch）或 unspecified。
#[cfg(feature = "ffmpeg")]
pub fn channel_layout_from_av(ch: &ff::AVChannelLayout) -> ChannelLayout {
    let channels = ch.nb_channels.max(0) as u16;

    // 安全兜底：channels=0 也别 panic
    if channels == 0 {
        return ChannelLayout::unspecified(0);
    }

    // 只有 NATIVE 才能解 u.mask。
    //
    // 这里不直接引用 `AV_CHANNEL_ORDER_NATIVE` 常量名：
    // ffmpeg-sys-next 的 bindings 会在 build 时生成到 OUT_DIR，不同平台/版本下符号名可能不同。
    // FFmpeg 头文件里 AVChannelOrder 的值序通常是：
    //   0=UNSPEC, 1=NATIVE, 2=CUSTOM, 3=AMBISONIC
    let order = ch.order as i32;
    let is_native = order == 1;
    if is_native {
        // bindgen union 读取需要 unsafe
        let mask = unsafe { ch.u.mask };
        if mask != 0 {
            return ChannelLayout { channels, mask };
        }
    }

    ChannelLayout::default_for_channels(channels)
}


