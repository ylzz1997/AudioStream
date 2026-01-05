//! AAC 编码器实现（流式 send_frame/receive_packet）。

use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrameView, Rational};

/// AAC 编码配置（最小集合，后续可以补 profile/vbr/tune 等）。
#[derive(Clone, Debug)]
pub struct AacEncoderConfig {
    pub input_format: AudioFormat,
    pub bitrate: Option<u32>,
}

impl AacEncoderConfig {
    pub fn new(input_format: AudioFormat) -> Self {
        Self {
            input_format,
            bitrate: None,
        }
    }
}

/// AAC 编码器。
///
/// - `not(feature="ffmpeg")`：占位实现（无外部依赖）
/// - `feature="ffmpeg"`：FFmpeg backend（libavcodec）
#[cfg(not(feature = "ffmpeg"))]
pub struct AacEncoder {
    cfg: AacEncoderConfig,
    flushed: bool,
}


#[cfg(not(feature = "ffmpeg"))]
impl AacEncoder {
    pub fn new(cfg: AacEncoderConfig) -> CodecResult<Self> {
        // 这里可以做一些格式校验（比如 sample_rate/channels > 0）
        Ok(Self { cfg, flushed: false })
    }

    /// AAC 的 AudioSpecificConfig（ASC，常用于解码初始化/封装）。
    ///
    /// 占位实现下返回 None。
    pub fn audio_specific_config(&self) -> Option<Vec<u8>> {
        None
    }

    fn unsupported() -> CodecError {
        CodecError::Unsupported("AAC encoder backend not linked (enable FFmpeg backend in your environment)")
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioEncoder for AacEncoder {
    fn name(&self) -> &'static str {
        "aac(placeholder)"
    }

    fn input_format(&self) -> Option<AudioFormat> {
        Some(self.cfg.input_format)
    }

    fn preferred_frame_samples(&self) -> Option<usize> {
        // AAC-LC 常见每帧 1024 samples（每声道）；LD 可能是 512 等。
        // 真接 FFmpeg 时可从 codec context/codec capabilities 读到精确信息。
        Some(1024)
    }

    fn lookahead_samples(&self) -> usize {
        // AAC filterbank 有重叠，但通常由编码器内部处理；这里给上层一个“存在延迟”的信号位。
        // 真接 FFmpeg 时可以填 codec delay / encoder delay。
        0
    }

    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        if frame.is_none() {
            self.flushed = true;
            // flush 成功：后续 receive_packet 应该最终返回 Eof
            return Ok(());
        }
        Err(Self::unsupported())
    }

    fn receive_packet(&mut self) -> CodecResult<CodecPacket> {
        if self.flushed {
            return Err(CodecError::Eof);
        }

        // 占位：真实实现里这里会从编码器状态机取一个 AVPacket
        // 并映射到 CodecPacket（设置 time_base/pts/duration 等）。
        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        self.flushed = false;
        Ok(())
    }
}

pub fn default_aac_packet_time_base(sample_rate: u32) -> Rational {
    Rational::new(1, sample_rate as i32)
}


#[cfg(feature = "ffmpeg")]
mod ffmpeg_backend {
    use super::*;
    use core::ptr;
    use std::ffi::{CStr, CString};

    extern crate ffmpeg_sys_next as ff;
    use crate::common::audio::audio::SampleFormat;
    use libc;

    /// AAC 编码器（FFmpeg backend）。
    pub struct AacEncoder {
        cfg: AacEncoderConfig,
        flushed: bool,
        ctx: *mut ff::AVCodecContext,
    }

    // 我们把 FFmpeg codec context 封装在本类型内部，并且不在多线程间共享同一个实例。
    // 因此在语义上是 Send 的（但 Rust 无法自动推断 raw pointer 的 Send）。
    unsafe impl Send for AacEncoder {}

    fn tb_from_avr(tb: ff::AVRational) -> Rational {
        Rational::new(tb.num, tb.den)
    }

    fn ff_err_to_string(err: i32) -> String {
        let mut buf = [0u8; 256];
        unsafe {
            // av_strerror expects a negative error code.
            ff::av_strerror(err, buf.as_mut_ptr() as *mut i8, buf.len());
        }
        let cstr = match CStr::from_bytes_until_nul(&buf) {
            Ok(s) => s,
            Err(_) => return format!("ffmpeg error {err}"),
        };
        cstr.to_string_lossy().into_owned()
    }

    fn map_ff_err(err: i32) -> CodecError {
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

    fn map_sample_format(sf: SampleFormat) -> CodecResult<ff::AVSampleFormat> {
        use ff::AVSampleFormat::*;
        let av = match sf {
            SampleFormat::U8 { planar: false } => AV_SAMPLE_FMT_U8,
            SampleFormat::U8 { planar: true } => AV_SAMPLE_FMT_U8P,
            SampleFormat::I16 { planar: false } => AV_SAMPLE_FMT_S16,
            SampleFormat::I16 { planar: true } => AV_SAMPLE_FMT_S16P,
            SampleFormat::I32 { planar: false } => AV_SAMPLE_FMT_S32,
            SampleFormat::I32 { planar: true } => AV_SAMPLE_FMT_S32P,
            SampleFormat::I64 { planar: false } => AV_SAMPLE_FMT_S64,
            SampleFormat::I64 { planar: true } => AV_SAMPLE_FMT_S64P,
            SampleFormat::F32 { planar: false } => AV_SAMPLE_FMT_FLT,
            SampleFormat::F32 { planar: true } => AV_SAMPLE_FMT_FLTP,
            SampleFormat::F64 { planar: false } => AV_SAMPLE_FMT_DBL,
            SampleFormat::F64 { planar: true } => AV_SAMPLE_FMT_DBLP,
        };
        Ok(av)
    }

    impl AacEncoder {
        pub fn new(cfg: AacEncoderConfig) -> CodecResult<Self> {
            unsafe {
                let name = CString::new("aac").map_err(|_| CodecError::InvalidData("codec name contains NUL"))?;
                let codec = ff::avcodec_find_encoder_by_name(name.as_ptr());
                if codec.is_null() {
                    return Err(CodecError::Unsupported("FFmpeg AAC encoder not found"));
                }

                let ctx = ff::avcodec_alloc_context3(codec);
                if ctx.is_null() {
                    return Err(CodecError::Other("avcodec_alloc_context3 failed".into()));
                }

                // 基础参数
                (*ctx).sample_rate = cfg.input_format.sample_rate as i32;
                let channels = cfg.input_format.channels() as i32;
                let ch_layout = cfg.input_format.channel_layout;
                // FFmpeg 6+: AVChannelLayout
                if ch_layout.mask != 0 {
                    let ret = ff::av_channel_layout_from_mask(&mut (*ctx).ch_layout, ch_layout.mask);
                    if ret < 0 {
                        ff::avcodec_free_context(&mut (ctx as *mut _));
                        return Err(map_ff_err(ret));
                    }
                } else {
                    ff::av_channel_layout_default(&mut (*ctx).ch_layout, channels);
                }

                (*ctx).sample_fmt = map_sample_format(cfg.input_format.sample_format)?;

                if let Some(br) = cfg.bitrate {
                    (*ctx).bit_rate = br as i64;
                }

                // 对音频来说常用 time_base = 1/sample_rate
                (*ctx).time_base = ff::AVRational {
                    num: 1,
                    den: (*ctx).sample_rate,
                };

                let ret = ff::avcodec_open2(ctx, codec, ptr::null_mut());
                if ret < 0 {
                    ff::avcodec_free_context(&mut (ctx as *mut _));
                    return Err(map_ff_err(ret));
                }

                Ok(Self {
                    cfg,
                    flushed: false,
                    ctx,
                })
            }
        }

        fn ctx_time_base(&self) -> Rational {
            unsafe { tb_from_avr((*self.ctx).time_base) }
        }

        /// 读取 FFmpeg encoder 的 extradata（通常就是 AAC 的 AudioSpecificConfig）。
        pub fn audio_specific_config(&self) -> Option<Vec<u8>> {
            unsafe {
                let size = (*self.ctx).extradata_size as usize;
                if size == 0 || (*self.ctx).extradata.is_null() {
                    return None;
                }
                let slice = core::slice::from_raw_parts((*self.ctx).extradata as *const u8, size);
                Some(slice.to_vec())
            }
        }
    }

    impl Drop for AacEncoder {
        fn drop(&mut self) {
            unsafe {
                if !self.ctx.is_null() {
                    ff::avcodec_free_context(&mut self.ctx);
                }
            }
        }
    }

    impl AudioEncoder for AacEncoder {
        fn name(&self) -> &'static str {
            "aac(ffmpeg)"
        }

        fn input_format(&self) -> Option<AudioFormat> {
            Some(self.cfg.input_format)
        }

        fn preferred_frame_samples(&self) -> Option<usize> {
            // AAC-LC 常见 1024（每声道）
            Some(1024)
        }

        fn lookahead_samples(&self) -> usize {
            0
        }

        fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()> {
            unsafe {
                if self.flushed {
                    return Err(CodecError::InvalidState("already flushed"));
                }

                if frame.is_none() {
                    let ret = ff::avcodec_send_frame(self.ctx, ptr::null());
                    if ret < 0 {
                        return Err(map_ff_err(ret));
                    }
                    self.flushed = true;
                    return Ok(());
                }

                let frame = frame.unwrap();
                let fmt = frame.format();
                if fmt != self.cfg.input_format {
                    return Err(CodecError::InvalidData("input AudioFormat mismatch (no resample/convert layer yet)"));
                }

                // 构造 AVFrame 并复制 PCM
                let avf = ff::av_frame_alloc();
                if avf.is_null() {
                    return Err(CodecError::Other("av_frame_alloc failed".into()));
                }

                (*avf).nb_samples = frame.nb_samples() as i32;
                (*avf).format = (*self.ctx).sample_fmt as i32;
                (*avf).sample_rate = (*self.ctx).sample_rate;
                // FFmpeg 6+: AVChannelLayout
                let ret = ff::av_channel_layout_copy(&mut (*avf).ch_layout, &(*self.ctx).ch_layout);
                if ret < 0 {
                    ff::av_frame_free(&mut (avf as *mut _));
                    return Err(map_ff_err(ret));
                }

                let ret = ff::av_frame_get_buffer(avf, 0);
                if ret < 0 {
                    ff::av_frame_free(&mut (avf as *mut _));
                    return Err(map_ff_err(ret));
                }

                let bps = fmt.sample_format.bytes_per_sample();
                if fmt.is_planar() {
                    let ch = fmt.channels() as usize;
                    for i in 0..ch {
                        let src = frame.plane(i).ok_or(CodecError::InvalidData("missing plane"))?;
                        let expected = frame.nb_samples() * bps;
                        if src.len() != expected {
                            ff::av_frame_free(&mut (avf as *mut _));
                            return Err(CodecError::InvalidData("unexpected plane size"));
                        }
                        let dst = (*avf).data[i] as *mut u8;
                        if dst.is_null() {
                            ff::av_frame_free(&mut (avf as *mut _));
                            return Err(CodecError::InvalidData("ffmpeg frame plane is null"));
                        }
                        ptr::copy_nonoverlapping(src.as_ptr(), dst, expected);
                    }
                } else {
                    let src = frame.plane(0).ok_or(CodecError::InvalidData("missing plane 0"))?;
                    let expected = frame.nb_samples() * (fmt.channels() as usize) * bps;
                    if src.len() != expected {
                        ff::av_frame_free(&mut (avf as *mut _));
                        return Err(CodecError::InvalidData("unexpected interleaved plane size"));
                    }
                    let dst = (*avf).data[0] as *mut u8;
                    if dst.is_null() {
                        ff::av_frame_free(&mut (avf as *mut _));
                        return Err(CodecError::InvalidData("ffmpeg frame data[0] is null"));
                    }
                    ptr::copy_nonoverlapping(src.as_ptr(), dst, expected);
                }

                // 时间戳：直接用上层的 pts（单位由其 time_base 决定）。
                (*avf).pts = frame.pts().unwrap_or(i64::MIN);

                let ret = ff::avcodec_send_frame(self.ctx, avf);
                ff::av_frame_free(&mut (avf as *mut _));
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
                Ok(())
            }
        }

        fn receive_packet(&mut self) -> CodecResult<CodecPacket> {
            unsafe {
                let pkt = ff::av_packet_alloc();
                if pkt.is_null() {
                    return Err(CodecError::Other("av_packet_alloc failed".into()));
                }

                let ret = ff::avcodec_receive_packet(self.ctx, pkt);
                if ret < 0 {
                    ff::av_packet_free(&mut (pkt as *mut _));
                    return Err(map_ff_err(ret));
                }

                let size = (*pkt).size as usize;
                let data = if size == 0 || (*pkt).data.is_null() {
                    Vec::new()
                } else {
                    let slice = core::slice::from_raw_parts((*pkt).data as *const u8, size);
                    slice.to_vec()
                };

                let out = CodecPacket {
                    data,
                    time_base: self.ctx_time_base(),
                    pts: if (*pkt).pts == i64::MIN { None } else { Some((*pkt).pts) },
                    dts: if (*pkt).dts == i64::MIN { None } else { Some((*pkt).dts) },
                    duration: if (*pkt).duration <= 0 { None } else { Some((*pkt).duration) },
                    flags: crate::codec::packet::PacketFlags::empty(),
                };

                ff::av_packet_free(&mut (pkt as *mut _));
                Ok(out)
            }
        }

        fn reset(&mut self) -> CodecResult<()> {
            unsafe {
                ff::avcodec_flush_buffers(self.ctx);
            }
            self.flushed = false;
            Ok(())
        }
    }
}

#[cfg(feature = "ffmpeg")]
pub use ffmpeg_backend::*;


