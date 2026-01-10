//! Opus 编码器实现（流式 send_frame/receive_packet）。
//!
//! 说明：
//! - Opus 生态里通常以 **48kHz** 作为内部时钟（packet duration/时间戳也常按 48kHz samples 计）。
//! - 因此本 `OpusEncoder` **仍然要求输入为 48kHz**（库内部目前不在 encoder 里做隐式重采样）。

use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrameView, Rational};

/// Opus 编码配置（最小集合）。
#[derive(Clone, Debug)]
pub struct OpusEncoderConfig {
    pub input_format: AudioFormat,
    pub bitrate: Option<u32>,
}

impl OpusEncoderConfig {
    pub fn new(input_format: AudioFormat) -> Self {
        Self {
            input_format,
            bitrate: None,
        }
    }
}

/// Opus 编码器。
///
/// - `not(feature="ffmpeg")`：占位实现（无外部依赖）
/// - `feature="ffmpeg"`：FFmpeg backend（libavcodec，优先 libopus）
#[cfg(not(feature = "ffmpeg"))]
pub struct OpusEncoder {
    cfg: OpusEncoderConfig,
    flushed: bool,
}

#[cfg(not(feature = "ffmpeg"))]
impl OpusEncoder {
    pub fn new(cfg: OpusEncoderConfig) -> CodecResult<Self> {
        if cfg.input_format.sample_rate != 48_000 {
            return Err(CodecError::InvalidData(
                "Opus encoder requires 48kHz input (no resample layer yet)",
            ));
        }
        Ok(Self { cfg, flushed: false })
    }

    /// 读取 encoder 的 extradata（FFmpeg backend 下通常是 OpusHead；占位实现为 None）。
    pub fn extradata(&self) -> Option<Vec<u8>> {
        None
    }

    fn unsupported() -> CodecError {
        CodecError::Unsupported("Opus encoder backend not linked (enable FFmpeg backend in your environment)")
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioEncoder for OpusEncoder {
    fn name(&self) -> &'static str {
        "opus(placeholder)"
    }

    fn input_format(&self) -> Option<AudioFormat> {
        Some(self.cfg.input_format)
    }

    fn preferred_frame_samples(&self) -> Option<usize> {
        // Opus 常见 20ms = 960 samples@48k（每声道）
        Some(960)
    }

    fn lookahead_samples(&self) -> usize {
        0
    }

    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        if frame.is_none() {
            self.flushed = true;
            return Ok(());
        }
        Err(Self::unsupported())
    }

    fn receive_packet(&mut self) -> CodecResult<CodecPacket> {
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        self.flushed = false;
        Ok(())
    }
}

pub fn default_opus_packet_time_base() -> Rational {
    // Opus 标准时间基一般按 48kHz 计（samples）。
    Rational::new(1, 48_000)
}

#[cfg(feature = "ffmpeg")]
mod ffmpeg_backend {
    use super::*;
    use core::ptr;
    use std::ffi::CString;

    extern crate ffmpeg_sys_next as ff;
    use crate::common::audio::audio::SampleFormat;
    use crate::common::ffmpeg_util::map_ff_err;

    /// Opus 编码器（FFmpeg backend）。
    pub struct OpusEncoder {
        cfg: OpusEncoderConfig,
        flushed: bool,
        ctx: *mut ff::AVCodecContext,
        frame_samples: Option<usize>,
    }

    unsafe impl Send for OpusEncoder {}

    fn tb_from_avr(tb: ff::AVRational) -> Rational {
        Rational::new(tb.num, tb.den)
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

    fn find_encoder_by_name(names: &[&str]) -> Result<*const ff::AVCodec, CodecError> {
        unsafe {
            for &n in names {
                let name = CString::new(n).map_err(|_| CodecError::InvalidData("codec name contains NUL"))?;
                let codec = ff::avcodec_find_encoder_by_name(name.as_ptr());
                if !codec.is_null() {
                    return Ok(codec);
                }
            }
        }
        Err(CodecError::Unsupported("FFmpeg Opus encoder not found (try libopus)"))
    }

    impl OpusEncoder {
        pub fn new(cfg: OpusEncoderConfig) -> CodecResult<Self> {
            if cfg.input_format.sample_rate != 48_000 {
                return Err(CodecError::InvalidData(
                    "Opus encoder requires 48kHz input (no resample layer yet)",
                ));
            }

            unsafe {
                // 优先 libopus（常见），其次尝试 native opus encoder（如果存在）。
                let codec = find_encoder_by_name(&["libopus", "opus"])?;

                let ctx = ff::avcodec_alloc_context3(codec);
                if ctx.is_null() {
                    return Err(CodecError::Other("avcodec_alloc_context3 failed".into()));
                }

                (*ctx).sample_rate = cfg.input_format.sample_rate as i32;
                let channels = cfg.input_format.channels() as i32;
                let ch_layout = cfg.input_format.channel_layout;
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

                // 对 Opus：常用 time_base = 1/48000（samples）
                (*ctx).time_base = ff::AVRational { num: 1, den: 48_000 };

                let ret = ff::avcodec_open2(ctx, codec, ptr::null_mut());
                if ret < 0 {
                    ff::avcodec_free_context(&mut (ctx as *mut _));
                    return Err(map_ff_err(ret));
                }

                let fs = if (*ctx).frame_size > 0 {
                    Some((*ctx).frame_size as usize)
                } else {
                    None
                };

                Ok(Self {
                    cfg,
                    flushed: false,
                    ctx,
                    frame_samples: fs,
                })
            }
        }

        fn ctx_time_base(&self) -> Rational {
            unsafe { tb_from_avr((*self.ctx).time_base) }
        }

        /// 读取 FFmpeg encoder 的 extradata（通常是 OpusHead）。
        pub fn extradata(&self) -> Option<Vec<u8>> {
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

    impl Drop for OpusEncoder {
        fn drop(&mut self) {
            unsafe {
                if !self.ctx.is_null() {
                    ff::avcodec_free_context(&mut self.ctx);
                }
            }
        }
    }

    impl AudioEncoder for OpusEncoder {
        fn name(&self) -> &'static str {
            "opus(ffmpeg)"
        }

        fn input_format(&self) -> Option<AudioFormat> {
            Some(self.cfg.input_format)
        }

        fn preferred_frame_samples(&self) -> Option<usize> {
            self.frame_samples.or(Some(960))
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
                    eprintln!(
                        "OpusEncoder input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                        crate::common::audio::audio::audio_format_diff(self.cfg.input_format, fmt)
                    );
                    return Err(CodecError::InvalidData(
                        "input AudioFormat mismatch (no resample/convert layer yet)",
                    ));
                }

                let avf = ff::av_frame_alloc();
                if avf.is_null() {
                    return Err(CodecError::Other("av_frame_alloc failed".into()));
                }

                (*avf).nb_samples = frame.nb_samples() as i32;
                (*avf).format = (*self.ctx).sample_fmt as i32;
                (*avf).sample_rate = (*self.ctx).sample_rate;
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


