use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrameView};

/// MP3 编码器（FFmpeg backend 为主）。
///
/// - `not(feature="ffmpeg")`：占位实现
/// - `feature="ffmpeg")`：libavcodec（优先 libmp3lame）
#[derive(Clone, Debug)]
pub struct Mp3EncoderConfig {
    pub input_format: AudioFormat,
    pub bitrate: Option<u32>,
}

impl Mp3EncoderConfig {
    pub fn new(input_format: AudioFormat) -> Self {
        Self {
            input_format,
            bitrate: Some(128_000),
        }
    }
}

#[cfg(not(feature = "ffmpeg"))]
pub struct Mp3Encoder {
    cfg: Mp3EncoderConfig,
    flushed: bool,
}

#[cfg(not(feature = "ffmpeg"))]
impl Mp3Encoder {
    pub fn new(cfg: Mp3EncoderConfig) -> CodecResult<Self> {
        Ok(Self { cfg, flushed: false })
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioEncoder for Mp3Encoder {
    fn name(&self) -> &'static str {
        "mp3(placeholder)"
    }

    fn input_format(&self) -> Option<AudioFormat> {
        Some(self.cfg.input_format)
    }

    fn preferred_frame_samples(&self) -> Option<usize> {
        // MP3 常见 1152 samples/每声道
        Some(1152)
    }

    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        if frame.is_none() {
            self.flushed = true;
            return Ok(());
        }
        Err(CodecError::Unsupported("MP3 encoder requires ffmpeg feature"))
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

#[cfg(feature = "ffmpeg")]
mod ffmpeg_backend {
    use super::*;
    use core::ptr;
    use std::ffi::CString;

    extern crate ffmpeg_sys_next as ff;
    use crate::common::audio::audio::{Rational, SampleFormat};
    use crate::common::ffmpeg_util::map_ff_err;

    pub struct Mp3Encoder {
        cfg: Mp3EncoderConfig,
        flushed: bool,
        ctx: *mut ff::AVCodecContext,
    }

    unsafe impl Send for Mp3Encoder {}

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

    impl Mp3Encoder {
        pub fn new(cfg: Mp3EncoderConfig) -> CodecResult<Self> {
            unsafe {
                // 优先 libmp3lame；如果不可用则退回 mp3
                let codec = {
                    let name = CString::new("libmp3lame").unwrap();
                    let c = ff::avcodec_find_encoder_by_name(name.as_ptr());
                    if !c.is_null() {
                        c
                    } else {
                        let name = CString::new("mp3").unwrap();
                        let c2 = ff::avcodec_find_encoder_by_name(name.as_ptr());
                        if c2.is_null() {
                            return Err(CodecError::Unsupported("FFmpeg MP3 encoder not found"));
                        }
                        c2
                    }
                };

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
    }

    impl Drop for Mp3Encoder {
        fn drop(&mut self) {
            unsafe {
                if !self.ctx.is_null() {
                    ff::avcodec_free_context(&mut self.ctx);
                }
            }
        }
    }

    impl AudioEncoder for Mp3Encoder {
        fn name(&self) -> &'static str {
            "mp3(ffmpeg)"
        }

        fn input_format(&self) -> Option<AudioFormat> {
            Some(self.cfg.input_format)
        }

        fn preferred_frame_samples(&self) -> Option<usize> {
            Some(1152)
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
                let channels = fmt.channels() as usize;
                if fmt.is_planar() {
                    let expected = frame.nb_samples() * bps;
                    for ch in 0..channels {
                        let src = frame.plane(ch).ok_or(CodecError::InvalidData("missing plane"))?;
                        if src.len() != expected {
                            ff::av_frame_free(&mut (avf as *mut _));
                            return Err(CodecError::InvalidData("unexpected plane size"));
                        }
                        let dst = (*avf).data[ch] as *mut u8;
                        if dst.is_null() {
                            ff::av_frame_free(&mut (avf as *mut _));
                            return Err(CodecError::InvalidData("ffmpeg plane is null"));
                        }
                        ptr::copy_nonoverlapping(src.as_ptr(), dst, expected);
                    }
                } else {
                    let expected = frame.nb_samples() * channels * bps;
                    let src = frame.plane(0).ok_or(CodecError::InvalidData("missing plane 0"))?;
                    if src.len() != expected {
                        ff::av_frame_free(&mut (avf as *mut _));
                        return Err(CodecError::InvalidData("unexpected interleaved plane size"));
                    }
                    let dst = (*avf).data[0] as *mut u8;
                    if dst.is_null() {
                        ff::av_frame_free(&mut (avf as *mut _));
                        return Err(CodecError::InvalidData("ffmpeg data[0] is null"));
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
pub use ffmpeg_backend::Mp3Encoder;


