use crate::codec::decoder::decoder_interface::AudioDecoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrame};

/// FLAC 解码器（FFmpeg backend 为主）。
///
/// - `not(feature="ffmpeg")`：占位实现
/// - `feature="ffmpeg"`：libavcodec（flac decoder）
#[cfg(not(feature = "ffmpeg"))]
pub struct FlacDecoder {
    flushed: bool,
}

#[cfg(not(feature = "ffmpeg"))]
impl FlacDecoder {
    pub fn new() -> CodecResult<Self> {
        Ok(Self { flushed: false })
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioDecoder for FlacDecoder {
    fn name(&self) -> &'static str {
        "flac(placeholder)"
    }

    fn output_format(&self) -> Option<AudioFormat> {
        None
    }

    fn send_packet(&mut self, packet: Option<CodecPacket>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        if packet.is_none() {
            self.flushed = true;
            return Ok(());
        }
        Err(CodecError::Unsupported("FLAC decoder requires ffmpeg feature"))
    }

    fn receive_frame(&mut self) -> CodecResult<AudioFrame> {
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
    use crate::common::audio::audio::{ChannelLayout, Rational, SampleFormat};
    use crate::common::ffmpeg_util::map_ff_err;

    pub struct FlacDecoder {
        output_format: Option<AudioFormat>,
        flushed: bool,
        ctx: *mut ff::AVCodecContext,
        last_time_base: Rational,
    }

    unsafe impl Send for FlacDecoder {}

    fn tb_from_avr(tb: ff::AVRational) -> Rational {
        Rational::new(tb.num, tb.den)
    }

    fn avr_from_tb(tb: Rational) -> ff::AVRational {
        ff::AVRational { num: tb.num, den: tb.den }
    }

    fn map_av_sample_format(av: ff::AVSampleFormat) -> CodecResult<SampleFormat> {
        use ff::AVSampleFormat::*;
        let sf = match av {
            AV_SAMPLE_FMT_U8 => SampleFormat::U8 { planar: false },
            AV_SAMPLE_FMT_U8P => SampleFormat::U8 { planar: true },
            AV_SAMPLE_FMT_S16 => SampleFormat::I16 { planar: false },
            AV_SAMPLE_FMT_S16P => SampleFormat::I16 { planar: true },
            AV_SAMPLE_FMT_S32 => SampleFormat::I32 { planar: false },
            AV_SAMPLE_FMT_S32P => SampleFormat::I32 { planar: true },
            AV_SAMPLE_FMT_S64 => SampleFormat::I64 { planar: false },
            AV_SAMPLE_FMT_S64P => SampleFormat::I64 { planar: true },
            AV_SAMPLE_FMT_FLT => SampleFormat::F32 { planar: false },
            AV_SAMPLE_FMT_FLTP => SampleFormat::F32 { planar: true },
            AV_SAMPLE_FMT_DBL => SampleFormat::F64 { planar: false },
            AV_SAMPLE_FMT_DBLP => SampleFormat::F64 { planar: true },
            _ => return Err(CodecError::Unsupported("unsupported FFmpeg sample format")),
        };
        Ok(sf)
    }

    impl FlacDecoder {
        pub fn new() -> CodecResult<Self> {
            unsafe {
                let name = CString::new("flac").map_err(|_| CodecError::InvalidData("codec name contains NUL"))?;
                let codec = ff::avcodec_find_decoder_by_name(name.as_ptr());
                if codec.is_null() {
                    return Err(CodecError::Unsupported("FFmpeg FLAC decoder not found"));
                }
                let ctx = ff::avcodec_alloc_context3(codec);
                if ctx.is_null() {
                    return Err(CodecError::Other("avcodec_alloc_context3 failed".into()));
                }
                let ret = ff::avcodec_open2(ctx, codec, ptr::null_mut());
                if ret < 0 {
                    ff::avcodec_free_context(&mut (ctx as *mut _));
                    return Err(map_ff_err(ret));
                }
                Ok(Self {
                    output_format: None,
                    flushed: false,
                    ctx,
                    last_time_base: tb_from_avr((*ctx).time_base),
                })
            }
        }
    }

    impl Drop for FlacDecoder {
        fn drop(&mut self) {
            unsafe {
                if !self.ctx.is_null() {
                    ff::avcodec_free_context(&mut self.ctx);
                }
            }
        }
    }

    impl AudioDecoder for FlacDecoder {
        fn name(&self) -> &'static str {
            "flac(ffmpeg)"
        }

        fn output_format(&self) -> Option<AudioFormat> {
            self.output_format
        }

        fn send_packet(&mut self, packet: Option<CodecPacket>) -> CodecResult<()> {
            unsafe {
                if self.flushed {
                    return Err(CodecError::InvalidState("already flushed"));
                }
                if packet.is_none() {
                    let ret = ff::avcodec_send_packet(self.ctx, ptr::null());
                    if ret < 0 {
                        return Err(map_ff_err(ret));
                    }
                    self.flushed = true;
                    return Ok(());
                }

                let packet = packet.unwrap();
                self.last_time_base = packet.time_base;
                (*self.ctx).pkt_timebase = avr_from_tb(packet.time_base);

                let pkt = ff::av_packet_alloc();
                if pkt.is_null() {
                    return Err(CodecError::Other("av_packet_alloc failed".into()));
                }
                let ret = ff::av_new_packet(pkt, packet.data.len() as i32);
                if ret < 0 {
                    ff::av_packet_free(&mut (pkt as *mut _));
                    return Err(map_ff_err(ret));
                }
                if !(*pkt).data.is_null() && !packet.data.is_empty() {
                    ptr::copy_nonoverlapping(packet.data.as_ptr(), (*pkt).data as *mut u8, packet.data.len());
                }
                (*pkt).pts = packet.pts.unwrap_or(i64::MIN);
                (*pkt).dts = packet.dts.unwrap_or(i64::MIN);
                (*pkt).duration = packet.duration.unwrap_or(0);

                let ret = ff::avcodec_send_packet(self.ctx, pkt);
                ff::av_packet_free(&mut (pkt as *mut _));
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
                Ok(())
            }
        }

        fn receive_frame(&mut self) -> CodecResult<AudioFrame> {
            unsafe {
                let avf = ff::av_frame_alloc();
                if avf.is_null() {
                    return Err(CodecError::Other("av_frame_alloc failed".into()));
                }
                let ret = ff::avcodec_receive_frame(self.ctx, avf);
                if ret < 0 {
                    ff::av_frame_free(&mut (avf as *mut _));
                    return Err(map_ff_err(ret));
                }

                let nb_samples = (*avf).nb_samples as usize;
                let channels = (*avf).ch_layout.nb_channels as u16;
                let sample_rate = (*avf).sample_rate as u32;
                let av_sf = core::mem::transmute::<i32, ff::AVSampleFormat>((*avf).format);
                let sf = map_av_sample_format(av_sf)?;

                let format = AudioFormat {
                    sample_rate,
                    sample_format: sf,
                    channel_layout: ChannelLayout::default_for_channels(channels),
                };

                let bps = format.sample_format.bytes_per_sample();
                let mut planes: Vec<Vec<u8>> = Vec::new();
                if format.is_planar() {
                    planes.reserve(channels as usize);
                    let expected = nb_samples * bps;
                    for ch in 0..(channels as usize) {
                        let src_ptr = (*avf).data[ch] as *const u8;
                        if src_ptr.is_null() {
                            ff::av_frame_free(&mut (avf as *mut _));
                            return Err(CodecError::InvalidData("ffmpeg frame plane is null"));
                        }
                        let src = core::slice::from_raw_parts(src_ptr, expected);
                        planes.push(src.to_vec());
                    }
                } else {
                    let expected = nb_samples * (channels as usize) * bps;
                    let src_ptr = (*avf).data[0] as *const u8;
                    if src_ptr.is_null() {
                        ff::av_frame_free(&mut (avf as *mut _));
                        return Err(CodecError::InvalidData("ffmpeg frame data[0] is null"));
                    }
                    let src = core::slice::from_raw_parts(src_ptr, expected);
                    planes.push(src.to_vec());
                }

                let pts = if (*avf).pts == i64::MIN { None } else { Some((*avf).pts) };
                let time_base = self.last_time_base;
                ff::av_frame_free(&mut (avf as *mut _));

                let out = AudioFrame::from_planes(format, nb_samples, time_base, pts, planes)
                    .map_err(|_| CodecError::InvalidData("failed to build AudioFrame"))?;
                self.output_format = Some(format);
                Ok(out)
            }
        }

        fn reset(&mut self) -> CodecResult<()> {
            unsafe {
                ff::avcodec_flush_buffers(self.ctx);
            }
            self.flushed = false;
            self.output_format = None;
            Ok(())
        }
    }
}

#[cfg(feature = "ffmpeg")]
pub use ffmpeg_backend::FlacDecoder;

