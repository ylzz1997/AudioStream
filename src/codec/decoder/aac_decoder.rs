//! AAC 解码器实现（流式 send_packet/receive_frame）。

use crate::codec::decoder::decoder_interface::AudioDecoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrame};

#[cfg(not(feature = "ffmpeg"))]
pub struct AacDecoder {
    output_format: Option<AudioFormat>,
    flushed: bool,
}

#[cfg(not(feature = "ffmpeg"))]
impl AacDecoder {
    pub fn new() -> CodecResult<Self> {
        Ok(Self {
            output_format: None,
            flushed: false,
        })
    }

    /// 使用 AAC AudioSpecificConfig(ASC) 初始化（占位实现会忽略）。
    pub fn new_with_asc(_asc: &[u8]) -> CodecResult<Self> {
        Self::new()
    }

    fn unsupported() -> CodecError {
        CodecError::Unsupported("AAC decoder backend not linked (enable FFmpeg backend in your environment)")
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl Default for AacDecoder {
    fn default() -> Self {
        Self::new().expect("failed to create placeholder AAC decoder")
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioDecoder for AacDecoder {
    fn name(&self) -> &'static str {
        "aac(placeholder)"
    }

    fn output_format(&self) -> Option<AudioFormat> {
        self.output_format
    }

    fn delay_samples(&self) -> usize {
        0
    }

    fn send_packet(&mut self, packet: Option<CodecPacket>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        if packet.is_none() {
            self.flushed = true;
            return Ok(());
        }
        Err(Self::unsupported())
    }

    fn receive_frame(&mut self) -> CodecResult<AudioFrame> {
        if self.flushed {
            return Err(CodecError::Eof);
        }

        // PLACEHOLDER
        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        self.flushed = false;
        self.output_format = None;
        Ok(())
    }
}

#[cfg(feature = "ffmpeg")]
mod ffmpeg_backend {
    use super::*;
    use crate::common::ffmpeg_util::channel_layout_from_av;
    use core::ptr;
    use std::ffi::CString;

    extern crate ffmpeg_sys_next as ff;
    use crate::common::audio::audio::{Rational, SampleFormat};
    use crate::common::ffmpeg_util::map_ff_err;

    pub struct AacDecoder {
        output_format: Option<AudioFormat>,
        flushed: bool,
        ctx: *mut ff::AVCodecContext,
        last_time_base: Rational,
    }

    // 同 encoder：实例不共享，语义上可 Send。
    unsafe impl Send for AacDecoder {}

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

    impl AacDecoder {
        pub fn new() -> CodecResult<Self> {
            Self::new_with_asc_internal(None)
        }

        /// 使用 AAC AudioSpecificConfig(ASC) 初始化解码器（推荐用于 raw AAC/ADTS 等场景）。
        pub fn new_with_asc(asc: &[u8]) -> CodecResult<Self> {
            Self::new_with_asc_internal(Some(asc))
        }

        fn new_with_asc_internal(asc: Option<&[u8]>) -> CodecResult<Self> {
            unsafe {
                let name = CString::new("aac").map_err(|_| CodecError::InvalidData("codec name contains NUL"))?;
                let codec = ff::avcodec_find_decoder_by_name(name.as_ptr());
                if codec.is_null() {
                    return Err(CodecError::Unsupported("FFmpeg AAC decoder not found"));
                }
                let ctx = ff::avcodec_alloc_context3(codec);
                if ctx.is_null() {
                    return Err(CodecError::Other("avcodec_alloc_context3 failed".into()));
                }

                // 关键：AAC 解码经常需要 ASC/extradata（尤其是 raw AAC frame 输入）。
                if let Some(asc) = asc {
                    if !asc.is_empty() {
                        let sz = asc.len();
                        // FFmpeg 要求 extradata 末尾有 AV_INPUT_BUFFER_PADDING_SIZE 个 0 字节。
                        let alloc_sz = sz + (ff::AV_INPUT_BUFFER_PADDING_SIZE as usize);
                        let buf = ff::av_mallocz(alloc_sz) as *mut u8;
                        if buf.is_null() {
                            ff::avcodec_free_context(&mut (ctx as *mut _));
                            return Err(CodecError::Other("av_malloc for extradata failed".into()));
                        }
                        ptr::copy_nonoverlapping(asc.as_ptr(), buf as *mut u8, sz);
                        (*ctx).extradata = buf;
                        (*ctx).extradata_size = sz as i32;
                    }
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

    impl Default for AacDecoder {
        fn default() -> Self {
            Self::new().expect("failed to create FFmpeg AAC decoder")
        }
    }

    impl Drop for AacDecoder {
        fn drop(&mut self) {
            unsafe {
                if !self.ctx.is_null() {
                    ff::avcodec_free_context(&mut self.ctx);
                }
            }
        }
    }

    impl AudioDecoder for AacDecoder {
        fn name(&self) -> &'static str {
            "aac(ffmpeg)"
        }

        fn output_format(&self) -> Option<AudioFormat> {
            self.output_format
        }

        fn delay_samples(&self) -> usize {
            // 真实情况下可从 ctx->delay / codec properties 拿到；这里保守为 0
            0
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

                // 尝试告诉解码器输入包时间基（如果字段存在则生效；否则不影响）
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
                // FFmpeg 6+: channels 从 AVChannelLayout 读取
                let channels = (*avf).ch_layout.nb_channels as u16;
                let sample_rate = (*avf).sample_rate as u32;

                let av_sf = core::mem::transmute::<i32, ff::AVSampleFormat>((*avf).format);
                let sf = map_av_sample_format(av_sf)?;

                let ch_layout = channel_layout_from_av(&(*avf).ch_layout);

                let format = AudioFormat {
                    sample_rate,
                    sample_format: sf,
                    channel_layout: ch_layout,
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
                        // audio linesize 一般就是 plane bytes（可能有对齐 padding）
                        let src_slice = core::slice::from_raw_parts(src_ptr, expected);
                        planes.push(src_slice.to_vec());
                    }
                } else {
                    let expected = nb_samples * (channels as usize) * bps;
                    let src_ptr = (*avf).data[0] as *const u8;
                    if src_ptr.is_null() {
                        ff::av_frame_free(&mut (avf as *mut _));
                        return Err(CodecError::InvalidData("ffmpeg frame data[0] is null"));
                    }
                    let src_slice = core::slice::from_raw_parts(src_ptr, expected);
                    planes.push(src_slice.to_vec());
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
pub use ffmpeg_backend::*;

