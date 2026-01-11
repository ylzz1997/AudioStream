use crate::codec::error::CodecError;
use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView, Rational};
use crate::common::audio::fifo::AudioFifo;
use crate::common::io::io::{AudioReader, AudioWriter};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use crate::codec::encoder::opus_encoder::OpusEncoderConfig;

use crate::common::io::io::{AudioIOResult, AudioIOError};

/// 一个最小、可流式的 Opus “自定义封装”：
const MAGIC: &[u8; 8] = b"ASTOPUS\0";
const VERSION: u8 = 1;

fn write_u16_le(w: &mut dyn Write, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u32_le(w: &mut dyn Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_i32_le(w: &mut dyn Write, v: i32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// 构造一个最小 OpusHead（Ogg Opus 的 stream header，常作为 FFmpeg Opus extradata）。
fn build_opus_head(channels: u8, input_sample_rate: u32) -> Vec<u8> {
    // https://wiki.xiph.org/OggOpus#ID_Header
    // 固定 19 字节（channel_mapping_family=0 无后续映射表）
    let mut v = Vec::with_capacity(19);
    v.extend_from_slice(b"OpusHead"); // 8
    v.push(1); // version
    v.push(channels); // channel count
    v.extend_from_slice(&0u16.to_le_bytes()); // pre_skip
    v.extend_from_slice(&input_sample_rate.to_le_bytes()); // input sample rate
    v.extend_from_slice(&0i16.to_le_bytes()); // output gain
    v.push(0); // channel mapping family
    v
}

/// 标准 Ogg Opus（.opus）写入配置：编码参数（encoder）+（预留）容器侧参数。
#[derive(Clone, Debug)]
pub struct OpusOggWriterConfig {
    pub encoder: OpusEncoderConfig,
}

pub struct OpusPacketWriter {
    file: File,
    encoder: crate::codec::encoder::opus_encoder::OpusEncoder,
    fifo: AudioFifo,
    frame_samples: usize,
    header_written: bool,
}

impl OpusPacketWriter {
    pub fn create<P: AsRef<Path>>(
        path: P,
        cfg: crate::codec::encoder::opus_encoder::OpusEncoderConfig,
    ) -> AudioIOResult<Self> {
        let mut file = File::create(path)?;
        let encoder = crate::codec::encoder::opus_encoder::OpusEncoder::new(cfg)?;

        let input_format = encoder
            .input_format()
            .ok_or(AudioIOError::Format("Opus encoder missing input_format"))?;
        if input_format.sample_rate != 48_000 {
            return Err(AudioIOError::Codec(CodecError::InvalidData(
                "Opus writer requires 48kHz input (no resample layer yet)",
            )));
        }

        let time_base = Rational::new(1, 48_000);
        let fifo =
            AudioFifo::new(input_format, time_base).map_err(|_| AudioIOError::Format("failed to create AudioFifo"))?;
        let frame_samples = encoder.preferred_frame_samples().unwrap_or(960);

        // extradata：优先用 encoder 提供的，否则用最小 OpusHead。
        let channels = input_format.channels() as u8;
        let extradata = encoder
            .extradata()
            .unwrap_or_else(|| build_opus_head(channels, input_format.sample_rate));

        // 先写 header（避免 finalize 时才写导致流式落盘不完整）
        file.write_all(MAGIC)?;
        file.write_all(&[VERSION])?;
        write_i32_le(&mut file, time_base.num)?;
        write_i32_le(&mut file, time_base.den)?;
        if extradata.len() > u16::MAX as usize {
            return Err(AudioIOError::Format("opus extradata too large"));
        }
        write_u16_le(&mut file, extradata.len() as u16)?;
        file.write_all(&extradata)?;

        Ok(Self {
            file,
            encoder,
            fifo,
            frame_samples,
            header_written: true,
        })
    }

    pub fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;
        self.fifo
            .push_frame(frame)
            .map_err(|_| AudioIOError::Format("AudioFifo push failed (format mismatch?)"))?;

        while let Some(chunk) = self
            .fifo
            .pop_frame(self.frame_samples)
            .map_err(|_| AudioIOError::Format("AudioFifo pop failed"))?
        {
            self.encoder.send_frame(Some(&chunk))?;
            self.drain_packets()?;
        }
        Ok(())
    }

    pub fn finalize(&mut self) -> AudioIOResult<()> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;

        let remain = self.fifo.available_samples();
        if remain > 0 {
            if let Some(last) = self
                .fifo
                .pop_frame(remain)
                .map_err(|_| AudioIOError::Format("AudioFifo pop last failed"))?
            {
                self.encoder.send_frame(Some(&last))?;
                self.drain_packets()?;
            }
        }

        self.encoder.send_frame(None)?;
        loop {
            match self.encoder.receive_packet() {
                Ok(pkt) => self.write_one_packet(&pkt)?,
                Err(CodecError::Again) => continue,
                Err(CodecError::Eof) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    fn drain_packets(&mut self) -> AudioIOResult<()> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;
        loop {
            match self.encoder.receive_packet() {
                Ok(pkt) => self.write_one_packet(&pkt)?,
                Err(CodecError::Again) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    fn write_one_packet(&mut self, pkt: &CodecPacket) -> AudioIOResult<()> {
        if !self.header_written {
            return Err(AudioIOError::Format("opus header not written"));
        }
        if pkt.data.len() > u32::MAX as usize {
            return Err(AudioIOError::Format("opus packet too large"));
        }
        write_u32_le(&mut self.file, pkt.data.len() as u32)?;
        write_i32_le(&mut self.file, pkt.duration.unwrap_or(0) as i32)?;
        self.file.write_all(&pkt.data)?;
        Ok(())
    }
}

impl AudioWriter for OpusPacketWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        self.write_frame(frame)
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        self.finalize()
    }
}

// -----------------------------
// Standard Ogg Opus (.opus) I/O
// -----------------------------
//
// 说明：
// - 这是标准 Ogg Opus 容器（播放器可直接播放的 .opus）
// - 依赖 FFmpeg 的 libavformat + libavcodec（feature="ffmpeg"）
// - 未开启 ffmpeg 时提供占位实现（返回 Unsupported）

#[cfg(not(feature = "ffmpeg"))]
pub struct OpusOggWriter;

#[cfg(not(feature = "ffmpeg"))]
impl OpusOggWriter {
    pub fn create<P: AsRef<Path>>(
        _path: P,
        _cfg: OpusOggWriterConfig,
    ) -> AudioIOResult<Self> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "Ogg Opus writer requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioWriter for OpusOggWriter {
    fn write_frame(&mut self, _frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "Ogg Opus writer requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
    fn finalize(&mut self) -> AudioIOResult<()> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "Ogg Opus writer requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
}

#[cfg(not(feature = "ffmpeg"))]
pub struct OpusOggReader;

#[cfg(not(feature = "ffmpeg"))]
impl OpusOggReader {
    pub fn open<P: AsRef<Path>>(_path: P) -> AudioIOResult<Self> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "Ogg Opus reader requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioReader for OpusOggReader {
    fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "Ogg Opus reader requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
}

#[cfg(feature = "ffmpeg")]
mod ogg_ffmpeg_backend {
    use super::*;
    use crate::codec::decoder::decoder_interface::AudioDecoder;
    use crate::codec::error::CodecError as CErr;
    use crate::common::audio::audio::AudioFrameViewMut;
    use core::ptr;
    use std::ffi::CString;

    extern crate ffmpeg_sys_next as ff;
    use libc;
    use crate::common::ffmpeg_util::ff_err_to_string;

    fn map_ff_err(err: i32) -> AudioIOError {
        AudioIOError::Codec(CodecError::Other(ff_err_to_string(err)))
    }

    fn tb_from_avr(tb: ff::AVRational) -> Rational {
        Rational::new(tb.num, tb.den)
    }

    fn map_sample_format(sf: crate::common::audio::audio::SampleFormat) -> AudioIOResult<ff::AVSampleFormat> {
        use ff::AVSampleFormat::*;
        let av = match sf {
            crate::common::audio::audio::SampleFormat::I16 { planar: false } => AV_SAMPLE_FMT_S16,
            crate::common::audio::audio::SampleFormat::F32 { planar: false } => AV_SAMPLE_FMT_FLT,
            crate::common::audio::audio::SampleFormat::U8 { planar: false } => AV_SAMPLE_FMT_U8,
            crate::common::audio::audio::SampleFormat::I32 { planar: false } => AV_SAMPLE_FMT_S32,
            _ => {
                return Err(AudioIOError::Format(
                    "Opus Ogg writer supports only packed (interleaved) U8/I16/I32/F32",
                ))
            }
        };
        Ok(av)
    }

    pub struct OpusOggWriter {
        cfg: OpusOggWriterConfig,
        out_c: CString,
        oc: *mut ff::AVFormatContext,
        st: *mut ff::AVStream,
        enc_ctx: *mut ff::AVCodecContext,
        fifo: Option<AudioFifo>,
        frame_samples: usize,
        next_pts: i64,
        finished: bool,
        initialized: bool,
    }

    unsafe impl Send for OpusOggWriter {}

    impl OpusOggWriter {
        pub fn create<P: AsRef<Path>>(
            path: P,
            cfg: OpusOggWriterConfig,
        ) -> AudioIOResult<Self> {
            let out_path = path.as_ref().to_string_lossy();
            let out_c = CString::new(out_path.as_bytes()).map_err(|_| AudioIOError::Format("path contains NUL"))?;
            Ok(Self {
                cfg,
                out_c,
                oc: ptr::null_mut(),
                st: ptr::null_mut(),
                enc_ctx: ptr::null_mut(),
                fifo: None,
                frame_samples: 0,
                next_pts: 0,
                finished: false,
                initialized: false,
            })
        }

        fn encode_and_mux(&mut self, frame: &AudioFrame) -> AudioIOResult<()> {
            unsafe {
                let fmt = frame.format();
                if fmt.sample_rate != 48_000 || fmt.sample_format.is_planar() {
                    return Err(AudioIOError::Format("Ogg Opus writer expects 48kHz packed samples"));
                }
                let nb = frame.nb_samples();
                let plane = frame.plane(0).ok_or(AudioIOError::Format("missing plane 0"))?;
                let bps = fmt.sample_format.bytes_per_sample();
                let expected = nb * (fmt.channels() as usize) * bps;
                if plane.len() != expected {
                    return Err(AudioIOError::Format("unexpected interleaved plane size"));
                }

                let avf = ff::av_frame_alloc();
                if avf.is_null() {
                    return Err(AudioIOError::Codec(CodecError::Other("av_frame_alloc failed".into())));
                }
                (*avf).nb_samples = nb as i32;
                (*avf).format = (*self.enc_ctx).sample_fmt as i32;
                (*avf).sample_rate = (*self.enc_ctx).sample_rate;
                let ret = ff::av_channel_layout_copy(&mut (*avf).ch_layout, &(*self.enc_ctx).ch_layout);
                if ret < 0 {
                    ff::av_frame_free(&mut (avf as *mut _));
                    return Err(map_ff_err(ret));
                }
                let ret = ff::av_frame_get_buffer(avf, 0);
                if ret < 0 {
                    ff::av_frame_free(&mut (avf as *mut _));
                    return Err(map_ff_err(ret));
                }
                let dst = (*avf).data[0] as *mut u8;
                if dst.is_null() {
                    ff::av_frame_free(&mut (avf as *mut _));
                    return Err(AudioIOError::Format("ffmpeg frame data[0] is null"));
                }
                ptr::copy_nonoverlapping(plane.as_ptr(), dst, expected);
                (*avf).pts = self.next_pts;
                self.next_pts += nb as i64;

                let ret = ff::avcodec_send_frame(self.enc_ctx, avf);
                ff::av_frame_free(&mut (avf as *mut _));
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }

                // drain packets
                loop {
                    let pkt = ff::av_packet_alloc();
                    if pkt.is_null() {
                        return Err(AudioIOError::Codec(CodecError::Other("av_packet_alloc failed".into())));
                    }
                    let ret = ff::avcodec_receive_packet(self.enc_ctx, pkt);
                    if ret < 0 {
                        ff::av_packet_free(&mut (pkt as *mut _));
                        #[cfg(unix)]
                        let is_again = ret == ff::AVERROR(libc::EAGAIN) || ret == ff::AVERROR(libc::EWOULDBLOCK);
                        #[cfg(windows)]
                        let is_again = ret == ff::AVERROR(libc::WSAEWOULDBLOCK);
                        if is_again {
                            break;
                        }
                        if ret == ff::AVERROR_EOF {
                            break;
                        }
                        return Err(map_ff_err(ret));
                    }
                    ff::av_packet_rescale_ts(pkt, (*self.enc_ctx).time_base, (*self.st).time_base);
                    (*pkt).stream_index = (*self.st).index;
                    let wret = ff::av_interleaved_write_frame(self.oc, pkt);
                    ff::av_packet_free(&mut (pkt as *mut _));
                    if wret < 0 {
                        return Err(map_ff_err(wret));
                    }
                }
                Ok(())
            }
        }

        fn flush_encoder(&mut self) -> AudioIOResult<()> {
            unsafe {
                let ret = ff::avcodec_send_frame(self.enc_ctx, ptr::null());
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
                loop {
                    let pkt = ff::av_packet_alloc();
                    if pkt.is_null() {
                        return Err(AudioIOError::Codec(CodecError::Other("av_packet_alloc failed".into())));
                    }
                    let ret = ff::avcodec_receive_packet(self.enc_ctx, pkt);
                    if ret < 0 {
                        ff::av_packet_free(&mut (pkt as *mut _));
                        #[cfg(unix)]
                        let is_again = ret == ff::AVERROR(libc::EAGAIN) || ret == ff::AVERROR(libc::EWOULDBLOCK);
                        #[cfg(windows)]
                        let is_again = ret == ff::AVERROR(libc::WSAEWOULDBLOCK);
                        if is_again {
                            continue;
                        }
                        if ret == ff::AVERROR_EOF {
                            break;
                        }
                        return Err(map_ff_err(ret));
                    }
                    ff::av_packet_rescale_ts(pkt, (*self.enc_ctx).time_base, (*self.st).time_base);
                    (*pkt).stream_index = (*self.st).index;
                    let wret = ff::av_interleaved_write_frame(self.oc, pkt);
                    ff::av_packet_free(&mut (pkt as *mut _));
                    if wret < 0 {
                        return Err(map_ff_err(wret));
                    }
                }
                Ok(())
            }
        }
    }

    impl Drop for OpusOggWriter {
        fn drop(&mut self) {
            unsafe {
                if self.finished {
                    return;
                }
                // best-effort cleanup
                if !self.oc.is_null() {
                    if !(*self.oc).pb.is_null() {
                        ff::avio_closep(&mut (*self.oc).pb);
                    }
                    ff::avformat_free_context(self.oc);
                    self.oc = ptr::null_mut();
                }
                if !self.enc_ctx.is_null() {
                    ff::avcodec_free_context(&mut self.enc_ctx);
                }
            }
        }
    }

    impl AudioWriter for OpusOggWriter {
        fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
            if self.finished {
                return Err(AudioIOError::Codec(CodecError::InvalidState("already finalized")));
            }
            if frame.nb_samples() == 0 {
                // 空帧不锁定格式
                return Ok(());
            }
            self.ensure_initialized(frame.format())?;
            {
                let fifo = self.fifo.as_mut().ok_or(AudioIOError::Format("opus writer not initialized"))?;
                fifo.push_frame(frame)
                    .map_err(|_| AudioIOError::Format("AudioFifo push failed (format mismatch?)"))?;
            }
            loop {
                let next = {
                    let fifo = self.fifo.as_mut().ok_or(AudioIOError::Format("opus writer not initialized"))?;
                    fifo.pop_frame(self.frame_samples)
                        .map_err(|_| AudioIOError::Format("AudioFifo pop failed"))?
                };
                let Some(chunk) = next else { break };
                self.encode_and_mux(&chunk)?;
            }
            Ok(())
        }

        fn finalize(&mut self) -> AudioIOResult<()> {
            if self.finished {
                return Ok(());
            }
            if !self.initialized {
                // 从未写入过有效帧：不写 header/trailer，保持空文件
                self.finished = true;
                return Ok(());
            }
            // pad tail to frame_samples
            let remain = self
                .fifo
                .as_ref()
                .ok_or(AudioIOError::Format("opus writer not initialized"))?
                .available_samples();
            if remain > 0 {
                let partial = {
                    let fifo = self.fifo.as_mut().ok_or(AudioIOError::Format("opus writer not initialized"))?;
                    fifo.pop_frame(remain)
                        .map_err(|_| AudioIOError::Format("AudioFifo pop last failed"))?
                };
                if let Some(partial) = partial {
                    // pad zeros
                    let fmt = self.fifo.as_ref().ok_or(AudioIOError::Format("opus writer not initialized"))?.format();
                    let mut padded = AudioFrame::new_alloc(fmt, self.frame_samples)
                        .map_err(|_| AudioIOError::Format("failed to alloc padded frame"))?;
                    padded.set_time_base(partial.time_base()).map_err(|_| AudioIOError::Format("invalid time_base"))?;
                    padded.set_pts(partial.pts());

                    let fmt = partial.format();
                    let bps = fmt.sample_format.bytes_per_sample();
                    let ch = fmt.channels() as usize;
                    let bytes = remain * ch * bps;
                    let src = partial.plane(0).ok_or(AudioIOError::Format("missing plane 0"))?;
                    let dst = padded.plane_mut(0).ok_or(AudioIOError::Format("missing padded plane 0"))?;
                    dst[..bytes].copy_from_slice(&src[..bytes]);
                    self.encode_and_mux(&padded)?;
                }
            }

            self.flush_encoder()?;

            unsafe {
                let ret = ff::av_write_trailer(self.oc);
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
                if !(*self.oc).pb.is_null() {
                    ff::avio_closep(&mut (*self.oc).pb);
                }
                ff::avcodec_free_context(&mut self.enc_ctx);
                ff::avformat_free_context(self.oc);
                self.oc = ptr::null_mut();
            }
            self.finished = true;
            Ok(())
        }
    }

    impl OpusOggWriter {
        fn ensure_initialized(&mut self, in_fmt: AudioFormat) -> AudioIOResult<()> {
            if self.initialized {
                return Ok(());
            }
            if let Some(expected) = self.cfg.encoder.input_format {
                if in_fmt != expected {
                    return Err(AudioIOError::Format("Ogg Opus writer input AudioFormat mismatch"));
                }
            }
            if in_fmt.sample_rate != 48_000 {
                return Err(AudioIOError::Format("Ogg Opus writer expects 48kHz input (resample before writing)"));
            }
            if in_fmt.sample_format.is_planar() {
                return Err(AudioIOError::Format("Ogg Opus writer expects packed/interleaved samples"));
            }

            unsafe {
                // allocate output context (deduce container by extension: .opus => ogg)
                let mut oc: *mut ff::AVFormatContext = ptr::null_mut();
                let ret = ff::avformat_alloc_output_context2(
                    &mut oc as *mut *mut ff::AVFormatContext,
                    ptr::null_mut(),
                    ptr::null(),
                    self.out_c.as_ptr(),
                );
                if ret < 0 || oc.is_null() {
                    return Err(map_ff_err(if ret < 0 { ret } else { -1 }));
                }

                // encoder (prefer libopus)
                let name = CString::new("libopus").map_err(|_| AudioIOError::Format("codec name contains NUL"))?;
                let codec = ff::avcodec_find_encoder_by_name(name.as_ptr());
                if codec.is_null() {
                    ff::avformat_free_context(oc);
                    return Err(AudioIOError::Codec(CodecError::Unsupported("FFmpeg libopus encoder not found")));
                }
                let enc_ctx = ff::avcodec_alloc_context3(codec);
                if enc_ctx.is_null() {
                    ff::avformat_free_context(oc);
                    return Err(AudioIOError::Codec(CodecError::Other("avcodec_alloc_context3 failed".into())));
                }

                (*enc_ctx).sample_rate = 48_000;
                (*enc_ctx).sample_fmt = map_sample_format(in_fmt.sample_format)?;
                (*enc_ctx).time_base = ff::AVRational { num: 1, den: 48_000 };
                (*enc_ctx).bit_rate = self.cfg.encoder.bitrate.unwrap_or(96_000) as i64;

                // channel layout
                let channels = in_fmt.channels() as i32;
                if in_fmt.channel_layout.mask != 0 {
                    let ret = ff::av_channel_layout_from_mask(&mut (*enc_ctx).ch_layout, in_fmt.channel_layout.mask);
                    if ret < 0 {
                        ff::avcodec_free_context(&mut (enc_ctx as *mut _));
                        ff::avformat_free_context(oc);
                        return Err(map_ff_err(ret));
                    }
                } else {
                    ff::av_channel_layout_default(&mut (*enc_ctx).ch_layout, channels);
                }

                // global header if needed by container
                if ((*(*oc).oformat).flags & (ff::AVFMT_GLOBALHEADER as i32)) != 0 {
                    (*enc_ctx).flags |= ff::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
                }

                let ret = ff::avcodec_open2(enc_ctx, codec, ptr::null_mut());
                if ret < 0 {
                    ff::avcodec_free_context(&mut (enc_ctx as *mut _));
                    ff::avformat_free_context(oc);
                    return Err(map_ff_err(ret));
                }

                // stream
                let st = ff::avformat_new_stream(oc, ptr::null());
                if st.is_null() {
                    ff::avcodec_free_context(&mut (enc_ctx as *mut _));
                    ff::avformat_free_context(oc);
                    return Err(AudioIOError::Codec(CodecError::Other("avformat_new_stream failed".into())));
                }
                (*st).time_base = (*enc_ctx).time_base;
                let ret = ff::avcodec_parameters_from_context((*st).codecpar, enc_ctx);
                if ret < 0 {
                    ff::avcodec_free_context(&mut (enc_ctx as *mut _));
                    ff::avformat_free_context(oc);
                    return Err(map_ff_err(ret));
                }

                // open IO
                if ((*(*oc).oformat).flags & (ff::AVFMT_NOFILE as i32)) == 0 {
                    let ret = ff::avio_open(&mut (*oc).pb, self.out_c.as_ptr(), ff::AVIO_FLAG_WRITE);
                    if ret < 0 {
                        ff::avcodec_free_context(&mut (enc_ctx as *mut _));
                        ff::avformat_free_context(oc);
                        return Err(map_ff_err(ret));
                    }
                }

                // write header
                let ret = ff::avformat_write_header(oc, ptr::null_mut());
                if ret < 0 {
                    if !(*oc).pb.is_null() {
                        ff::avio_closep(&mut (*oc).pb);
                    }
                    ff::avcodec_free_context(&mut (enc_ctx as *mut _));
                    ff::avformat_free_context(oc);
                    return Err(map_ff_err(ret));
                }

                self.frame_samples = if (*enc_ctx).frame_size > 0 {
                    (*enc_ctx).frame_size as usize
                } else {
                    960
                };
                let fifo = AudioFifo::new(in_fmt, Rational::new(1, 48_000))
                    .map_err(|_| AudioIOError::Format("failed to create AudioFifo"))?;

                self.oc = oc;
                self.st = st;
                self.enc_ctx = enc_ctx;
                self.fifo = Some(fifo);
                self.next_pts = 0;
                self.initialized = true;
                Ok(())
            }
        }
    }

    pub struct OpusOggReader {
        fmt_ctx: *mut ff::AVFormatContext,
        stream_index: i32,
        pkt_tb: Rational,
        dec: crate::codec::decoder::opus_decoder::OpusDecoder,
        flushed: bool,
    }

    unsafe impl Send for OpusOggReader {}

    impl OpusOggReader {
        pub fn open<P: AsRef<Path>>(path: P) -> AudioIOResult<Self> {
            let in_path = path.as_ref().to_string_lossy();
            let in_c = CString::new(in_path.as_bytes()).map_err(|_| AudioIOError::Format("path contains NUL"))?;

            unsafe {
                let mut ic: *mut ff::AVFormatContext = ptr::null_mut();
                let ret = ff::avformat_open_input(&mut ic as *mut *mut ff::AVFormatContext, in_c.as_ptr(), ptr::null_mut(), ptr::null_mut());
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
                let ret = ff::avformat_find_stream_info(ic, ptr::null_mut());
                if ret < 0 {
                    ff::avformat_close_input(&mut ic);
                    return Err(map_ff_err(ret));
                }

                // find best audio stream
                let mut best = -1i32;
                for i in 0..((*ic).nb_streams as i32) {
                    let st = *(*ic).streams.offset(i as isize);
                    if (*(*st).codecpar).codec_type == ff::AVMediaType::AVMEDIA_TYPE_AUDIO {
                        best = i;
                        break;
                    }
                }
                if best < 0 {
                    ff::avformat_close_input(&mut ic);
                    return Err(AudioIOError::Format("no audio stream found"));
                }
                let st = *(*ic).streams.offset(best as isize);
                let tb = tb_from_avr((*st).time_base);

                // extradata -> OpusDecoder
                let cp = (*st).codecpar;
                let mut extradata: Vec<u8> = Vec::new();
                if !(*cp).extradata.is_null() && (*cp).extradata_size > 0 {
                    let sz = (*cp).extradata_size as usize;
                    let slice = core::slice::from_raw_parts((*cp).extradata as *const u8, sz);
                    extradata = slice.to_vec();
                }
                let dec = if extradata.is_empty() {
                    crate::codec::decoder::opus_decoder::OpusDecoder::new()?
                } else {
                    crate::codec::decoder::opus_decoder::OpusDecoder::new_with_extradata(&extradata)?
                };

                Ok(Self {
                    fmt_ctx: ic,
                    stream_index: best,
                    pkt_tb: tb,
                    dec,
                    flushed: false,
                })
            }
        }

        fn read_next_packet(&mut self) -> AudioIOResult<Option<CodecPacket>> {
            unsafe {
                let pkt = ff::av_packet_alloc();
                if pkt.is_null() {
                    return Err(AudioIOError::Codec(CodecError::Other("av_packet_alloc failed".into())));
                }
                loop {
                    let ret = ff::av_read_frame(self.fmt_ctx, pkt);
                    if ret < 0 {
                        ff::av_packet_free(&mut (pkt as *mut _));
                        return Ok(None);
                    }
                    if (*pkt).stream_index == self.stream_index {
                        break;
                    }
                    ff::av_packet_unref(pkt);
                }

                let size = (*pkt).size as usize;
                let data = if size == 0 || (*pkt).data.is_null() {
                    Vec::new()
                } else {
                    let slice = core::slice::from_raw_parts((*pkt).data as *const u8, size);
                    slice.to_vec()
                };
                let pts = if (*pkt).pts == i64::MIN { None } else { Some((*pkt).pts) };
                let dur = if (*pkt).duration <= 0 { None } else { Some((*pkt).duration) };
                ff::av_packet_free(&mut (pkt as *mut _));

                Ok(Some(CodecPacket {
                    data,
                    time_base: self.pkt_tb,
                    pts,
                    dts: None,
                    duration: dur,
                    flags: crate::codec::packet::PacketFlags::empty(),
                }))
            }
        }
    }

    impl Drop for OpusOggReader {
        fn drop(&mut self) {
            unsafe {
                if !self.fmt_ctx.is_null() {
                    ff::avformat_close_input(&mut self.fmt_ctx);
                }
            }
        }
    }

    impl AudioReader for OpusOggReader {
        fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
            loop {
                match self.dec.receive_frame() {
                    Ok(f) => return Ok(Some(f)),
                    Err(CErr::Again) => {}
                    Err(CErr::Eof) => return Ok(None),
                    Err(e) => return Err(e.into()),
                }

                if let Some(pkt) = self.read_next_packet()? {
                    self.dec.send_packet(Some(pkt))?;
                } else if !self.flushed {
                    self.dec.send_packet(None)?;
                    self.flushed = true;
                } else {
                    // 已 flush：继续 receive_frame() 会转为 Eof
                }
            }
        }
    }
}

#[cfg(feature = "ffmpeg")]
pub use ogg_ffmpeg_backend::{OpusOggReader, OpusOggWriter};
