use crate::common::io::io::{AudioIOResult, AudioIOError};
use crate::codec::error::CodecError;
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView, Rational};
use crate::common::audio::fifo::AudioFifo;
use crate::common::io::io::{AudioReader, AudioWriter};
use crate::codec::encoder::flac_encoder::FlacEncoderConfig;
use std::path::Path;

/// FLAC 文件写入配置：编码参数（encoder）+（预留）writer/容器侧参数。
#[derive(Clone, Debug)]
pub struct FlacWriterConfig {
    pub encoder: FlacEncoderConfig,
}

// -----------------------------
// Standard FLAC (.flac) I/O
// -----------------------------
// - 依赖 FFmpeg 的 libavformat + libavcodec（feature="ffmpeg"）
// - 未开启 ffmpeg 时提供占位实现（返回 Unsupported）

#[cfg(not(feature = "ffmpeg"))]
pub struct FlacWriter;

#[cfg(not(feature = "ffmpeg"))]
impl FlacWriter {
    pub fn create<P: AsRef<Path>>(_path: P, _cfg: FlacWriterConfig) -> AudioIOResult<Self> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "FLAC writer requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioWriter for FlacWriter {
    fn write_frame(&mut self, _frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "FLAC writer requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "FLAC writer requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
}

#[cfg(not(feature = "ffmpeg"))]
pub struct FlacReader;

#[cfg(not(feature = "ffmpeg"))]
impl FlacReader {
    pub fn open<P: AsRef<Path>>(_path: P) -> AudioIOResult<Self> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "FLAC reader requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
}

#[cfg(not(feature = "ffmpeg"))]
impl AudioReader for FlacReader {
    fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
        Err(AudioIOError::Codec(CodecError::Unsupported(
            "FLAC reader requires FFmpeg backend (enable feature=ffmpeg)",
        )))
    }
}

#[cfg(feature = "ffmpeg")]
mod ffmpeg_backend {
    use super::*;
    use crate::codec::decoder::decoder_interface::AudioDecoder;
    use core::ptr;
    use std::ffi::CString;

    extern crate ffmpeg_sys_next as ff;
    use crate::common::audio::audio::SampleFormat;
    use libc;
    use crate::common::ffmpeg_util::ff_err_to_string;

    fn map_ff_err(err: i32) -> AudioIOError {
        AudioIOError::Codec(CodecError::Other(ff_err_to_string(err)))
    }

    fn tb_from_avr(tb: ff::AVRational) -> Rational {
        Rational::new(tb.num, tb.den)
    }

    fn map_sample_format(sf: SampleFormat) -> AudioIOResult<ff::AVSampleFormat> {
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

    pub struct FlacWriter {
        cfg: FlacWriterConfig,
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

    unsafe impl Send for FlacWriter {}

    impl FlacWriter {
        pub fn create<P: AsRef<Path>>(path: P, cfg: FlacWriterConfig) -> AudioIOResult<Self> {
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
                if fmt.sample_rate as i32 != (*self.enc_ctx).sample_rate {
                    return Err(AudioIOError::Format("FLAC writer expects fixed sample_rate"));
                }
                let fifo_fmt = self.fifo.as_ref().map(|f| f.format()).ok_or(AudioIOError::Format("FLAC writer not initialized"))?;
                if fmt != fifo_fmt {
                    return Err(AudioIOError::Format("FLAC writer format mismatch (no resample/convert layer yet)"));
                }

                let nb = frame.nb_samples();
                let bps = fmt.sample_format.bytes_per_sample();
                let channels = fmt.channels() as usize;

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

                if fmt.is_planar() {
                    let expected = nb * bps;
                    for ch in 0..channels {
                        let src = frame.plane(ch).ok_or(AudioIOError::Format("missing plane"))?;
                        if src.len() != expected {
                            ff::av_frame_free(&mut (avf as *mut _));
                            return Err(AudioIOError::Format("unexpected plane size"));
                        }
                        let dst = (*avf).data[ch] as *mut u8;
                        if dst.is_null() {
                            ff::av_frame_free(&mut (avf as *mut _));
                            return Err(AudioIOError::Format("ffmpeg frame plane is null"));
                        }
                        ptr::copy_nonoverlapping(src.as_ptr(), dst, expected);
                    }
                } else {
                    let expected = nb * channels * bps;
                    let src = frame.plane(0).ok_or(AudioIOError::Format("missing plane 0"))?;
                    if src.len() != expected {
                        ff::av_frame_free(&mut (avf as *mut _));
                        return Err(AudioIOError::Format("unexpected interleaved plane size"));
                    }
                    let dst = (*avf).data[0] as *mut u8;
                    if dst.is_null() {
                        ff::av_frame_free(&mut (avf as *mut _));
                        return Err(AudioIOError::Format("ffmpeg frame data[0] is null"));
                    }
                    ptr::copy_nonoverlapping(src.as_ptr(), dst, expected);
                }

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

    impl Drop for FlacWriter {
        fn drop(&mut self) {
            unsafe {
                if self.finished {
                    return;
                }
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

    impl AudioWriter for FlacWriter {
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
                let fifo = self
                    .fifo
                    .as_mut()
                    .ok_or(AudioIOError::Format("FLAC writer not initialized"))?;
                fifo
                .push_frame(frame)
                .map_err(|_| AudioIOError::Format("AudioFifo push failed (format mismatch?)"))?;
            }

            loop {
                let next = {
                    let fifo = self
                        .fifo
                        .as_mut()
                        .ok_or(AudioIOError::Format("FLAC writer not initialized"))?;
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

            // flush FIFO tail（允许 < frame_samples，FLAC 可变帧长）
            let remain = self
                .fifo
                .as_ref()
                .ok_or(AudioIOError::Format("FLAC writer not initialized"))?
                .available_samples();
            if remain > 0 {
                let last = {
                    let fifo = self
                        .fifo
                        .as_mut()
                        .ok_or(AudioIOError::Format("FLAC writer not initialized"))?;
                    fifo.pop_frame(remain)
                        .map_err(|_| AudioIOError::Format("AudioFifo pop last failed"))?
                };
                if let Some(last) = last {
                    self.encode_and_mux(&last)?;
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

    impl FlacWriter {
        fn ensure_initialized(&mut self, in_fmt: AudioFormat) -> AudioIOResult<()> {
            if self.initialized {
                return Ok(());
            }
            if let Some(expected) = self.cfg.encoder.input_format {
                if in_fmt != expected {
                    eprintln!(
                        "FlacWriter input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                        crate::common::audio::audio::audio_format_diff(expected, in_fmt)
                    );
                    return Err(AudioIOError::Format("FLAC writer input AudioFormat mismatch"));
                }
            }

            let time_base = Rational::new(1, in_fmt.sample_rate as i32);
            let fifo = AudioFifo::new(in_fmt, time_base).map_err(|_| AudioIOError::Format("failed to create AudioFifo"))?;

            unsafe {
                // allocate output context (deduce container by extension: .flac => flac)
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

                // encoder (flac)
                let name = CString::new("flac").map_err(|_| AudioIOError::Format("codec name contains NUL"))?;
                let codec = ff::avcodec_find_encoder_by_name(name.as_ptr());
                if codec.is_null() {
                    ff::avformat_free_context(oc);
                    return Err(AudioIOError::Codec(CodecError::Unsupported("FFmpeg FLAC encoder not found")));
                }
                let enc_ctx = ff::avcodec_alloc_context3(codec);
                if enc_ctx.is_null() {
                    ff::avformat_free_context(oc);
                    return Err(AudioIOError::Codec(CodecError::Other("avcodec_alloc_context3 failed".into())));
                }

                (*enc_ctx).sample_rate = in_fmt.sample_rate as i32;
                (*enc_ctx).sample_fmt = map_sample_format(in_fmt.sample_format)?;
                (*enc_ctx).time_base = ff::AVRational {
                    num: 1,
                    den: (*enc_ctx).sample_rate,
                };
                if let Some(level) = self.cfg.encoder.compression_level {
                    (*enc_ctx).compression_level = level;
                }

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

                // frame size: flac 通常可变，FFmpeg 可能给 0；这里给个稳妥 chunk
                self.frame_samples = if (*enc_ctx).frame_size > 0 {
                    (*enc_ctx).frame_size as usize
                } else {
                    4096
                };

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

    pub struct FlacReader {
        fmt_ctx: *mut ff::AVFormatContext,
        stream_index: i32,
        pkt_tb: Rational,
        dec: crate::codec::decoder::flac_decoder::FlacDecoder,
        flushed: bool,
    }

    unsafe impl Send for FlacReader {}

    impl FlacReader {
        pub fn open<P: AsRef<Path>>(path: P) -> AudioIOResult<Self> {
            let in_path = path.as_ref().to_string_lossy();
            let in_c = CString::new(in_path.as_bytes()).map_err(|_| AudioIOError::Format("path contains NUL"))?;

            unsafe {
                let mut ic: *mut ff::AVFormatContext = ptr::null_mut();
                let ret = ff::avformat_open_input(
                    &mut ic as *mut *mut ff::AVFormatContext,
                    in_c.as_ptr(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                );
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
                let ret = ff::avformat_find_stream_info(ic, ptr::null_mut());
                if ret < 0 {
                    ff::avformat_close_input(&mut ic);
                    return Err(map_ff_err(ret));
                }

                // find first audio stream
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

                let dec = crate::codec::decoder::flac_decoder::FlacDecoder::new()?;

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

    impl Drop for FlacReader {
        fn drop(&mut self) {
            unsafe {
                if !self.fmt_ctx.is_null() {
                    ff::avformat_close_input(&mut self.fmt_ctx);
                }
            }
        }
    }

    impl AudioReader for FlacReader {
        fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
            loop {
                match self.dec.receive_frame() {
                    Ok(f) => return Ok(Some(f)),
                    Err(crate::codec::error::CodecError::Again) => {}
                    Err(crate::codec::error::CodecError::Eof) => return Ok(None),
                    Err(e) => return Err(e.into()),
                }

                if let Some(pkt) = self.read_next_packet()? {
                    self.dec.send_packet(Some(pkt))?;
                } else if !self.flushed {
                    self.dec.send_packet(None)?;
                    self.flushed = true;
                }
            }
        }
    }
}

#[cfg(feature = "ffmpeg")]
pub use ffmpeg_backend::{FlacReader, FlacWriter};

