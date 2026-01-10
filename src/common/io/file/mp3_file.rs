use crate::common::io::io::{AudioIOResult, AudioIOError};
use crate::common::io::io::{AudioReader, AudioWriter};
use crate::common::audio::fifo::AudioFifo;
use crate::common::audio::audio::{AudioFrame, AudioFrameView, Rational};
use crate::codec::encoder::mp3_encoder::Mp3EncoderConfig;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
#[cfg(feature = "ffmpeg")]
use core::ptr;

/// MP3 文件写入配置：编码参数（encoder）+（预留）writer/容器侧参数。
#[derive(Clone, Debug)]
pub struct Mp3WriterConfig {
    pub encoder: Mp3EncoderConfig,
}

impl Mp3WriterConfig {
    pub fn new(input_format: crate::common::audio::audio::AudioFormat) -> Self {
        Self {
            encoder: Mp3EncoderConfig::new(input_format),
        }
    }
}

pub struct Mp3Writer {
    w: BufWriter<File>,
    enc: crate::codec::encoder::mp3_encoder::Mp3Encoder,
    fifo: AudioFifo,
    frame_samples: usize,
}

impl Mp3Writer {
    pub fn create<P: AsRef<Path>>(path: P, cfg: Mp3WriterConfig) -> AudioIOResult<Self> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;
        let w = BufWriter::new(File::create(path)?);
        let enc_cfg = cfg.encoder;
        let fmt = enc_cfg.input_format;
        let sample_rate = fmt.sample_rate;
        let enc = crate::codec::encoder::mp3_encoder::Mp3Encoder::new(enc_cfg)?;
        // MP3 常见 1152；若编码器未声明，则退回 1152
        let frame_samples = enc.preferred_frame_samples().unwrap_or(1152);
        let time_base = Rational::new(1, sample_rate as i32);
        let fifo = AudioFifo::new(fmt, time_base)
            .map_err(|_| AudioIOError::Format("failed to create AudioFifo"))?;
        Ok(Self { w, enc, fifo, frame_samples })
    }
}

impl AudioWriter for Mp3Writer {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;
        // 先入 FIFO，再按 mp3 frame_size 重分帧输出
        self.fifo
            .push_frame(frame)
            .map_err(|_| AudioIOError::Format("AudioFifo push failed (format mismatch?)"))?;

        while let Some(chunk) = self
            .fifo
            .pop_frame(self.frame_samples)
            .map_err(|_| AudioIOError::Format("AudioFifo pop failed"))?
        {
            self.enc.send_frame(Some(&chunk))?;
            drain_mp3_packets(&mut self.enc, &mut self.w)?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;
        use crate::codec::error::CodecError;
        // 先把 FIFO 里剩余数据作为“最后一帧”（允许 < frame_samples）送入
        let remain = self.fifo.available_samples();
        if remain > 0 {
            if let Some(last) = self
                .fifo
                .pop_frame(remain)
                .map_err(|_| AudioIOError::Format("AudioFifo pop last failed"))?
            {
                self.enc.send_frame(Some(&last))?;
                drain_mp3_packets(&mut self.enc, &mut self.w)?;
            }
        }

        // flush encoder
        self.enc.send_frame(None)?;
        loop {
            match self.enc.receive_packet() {
                Ok(pkt) => self.w.write_all(&pkt.data)?,
                Err(CodecError::Again) => continue,
                Err(CodecError::Eof) => break,
                Err(e) => return Err(e.into()),
            }
        }
        self.w.flush()?;
        Ok(())
    }
}

fn drain_mp3_packets(
    enc: &mut crate::codec::encoder::mp3_encoder::Mp3Encoder,
    w: &mut BufWriter<File>,
) -> AudioIOResult<()> {
    use crate::codec::encoder::encoder_interface::AudioEncoder;
    use crate::codec::error::CodecError;
    loop {
        match enc.receive_packet() {
            Ok(pkt) => w.write_all(&pkt.data)?,
            Err(CodecError::Again) => break,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

pub struct Mp3Reader {
    data: Vec<u8>,
    pos: usize,
    dec: crate::codec::decoder::mp3_decoder::Mp3Decoder,
    time_base: Rational,
    // ffmpeg parser only under feature
    #[cfg(feature = "ffmpeg")]
    parser: *mut ffmpeg_sys_next::AVCodecParserContext,
    #[cfg(feature = "ffmpeg")]
    dec_ctx: *mut ffmpeg_sys_next::AVCodecContext,
    flushed: bool,
}

#[cfg(feature = "ffmpeg")]
unsafe impl Send for Mp3Reader {}

impl Mp3Reader {
    pub fn open<P: AsRef<Path>>(path: P) -> AudioIOResult<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        // decoder time_base：用采样率为单位更直观，先设一个占位，后面收到首帧再修正
        let time_base = Rational::new(1, 48_000);
        let start_pos = skip_id3v2(&data);

        #[cfg(feature = "ffmpeg")]
        unsafe {
            extern crate ffmpeg_sys_next as ff;
            let dec = crate::codec::decoder::mp3_decoder::Mp3Decoder::new()?;
            // 这里需要 parser + AVCodecContext 来 parse2；我们从 decoder 的内部 ctx 拿不到，
            // 所以先在 file 层自己建一个 codec ctx 仅用于 parser。
            let codec = ff::avcodec_find_decoder_by_name(std::ffi::CString::new("mp3").unwrap().as_ptr());
            if codec.is_null() {
                return Err(AudioIOError::Codec(crate::codec::error::CodecError::Unsupported(
                    "FFmpeg MP3 decoder not found",
                )));
            }
            let dec_ctx = ff::avcodec_alloc_context3(codec);
            if dec_ctx.is_null() {
                return Err(AudioIOError::Codec(crate::codec::error::CodecError::Other(
                    "avcodec_alloc_context3 failed".into(),
                )));
            }
            // parser 需要 codec id
            let parser = ff::av_parser_init((*codec).id as i32);
            if parser.is_null() {
                ff::avcodec_free_context(&mut (dec_ctx as *mut _));
                return Err(AudioIOError::Format("av_parser_init failed"));
            }
            return Ok(Self {
                data,
                pos: start_pos,
                dec,
                time_base,
                parser,
                dec_ctx,
                flushed: false,
            });
        }

        #[cfg(not(feature = "ffmpeg"))]
        {
            let dec = crate::codec::decoder::mp3_decoder::Mp3Decoder::new()?;
            Ok(Self {
                data,
                pos: start_pos,
                dec,
                time_base,
                flushed: false,
            })
        }
    }
}

impl Drop for Mp3Reader {
    fn drop(&mut self) {
        #[cfg(feature = "ffmpeg")]
        unsafe {
            extern crate ffmpeg_sys_next as ff;
            if !self.parser.is_null() {
                ff::av_parser_close(self.parser);
                self.parser = ptr::null_mut();
            }
            if !self.dec_ctx.is_null() {
                ff::avcodec_free_context(&mut self.dec_ctx);
            }
        }
    }
}

impl AudioReader for Mp3Reader {
    fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
        use crate::codec::decoder::decoder_interface::AudioDecoder;
        use crate::codec::error::CodecError;

        loop {
            match self.dec.receive_frame() {
                Ok(f) => return Ok(Some(f)),
                Err(CodecError::Again) => { /* need more input */ }
                Err(CodecError::Eof) => return Ok(None),
                Err(e) => return Err(e.into()),
            }

            // feed more data
            if let Some(pkt) = self.next_packet()? {
                self.dec.send_packet(Some(pkt))?;
            } else if !self.flushed {
                self.dec.send_packet(None)?;
                self.flushed = true;
            }
        }
    }
}

impl Mp3Reader {
    fn next_packet(&mut self) -> AudioIOResult<Option<crate::codec::packet::CodecPacket>> {
        #[cfg(not(feature = "ffmpeg"))]
        {
            return Err(AudioIOError::Codec(crate::codec::error::CodecError::Unsupported(
                "MP3 reader requires ffmpeg feature (parser)",
            )));
        }

        #[cfg(feature = "ffmpeg")]
        unsafe {
            extern crate ffmpeg_sys_next as ff;
            use crate::codec::packet::CodecPacket;

            while self.pos < self.data.len() {
                let in_ptr = self.data[self.pos..].as_ptr() as *mut u8;
                let in_len = (self.data.len() - self.pos) as i32;
                let mut out_ptr: *mut u8 = ptr::null_mut();
                let mut out_len: i32 = 0;

                let consumed = ff::av_parser_parse2(
                    self.parser,
                    self.dec_ctx,
                    &mut out_ptr,
                    &mut out_len,
                    in_ptr,
                    in_len,
                    ff::AV_NOPTS_VALUE,
                    ff::AV_NOPTS_VALUE,
                    0,
                );

                if consumed < 0 {
                    return Err(AudioIOError::Format("mp3 parse failed"));
                }
                if consumed == 0 && out_len == 0 {
                    // 防止极端情况下死循环：无法前进则结束
                    self.pos = self.data.len();
                    break;
                }
                self.pos += consumed as usize;

                if out_len > 0 && !out_ptr.is_null() {
                    let slice = core::slice::from_raw_parts(out_ptr as *const u8, out_len as usize);
                    // 避免把 ID3/tag/垃圾数据喂给 decoder：要求以 MP3 帧同步字开头
                    if !looks_like_mp3_frame(slice) {
                        continue;
                    }
                    return Ok(Some(CodecPacket {
                        data: slice.to_vec(),
                        time_base: self.time_base,
                        pts: None,
                        dts: None,
                        duration: None,
                        flags: crate::codec::packet::PacketFlags::empty(),
                    }));
                }
            }
            Ok(None)
        }
    }
}

fn looks_like_mp3_frame(buf: &[u8]) -> bool {
    // MP3 frame sync: 11 bits set (0xFFE)
    buf.len() >= 2 && buf[0] == 0xFF && (buf[1] & 0xE0) == 0xE0
}

fn skip_id3v2(data: &[u8]) -> usize {
    // ID3v2 header: "ID3" + ver(2) + flags(1) + size(4 syncsafe) = 10 bytes
    if data.len() < 10 {
        return 0;
    }
    if &data[0..3] != b"ID3" {
        return 0;
    }
    let sz = syncsafe_u32(&data[6..10]) as usize;
    (10usize.saturating_add(sz)).min(data.len())
}

fn syncsafe_u32(b: &[u8]) -> u32 {
    if b.len() < 4 {
        return 0;
    }
    ((b[0] as u32 & 0x7F) << 21)
        | ((b[1] as u32 & 0x7F) << 14)
        | ((b[2] as u32 & 0x7F) << 7)
        | (b[3] as u32 & 0x7F)
}

