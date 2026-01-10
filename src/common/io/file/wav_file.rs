use crate::common::io::io::{AudioReader, AudioWriter, AudioIOResult, AudioIOError};
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView, ChannelLayout, Rational, SampleFormat};
use crate::codec::encoder::wav_encoder::WavEncoderConfig;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// WAV 写出样本格式（容器侧参数）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavOutputSampleFormat {
    /// PCM 16-bit little-endian（audio_format=1, bits=16）
    Pcm16Le,
    /// IEEE float 32-bit little-endian（audio_format=3, bits=32）
    Float32Le,
}

/// WAV 文件写入配置：编码/输入参数（encoder）+ writer/容器侧参数（output_format）。
#[derive(Clone, Debug)]
pub struct WavWriterConfig {
    pub encoder: WavEncoderConfig,
    pub output_format: WavOutputSampleFormat,
}

impl WavWriterConfig {
    /// 默认行为：写出 PCM16LE（与旧实现一致）。
    pub fn pcm16le(input_format: AudioFormat) -> Self {
        Self {
            encoder: WavEncoderConfig { input_format },
            output_format: WavOutputSampleFormat::Pcm16Le,
        }
    }

    pub fn f32le(input_format: AudioFormat) -> Self {
        Self {
            encoder: WavEncoderConfig { input_format },
            output_format: WavOutputSampleFormat::Float32Le,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WavFmt {
    audio_format: u16, // 1=PCM, 3=IEEE float
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

fn read_u16_le(r: &mut impl Read) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32_le(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_u16_le(w: &mut impl Write, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u32_le(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub struct WavWriter {
    w: BufWriter<File>,
    cfg: WavWriterConfig,
    data_bytes: u32,
    header_written: bool,
}

impl WavWriter {
    pub fn create<P: AsRef<Path>>(path: P, cfg: WavWriterConfig) -> AudioIOResult<Self> {
        Ok(Self {
            w: BufWriter::new(File::create(path)?),
            cfg,
            data_bytes: 0,
            header_written: false,
        })
    }

    fn write_header(&mut self) -> AudioIOResult<()> {
        if self.header_written {
            return Ok(());
        }
        // RIFF header
        self.w.write_all(b"RIFF")?;
        write_u32_le(&mut self.w, 0)?; // placeholder riff size
        self.w.write_all(b"WAVE")?;

        // fmt chunk (PCM16LE)
        self.w.write_all(b"fmt ")?;
        write_u32_le(&mut self.w, 16)?; // PCM fmt size
        let (audio_format, bits_per_sample) = match self.cfg.output_format {
            WavOutputSampleFormat::Pcm16Le => (1u16, 16u16),
            WavOutputSampleFormat::Float32Le => (3u16, 32u16),
        };
        write_u16_le(&mut self.w, audio_format)?;
        write_u16_le(&mut self.w, self.cfg.encoder.input_format.channels())?;
        write_u32_le(&mut self.w, self.cfg.encoder.input_format.sample_rate)?;

        let block_align = self.cfg.encoder.input_format.channels() * (bits_per_sample / 8);
        let byte_rate = self.cfg.encoder.input_format.sample_rate * (block_align as u32);
        write_u32_le(&mut self.w, byte_rate)?;
        write_u16_le(&mut self.w, block_align)?;
        write_u16_le(&mut self.w, bits_per_sample)?;

        // data chunk
        self.w.write_all(b"data")?;
        write_u32_le(&mut self.w, 0)?; // placeholder data size

        self.header_written = true;
        Ok(())
    }

    fn write_samples_i16_interleaved(&mut self, samples: &[i16]) -> AudioIOResult<()> {
        let bytes = (samples.len() * 2) as u32;
        self.w.write_all(bytemuck_i16_as_u8(samples))?;
        self.data_bytes = self.data_bytes.saturating_add(bytes);
        Ok(())
    }

    fn write_samples_f32_interleaved(&mut self, samples: &[f32]) -> AudioIOResult<()> {
        let bytes = (samples.len() * 4) as u32;
        self.w.write_all(bytemuck_f32_as_u8(samples))?;
        self.data_bytes = self.data_bytes.saturating_add(bytes);
        Ok(())
    }
}

impl AudioWriter for WavWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        self.write_header()?;
        let fmt = frame.format();
        let ch = fmt.channels() as usize;
        if ch == 0 {
            return Err(AudioIOError::Format("channels=0"));
        }
        if fmt.sample_rate != self.cfg.encoder.input_format.sample_rate
            || fmt.channels() != self.cfg.encoder.input_format.channels()
        {
            return Err(AudioIOError::Format("WAV writer expects fixed sample_rate/channels (no resample layer yet)"));
        }

        let ns = frame.nb_samples();
        match self.cfg.output_format {
            WavOutputSampleFormat::Pcm16Le => {
                // 转为 interleaved PCM16
                let mut out: Vec<i16> = Vec::with_capacity(ns * ch);
                match fmt.sample_format {
                    SampleFormat::I16 { planar: false } => {
                        let plane = frame.plane(0).ok_or(AudioIOError::Format("missing plane 0"))?;
                        if plane.len() != ns * ch * 2 {
                            return Err(AudioIOError::Format("unexpected i16 interleaved plane size"));
                        }
                        // 直接拷贝
                        for i in 0..(ns * ch) {
                            let off = i * 2;
                            out.push(i16::from_le_bytes([plane[off], plane[off + 1]]));
                        }
                    }
                    SampleFormat::I16 { planar: true } => {
                        for s in 0..ns {
                            for c in 0..ch {
                                let p = frame.plane(c).ok_or(AudioIOError::Format("missing plane"))?;
                                let off = s * 2;
                                out.push(i16::from_le_bytes([p[off], p[off + 1]]));
                            }
                        }
                    }
                    SampleFormat::F32 { planar: false } => {
                        let plane = frame.plane(0).ok_or(AudioIOError::Format("missing plane 0"))?;
                        if plane.len() != ns * ch * 4 {
                            return Err(AudioIOError::Format("unexpected f32 interleaved plane size"));
                        }
                        for i in 0..(ns * ch) {
                            let off = i * 4;
                            let v = f32::from_le_bytes([plane[off], plane[off + 1], plane[off + 2], plane[off + 3]]);
                            out.push(float_to_i16(v));
                        }
                    }
                    SampleFormat::F32 { planar: true } => {
                        for s in 0..ns {
                            for c in 0..ch {
                                let p = frame.plane(c).ok_or(AudioIOError::Format("missing plane"))?;
                                let off = s * 4;
                                let v = f32::from_le_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]]);
                                out.push(float_to_i16(v));
                            }
                        }
                    }
                    _ => return Err(AudioIOError::Format("WAV writer supports only I16/F32 input currently")),
                }
                self.write_samples_i16_interleaved(&out)?;
            }
            WavOutputSampleFormat::Float32Le => {
                // 转为 interleaved F32LE
                let mut out: Vec<f32> = Vec::with_capacity(ns * ch);
                match fmt.sample_format {
                    SampleFormat::F32 { planar: false } => {
                        let plane = frame.plane(0).ok_or(AudioIOError::Format("missing plane 0"))?;
                        if plane.len() != ns * ch * 4 {
                            return Err(AudioIOError::Format("unexpected f32 interleaved plane size"));
                        }
                        for i in 0..(ns * ch) {
                            let off = i * 4;
                            out.push(f32::from_le_bytes([plane[off], plane[off + 1], plane[off + 2], plane[off + 3]]));
                        }
                    }
                    SampleFormat::F32 { planar: true } => {
                        for s in 0..ns {
                            for c in 0..ch {
                                let p = frame.plane(c).ok_or(AudioIOError::Format("missing plane"))?;
                                let off = s * 4;
                                out.push(f32::from_le_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]]));
                            }
                        }
                    }
                    SampleFormat::I16 { planar: false } => {
                        let plane = frame.plane(0).ok_or(AudioIOError::Format("missing plane 0"))?;
                        if plane.len() != ns * ch * 2 {
                            return Err(AudioIOError::Format("unexpected i16 interleaved plane size"));
                        }
                        for i in 0..(ns * ch) {
                            let off = i * 2;
                            let v = i16::from_le_bytes([plane[off], plane[off + 1]]);
                            out.push(i16_to_f32(v));
                        }
                    }
                    SampleFormat::I16 { planar: true } => {
                        for s in 0..ns {
                            for c in 0..ch {
                                let p = frame.plane(c).ok_or(AudioIOError::Format("missing plane"))?;
                                let off = s * 2;
                                let v = i16::from_le_bytes([p[off], p[off + 1]]);
                                out.push(i16_to_f32(v));
                            }
                        }
                    }
                    _ => return Err(AudioIOError::Format("WAV writer supports only I16/F32 input currently")),
                }
                self.write_samples_f32_interleaved(&out)?;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        self.w.flush()?;

        // 回写 RIFF/data size
        let mut f = self.w.get_ref().try_clone()?;
        // RIFF chunk size at offset 4: file_size - 8
        let file_size = (44u32).saturating_add(self.data_bytes);
        let riff_size = file_size.saturating_sub(8);
        f.seek(SeekFrom::Start(4))?;
        f.write_all(&riff_size.to_le_bytes())?;
        // data chunk size at offset 40 (固定 PCM fmt 16 bytes 情况)
        f.seek(SeekFrom::Start(40))?;
        f.write_all(&self.data_bytes.to_le_bytes())?;
        Ok(())
    }
}

pub struct WavReader {
    r: BufReader<File>,
    fmt: WavFmt,
    data_len: u32,
    data_pos: u32,
    time_base: Rational,
    next_pts_samples: i64,
    chunk_samples: usize,
}

/// WAV reader 读取配置（主要控制 `next_frame()` 每次吐出的样本数）。
#[derive(Clone, Copy, Debug)]
pub struct WavReaderConfig {
    /// 每次 `next_frame()` 返回的最大 samples/每声道（最后一帧可能更小）。
    pub chunk_samples: usize,
}

impl Default for WavReaderConfig {
    fn default() -> Self {
        Self { chunk_samples: 1024 }
    }
}

impl WavReader {
    pub fn open<P: AsRef<Path>>(path: P) -> AudioIOResult<Self> {
        Self::open_with_config(path, WavReaderConfig::default())
    }

    pub fn open_with_config<P: AsRef<Path>>(path: P, cfg: WavReaderConfig) -> AudioIOResult<Self> {
        let mut r = BufReader::new(File::open(path)?);

        let mut riff = [0u8; 4];
        r.read_exact(&mut riff)?;
        if &riff != b"RIFF" {
            return Err(AudioIOError::Format("not RIFF"));
        }
        let _riff_size = read_u32_le(&mut r)?;
        let mut wave = [0u8; 4];
        r.read_exact(&mut wave)?;
        if &wave != b"WAVE" {
            return Err(AudioIOError::Format("not WAVE"));
        }

        let mut fmt: Option<WavFmt> = None;
        let mut data_start: Option<u64> = None;
        let mut data_len: Option<u32> = None;

        loop {
            let mut id = [0u8; 4];
            if r.read_exact(&mut id).is_err() {
                break;
            }
            let sz = read_u32_le(&mut r)?;
            let here = r.stream_position()?;
            match &id {
                b"fmt " => {
                    let audio_format = read_u16_le(&mut r)?;
                    let channels = read_u16_le(&mut r)?;
                    let sample_rate = read_u32_le(&mut r)?;
                    let _byte_rate = read_u32_le(&mut r)?;
                    let _block_align = read_u16_le(&mut r)?;
                    let bits_per_sample = read_u16_le(&mut r)?;
                    // 跳过多余 fmt 扩展
                    if sz > 16 {
                        r.seek(SeekFrom::Current((sz - 16) as i64))?;
                    }
                    fmt = Some(WavFmt {
                        audio_format,
                        channels,
                        sample_rate,
                        bits_per_sample,
                    });
                }
                b"data" => {
                    data_start = Some(r.stream_position()?);
                    data_len = Some(sz);
                    // 先跳过 data，后面再 seek 回来开始读
                    r.seek(SeekFrom::Current(sz as i64))?;
                }
                _ => {
                    // skip unknown chunk (pad to even)
                    let skip = sz + (sz % 2);
                    r.seek(SeekFrom::Start(here + skip as u64))?;
                }
            }
        }

        let fmt = fmt.ok_or(AudioIOError::Format("missing fmt chunk"))?;
        let data_start = data_start.ok_or(AudioIOError::Format("missing data chunk"))?;
        let data_len = data_len.ok_or(AudioIOError::Format("missing data chunk size"))?;

        // 定位到 data
        r.seek(SeekFrom::Start(data_start))?;

        let time_base = Rational::new(1, fmt.sample_rate as i32);
        Ok(Self {
            r,
            fmt,
            data_len,
            data_pos: 0,
            time_base,
            next_pts_samples: 0,
            chunk_samples: cfg.chunk_samples.max(1),
        })
    }

    pub fn format(&self) -> AudioFormat {
        // 对外统一输出 planar F32
        AudioFormat {
            sample_rate: self.fmt.sample_rate,
            sample_format: SampleFormat::F32 { planar: true },
            channel_layout: ChannelLayout::default_for_channels(self.fmt.channels),
        }
    }
}

impl AudioReader for WavReader {
    fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
        if self.data_pos >= self.data_len {
            return Ok(None);
        }
        let ch = self.fmt.channels as usize;
        let bps = match (self.fmt.audio_format, self.fmt.bits_per_sample) {
            (1, 16) => 2, // PCM16
            (3, 32) => 4, // float32
            _ => return Err(AudioIOError::Format("unsupported WAV format (only PCM16LE / Float32LE)")),
        };
        let bytes_per_sample_all_ch = ch * bps;
        let remain = (self.data_len - self.data_pos) as usize;
        let max_samples = remain / bytes_per_sample_all_ch;
        if max_samples == 0 {
            self.data_pos = self.data_len;
            return Ok(None);
        }
        let ns = max_samples.min(self.chunk_samples);
        let mut buf = vec![0u8; ns * bytes_per_sample_all_ch];
        self.r.read_exact(&mut buf)?;
        self.data_pos += buf.len() as u32;

        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(ch);
        for _ in 0..ch {
            planes.push(Vec::with_capacity(ns * 4));
        }

        if bps == 2 {
            for s in 0..ns {
                for c in 0..ch {
                    let off = (s * ch + c) * 2;
                    let v = i16::from_le_bytes([buf[off], buf[off + 1]]);
                    let f = (v as f32) / 32768.0;
                    planes[c].extend_from_slice(&f.to_le_bytes());
                }
            }
        } else {
            for s in 0..ns {
                for c in 0..ch {
                    let off = (s * ch + c) * 4;
                    let f = f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
                    planes[c].extend_from_slice(&f.to_le_bytes());
                }
            }
        }

        let out = AudioFrame::from_planes(
            self.format(),
            ns,
            self.time_base,
            Some(self.next_pts_samples),
            planes,
        )
        .map_err(|_| AudioIOError::Format("failed to build AudioFrame"))?;

        self.next_pts_samples += ns as i64;
        Ok(Some(out))
    }
}

fn float_to_i16(v: f32) -> i16 {
    let x = (v.max(-1.0).min(1.0) * 32767.0).round() as i32;
    x.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn i16_to_f32(v: i16) -> f32 {
    // [-32768, 32767] -> [-1.0, 1.0)
    (v as f32) / 32768.0
}

fn bytemuck_i16_as_u8(v: &[i16]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

fn bytemuck_f32_as_u8(v: &[f32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

