//! ResampleProcessor：PCM->PCM 重采样 processor（流式）。
//!
//! - 默认：纯 Rust 线性重采样（不抗混叠，够用来解除 Opus 48k 输入限制）
//! - feature=ffmpeg：可选 swresample backend（质量/覆盖面更好）

use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView, AudioFrameViewMut, Rational, SampleType};
use crate::common::audio::fifo::AudioFifo;
use std::collections::VecDeque;

use crate::function::resample::LinearResampler;

#[cfg(feature = "ffmpeg")]
use crate::function::resample::ffmpeg_backend::FfmpegResampler;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResampleBackend {
    /// 纯 Rust 线性插值（最快接入、无依赖；downsample 时不抗混叠）。
    Linear,
    /// FFmpeg swresample（需要 `--features ffmpeg`）。
    #[cfg(feature = "ffmpeg")]
    Ffmpeg,
}

pub struct ResampleProcessor {
    backend: ResampleBackend,
    in_fmt: AudioFormat,
    out_fmt: AudioFormat,
    // 线性 backend 状态：按声道一个 resampler（即使 interleaved 也按声道拆开处理）
    linear: Option<Vec<LinearResampler>>,
    // ffmpeg backend
    #[cfg(feature = "ffmpeg")]
    ff: Option<FfmpegResampler>,

    // 可选：把输出通过 FIFO “重分帧”为固定帧长（例如 Opus 常用 960@48k）
    out_chunk_samples: Option<usize>,
    out_pad_final: bool,
    out_fifo: Option<AudioFifo>,

    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

impl ResampleProcessor {
    /// 创建一个 resample processor（默认使用纯 Rust 线性后端）。
    ///
    /// 约束（线性后端）：
    /// - 采样类型必须为 f32/i16/i32
    /// - planar/interleaved 必须一致（不做布局转换）
    /// - channels 必须一致（不做 remix）
    pub fn new_linear(in_fmt: AudioFormat, out_fmt: AudioFormat) -> CodecResult<Self> {
        validate_linear_formats(in_fmt, out_fmt)?;
        let ch = in_fmt.channels() as usize;
        let mut rs = Vec::with_capacity(ch);
        for _ in 0..ch {
            rs.push(
                LinearResampler::new(in_fmt.sample_rate, out_fmt.sample_rate)
                    .map_err(|_| CodecError::InvalidData("invalid sample_rate"))?,
            );
        }
        Ok(Self {
            backend: ResampleBackend::Linear,
            in_fmt,
            out_fmt,
            linear: Some(rs),
            #[cfg(feature = "ffmpeg")]
            ff: None,
            out_chunk_samples: None,
            out_pad_final: false,
            out_fifo: None,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    /// 创建一个 resample processor（优先使用 FFmpeg swresample；失败则报错）。
    #[cfg(feature = "ffmpeg")]
    pub fn new_ffmpeg(in_fmt: AudioFormat, out_fmt: AudioFormat) -> CodecResult<Self> {
        let ff = FfmpegResampler::new(in_fmt, out_fmt)
            .map_err(|e| CodecError::Other(format!("ffmpeg resample init failed: {e}")))?;
        Ok(Self {
            backend: ResampleBackend::Ffmpeg,
            in_fmt,
            out_fmt,
            linear: None,
            ff: Some(ff),
            out_chunk_samples: None,
            out_pad_final: false,
            out_fifo: None,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    /// 自动选择后端：
    /// - 有 ffmpeg：优先 ffmpeg
    /// - 否则：linear
    pub fn new(in_fmt: AudioFormat, out_fmt: AudioFormat) -> CodecResult<Self> {
        #[cfg(feature = "ffmpeg")]
        {
            // swresample 覆盖面更好：允许更多采样率/采样格式组合
            if let Ok(p) = Self::new_ffmpeg(in_fmt, out_fmt) {
                return Ok(p);
            }
        }
        Self::new_linear(in_fmt, out_fmt)
    }

    pub fn backend(&self) -> ResampleBackend {
        self.backend
    }

    /// 启用/关闭输出“重分帧”：
    /// - `chunk_samples=Some(n)`: 输出会被缓存并按 n samples（每声道）固定弹出
    /// - `pad_final=true`: flush 时如果剩余样本不足 n，会补 0 到 n
    /// - 如果 chunk_samples 为 None，则不启用重分帧
    /// 
    /// 典型用法：Opus encoder 前设置 `chunk_samples=Some(960), pad_final=true`。
    pub fn set_output_chunker(&mut self, chunk_samples: Option<usize>, pad_final: bool) -> CodecResult<()> {
        self.out_chunk_samples = chunk_samples;
        self.out_pad_final = pad_final;
        if let Some(n) = chunk_samples {
            if n == 0 {
                return Err(CodecError::InvalidData("chunk_samples must be > 0"));
            }
            let tb = Rational::new(1, self.out_fmt.sample_rate as i32);
            let fifo = AudioFifo::new(self.out_fmt, tb)
                .map_err(|e| CodecError::Other(format!("AudioFifo init failed: {e}")))?;
            self.out_fifo = Some(fifo);
        } else {
            self.out_fifo = None;
        }
        Ok(())
    }
}

impl AudioProcessor for ResampleProcessor {
    fn name(&self) -> &'static str {
        match self.backend {
            ResampleBackend::Linear => "resample(linear)",
            #[cfg(feature = "ffmpeg")]
            ResampleBackend::Ffmpeg => "resample(ffmpeg)",
        }
    }

    fn input_format(&self) -> Option<AudioFormat> {
        Some(self.in_fmt)
    }

    fn output_format(&self) -> Option<AudioFormat> {
        Some(self.out_fmt)
    }

    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }

        if frame.is_none() {
            // flush
            self.flushed = true;
            match self.backend {
                ResampleBackend::Linear => {
                    let Some(rs) = self.linear.as_mut() else {
                        return Err(CodecError::InvalidState("linear resampler not initialized"));
                    };
                    // 每声道补齐尾部
                    let (nb_out, outs) = match self.out_fmt.sample_format.sample_type() {
                        SampleType::F32 => {
                            let mut ch_out: Vec<Vec<f32>> = Vec::with_capacity(rs.len());
                            for r in rs.iter_mut() {
                                ch_out.push(r.flush_hold_last::<f32>());
                            }
                            let nb_out = ch_out.get(0).map(|v| v.len()).unwrap_or(0);
                            let outs = encode_output_planes::<f32>(&ch_out, self.out_fmt)?;
                            (nb_out, outs)
                        }
                        SampleType::I16 => {
                            let mut ch_out: Vec<Vec<i16>> = Vec::with_capacity(rs.len());
                            for r in rs.iter_mut() {
                                ch_out.push(r.flush_hold_last::<i16>());
                            }
                            let nb_out = ch_out.get(0).map(|v| v.len()).unwrap_or(0);
                            let outs = encode_output_planes::<i16>(&ch_out, self.out_fmt)?;
                            (nb_out, outs)
                        }
                        SampleType::I32 => {
                            let mut ch_out: Vec<Vec<i32>> = Vec::with_capacity(rs.len());
                            for r in rs.iter_mut() {
                                ch_out.push(r.flush_hold_last::<i32>());
                            }
                            let nb_out = ch_out.get(0).map(|v| v.len()).unwrap_or(0);
                            let outs = encode_output_planes::<i32>(&ch_out, self.out_fmt)?;
                            (nb_out, outs)
                        }
                        _ => return Err(CodecError::Unsupported("linear resample supports only f32/i16/i32")),
                    };
                    if nb_out != 0 {
                        let out_tb = Rational::new(1, self.out_fmt.sample_rate as i32);
                        let f = AudioFrame::from_planes(self.out_fmt, nb_out, out_tb, None, outs)
                            .map_err(|_| CodecError::Other("failed to build AudioFrame".into()))?;
                        self.enqueue_output_frame(f)?;
                    }
                }
                #[cfg(feature = "ffmpeg")]
                ResampleBackend::Ffmpeg => {
                    // swresample 需要 flush：这里先不产生额外输出（留给后续扩展 swr_convert(NULL)）
                }
            }
            // flush 时处理 FIFO 尾巴
            self.flush_output_chunker()?;
            return Ok(());
        }

        let frame = frame.unwrap();
        if frame.format() != self.in_fmt {
            return Err(CodecError::InvalidData("ResampleProcessor input AudioFormat mismatch"));
        }

        match self.backend {
            ResampleBackend::Linear => self.send_linear(frame),
            #[cfg(feature = "ffmpeg")]
            ResampleBackend::Ffmpeg => self.send_ffmpeg(frame),
        }
    }

    fn receive_frame(&mut self) -> CodecResult<AudioFrame> {
        if let Some(f) = self.out_q.pop_front() {
            return Ok(f);
        }
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        self.out_q.clear();
        self.flushed = false;
        if let Some(fifo) = self.out_fifo.as_mut() {
            fifo.clear();
        }
        if let Some(rs) = self.linear.as_mut() {
            for r in rs.iter_mut() {
                r.reset();
            }
        }
        #[cfg(feature = "ffmpeg")]
        if let Some(ff) = self.ff.as_mut() {
            ff.reset()
                .map_err(|e| CodecError::Other(format!("ffmpeg resample reset failed: {e}")))?;
        }
        Ok(())
    }
}

impl ResampleProcessor {
    fn send_linear(&mut self, frame: &dyn AudioFrameView) -> CodecResult<()> {
        // 线性重采样目前仅支持 sample_type=f32/i16/i32，且布局/声道一致
        validate_linear_formats(self.in_fmt, self.out_fmt)?;

        let Some(rs) = self.linear.as_mut() else {
            return Err(CodecError::InvalidState("linear resampler not initialized"));
        };

        let in_nb = frame.nb_samples();
        if in_nb == 0 {
            return Ok(());
        }

        let out_tb = Rational::new(1, self.out_fmt.sample_rate as i32);
        let pts = frame.pts().and_then(|p| rescale_pts(p, frame.time_base(), out_tb));

        let channels = self.in_fmt.channels() as usize;
        let maybe_out = match self.in_fmt.sample_format.sample_type() {
            SampleType::F32 => linear_process_frame::<f32>(frame, rs, channels, self.out_fmt, out_tb, pts)?,
            SampleType::I16 => linear_process_frame::<i16>(frame, rs, channels, self.out_fmt, out_tb, pts)?,
            SampleType::I32 => linear_process_frame::<i32>(frame, rs, channels, self.out_fmt, out_tb, pts)?,
            _ => return Err(CodecError::Unsupported("linear resample supports only f32/i16/i32")),
        };
        if let Some(out) = maybe_out {
            self.enqueue_output_frame(out)?;
        }
        Ok(())
    }

    #[cfg(feature = "ffmpeg")]
    fn send_ffmpeg(&mut self, frame: &dyn AudioFrameView) -> CodecResult<()> {
        let Some(ff) = self.ff.as_mut() else {
            return Err(CodecError::InvalidState("ffmpeg resampler not initialized"));
        };
        let out = ff
            .process_frame(frame)
            .map_err(|e| CodecError::Other(format!("ffmpeg resample failed: {e}")))?;
        self.enqueue_output_frame(out)?;
        Ok(())
    }

    fn enqueue_output_frame(&mut self, frame: AudioFrame) -> CodecResult<()> {
        if let Some(chunk) = self.out_chunk_samples {
            let fifo = self
                .out_fifo
                .as_mut()
                .ok_or(CodecError::InvalidState("output chunker enabled but fifo missing"))?;
            fifo.push_frame(&frame)
                .map_err(|e| CodecError::Other(format!("AudioFifo push failed: {e}")))?;
            loop {
                match fifo.pop_frame(chunk) {
                    Ok(Some(f)) => self.out_q.push_back(f),
                    Ok(None) => break,
                    Err(e) => return Err(CodecError::Other(format!("AudioFifo pop failed: {e}"))),
                }
            }
            Ok(())
        } else {
            self.out_q.push_back(frame);
            Ok(())
        }
    }

    fn flush_output_chunker(&mut self) -> CodecResult<()> {
        let Some(chunk) = self.out_chunk_samples else {
            return Ok(());
        };
        let Some(fifo) = self.out_fifo.as_mut() else {
            return Ok(());
        };
        let remain = fifo.available_samples();
        if remain == 0 {
            return Ok(());
        }
        if !self.out_pad_final {
            fifo.clear();
            return Ok(());
        }

        // 取出剩余并 pad 0 到 chunk_samples
        let Some(partial) = fifo
            .pop_frame(remain)
            .map_err(|e| CodecError::Other(format!("AudioFifo pop tail failed: {e}")))?
        else {
            return Ok(());
        };
        fifo.clear();

        // 直接重建一个 chunk 长度的帧，后半部分保持 0
        let mut padded = AudioFrame::new_alloc(self.out_fmt, chunk)
            .map_err(|_| CodecError::Other("failed to alloc padded AudioFrame".into()))?;
        padded.set_time_base(partial.time_base()).map_err(|e| CodecError::Other(format!("{e}")))?;
        padded.set_pts(partial.pts());

        let fmt = self.out_fmt;
        let bps = fmt.sample_format.bytes_per_sample();
        if fmt.is_planar() {
            let ch = fmt.channels() as usize;
            for c in 0..ch {
                let src = partial.plane(c).ok_or(CodecError::InvalidData("missing partial plane"))?;
                let dst = padded.plane_mut(c).ok_or(CodecError::InvalidData("missing padded plane"))?;
                let bytes = remain * bps;
                dst[..bytes].copy_from_slice(&src[..bytes]);
            }
        } else {
            let src = partial.plane(0).ok_or(CodecError::InvalidData("missing partial plane 0"))?;
            let dst = padded.plane_mut(0).ok_or(CodecError::InvalidData("missing padded plane 0"))?;
            let bytes = remain * (fmt.channels() as usize) * bps;
            dst[..bytes].copy_from_slice(&src[..bytes]);
        }

        self.out_q.push_back(padded);
        Ok(())
    }
}

fn linear_process_frame<T: crate::function::resample::Sample>(
    frame: &dyn AudioFrameView,
    rs: &mut [LinearResampler],
    channels: usize,
    out_fmt: AudioFormat,
    out_tb: Rational,
    pts: Option<i64>,
) -> CodecResult<Option<AudioFrame>> {
    let in_ch: Vec<Vec<T>> = decode_input_planes::<T>(frame, channels)?;
    let mut out_ch: Vec<Vec<T>> = Vec::with_capacity(channels);
    for c in 0..channels {
        out_ch.push(rs[c].process::<T>(&in_ch[c]));
    }
    let nb_out = out_ch.get(0).map(|v| v.len()).unwrap_or(0);
    if nb_out == 0 {
        return Ok(None);
    }
    let planes = encode_output_planes::<T>(&out_ch, out_fmt)?;
    let out = AudioFrame::from_planes(out_fmt, nb_out, out_tb, pts, planes)
        .map_err(|_| CodecError::Other("failed to build AudioFrame".into()))?;
    Ok(Some(out))
}

fn validate_linear_formats(in_fmt: AudioFormat, out_fmt: AudioFormat) -> CodecResult<()> {
    if in_fmt.channels() != out_fmt.channels() {
        return Err(CodecError::Unsupported("linear resample does not remix channels"));
    }
    if in_fmt.sample_format.is_planar() != out_fmt.sample_format.is_planar() {
        return Err(CodecError::Unsupported("linear resample does not change planar/interleaved layout"));
    }
    if in_fmt.sample_format.sample_type() != out_fmt.sample_format.sample_type() {
        return Err(CodecError::Unsupported("linear resample does not change sample type"));
    }
    match in_fmt.sample_format.sample_type() {
        SampleType::F32 | SampleType::I16 | SampleType::I32 => Ok(()),
        _ => Err(CodecError::Unsupported("linear resample supports only f32/i16/i32")),
    }
}

fn rescale_pts(pts: i64, tb_in: Rational, tb_out: Rational) -> Option<i64> {
    if tb_in.den == 0 || tb_out.den == 0 || tb_out.num == 0 {
        return None;
    }
    let num = (pts as i128)
        .checked_mul(tb_in.num as i128)?
        .checked_mul(tb_out.den as i128)?;
    let den = (tb_in.den as i128).checked_mul(tb_out.num as i128)?;
    Some((num / den) as i64)
}

fn decode_input_planes<T: crate::function::resample::Sample>(
    frame: &dyn AudioFrameView,
    channels: usize,
) -> CodecResult<Vec<Vec<T>>> {
    let fmt = frame.format();
    let bps = fmt.sample_format.bytes_per_sample();
    let nb = frame.nb_samples();
    if fmt.is_planar() {
        let mut out = Vec::with_capacity(channels);
        for c in 0..channels {
            let p = frame.plane(c).ok_or(CodecError::InvalidData("missing plane"))?;
            if p.len() != nb * bps {
                return Err(CodecError::InvalidData("unexpected plane size"));
            }
            out.push(decode_bytes::<T>(p, fmt.sample_format.sample_type())?);
        }
        Ok(out)
    } else {
        let p = frame.plane(0).ok_or(CodecError::InvalidData("missing plane 0"))?;
        if p.len() != nb * channels * bps {
            return Err(CodecError::InvalidData("unexpected interleaved plane size"));
        }
        let all = decode_bytes::<T>(p, fmt.sample_format.sample_type())?;
        if all.len() != nb * channels {
            return Err(CodecError::InvalidData("unexpected decoded sample count"));
        }
        let mut out = vec![Vec::with_capacity(nb); channels];
        for i in 0..nb {
            for c in 0..channels {
                out[c].push(all[i * channels + c]);
            }
        }
        Ok(out)
    }
}

fn encode_output_planes<T: crate::function::resample::Sample>(
    ch_out: &[Vec<T>],
    out_fmt: AudioFormat,
) -> CodecResult<Vec<Vec<u8>>> {
    let channels = out_fmt.channels() as usize;
    if channels == 0 {
        return Err(CodecError::InvalidData("channels=0"));
    }
    if ch_out.len() != channels {
        return Err(CodecError::InvalidData("channel count mismatch"));
    }
    let nb = ch_out[0].len();
    for c in 1..channels {
        if ch_out[c].len() != nb {
            return Err(CodecError::InvalidData("channel output length mismatch"));
        }
    }
    if out_fmt.is_planar() {
        let mut planes = Vec::with_capacity(channels);
        for c in 0..channels {
            planes.push(encode_bytes::<T>(&ch_out[c], out_fmt.sample_format.sample_type()));
        }
        Ok(planes)
    } else {
        let mut interleaved: Vec<T> = Vec::with_capacity(nb * channels);
        for i in 0..nb {
            for c in 0..channels {
                interleaved.push(ch_out[c][i]);
            }
        }
        Ok(vec![encode_bytes::<T>(
            &interleaved,
            out_fmt.sample_format.sample_type(),
        )])
    }
}

fn decode_bytes<T: crate::function::resample::Sample>(bytes: &[u8], ty: SampleType) -> CodecResult<Vec<T>> {
    match ty {
        SampleType::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for ch in bytes.chunks_exact(4) {
                let v = f32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]);
                out.push(T::from_f32(v));
            }
            Ok(out)
        }
        SampleType::I16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for ch in bytes.chunks_exact(2) {
                let v = i16::from_ne_bytes([ch[0], ch[1]]) as f32;
                out.push(T::from_f32(v));
            }
            Ok(out)
        }
        SampleType::I32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for ch in bytes.chunks_exact(4) {
                let v = i32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]) as f32;
                out.push(T::from_f32(v));
            }
            Ok(out)
        }
        _ => Err(CodecError::Unsupported("unsupported sample type for linear resample")),
    }
}

fn encode_bytes<T: crate::function::resample::Sample>(samples: &[T], ty: SampleType) -> Vec<u8> {
    match ty {
        SampleType::F32 => {
            let mut out = Vec::with_capacity(samples.len() * 4);
            for &s in samples {
                let v = s.to_f32().to_ne_bytes();
                out.extend_from_slice(&v);
            }
            out
        }
        SampleType::I16 => {
            let mut out = Vec::with_capacity(samples.len() * 2);
            for &s in samples {
                let v = (s.to_f32().round() as i16).to_ne_bytes();
                out.extend_from_slice(&v);
            }
            out
        }
        SampleType::I32 => {
            let mut out = Vec::with_capacity(samples.len() * 4);
            for &s in samples {
                let v = (s.to_f32().round() as i32).to_ne_bytes();
                out.extend_from_slice(&v);
            }
            out
        }
        _ => Vec::new(),
    }
}


