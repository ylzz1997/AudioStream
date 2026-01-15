//! FirProcessor：PCM->PCM FIR 滤波器（流式）
//!
//! - 不改变 AudioFormat
//! - 支持 planar / interleaved
//! - 使用每声道独立的 FIR 状态（共享同一组 taps）

use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView, SampleType};
use std::collections::VecDeque;

pub struct FirProcessor {
    taps: Vec<f64>,
    // 如果为 None：吃到首帧后再锁定格式（并初始化 state）
    fmt: Option<AudioFormat>,
    // 是否由构造参数固定（true => reset 不清空；false => reset 清空推断）
    locked: bool,
    // 每声道 FIR 状态：保存最近的 taps_len-1 个输入样本（x[n-1], x[n-2], ...）
    states: Vec<Vec<f64>>,
    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

impl FirProcessor {
    /// 创建 FIR processor（taps 为滤波器系数，h[0] 对应当前样本）。
    pub fn new(taps: Vec<f32>) -> CodecResult<Self> {
        let taps = validate_taps(taps)?;
        Ok(Self {
            taps,
            fmt: None,
            locked: false,
            states: Vec::new(),
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    /// 创建并固定输入/输出格式（后续如果输入格式不匹配会报错）。
    pub fn new_with_format(fmt: AudioFormat, taps: Vec<f32>) -> CodecResult<Self> {
        let taps = validate_taps(taps)?;
        let states = init_states(fmt, taps.len());
        Ok(Self {
            taps,
            fmt: Some(fmt),
            locked: true,
            states,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    pub fn taps(&self) -> &[f64] {
        &self.taps
    }

    fn ensure_format(&mut self, frame: &dyn AudioFrameView) -> CodecResult<()> {
        if let Some(expected) = self.fmt {
            let actual = frame.format();
            if actual != expected {
                eprintln!(
                    "FirProcessor input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                    crate::common::audio::audio::audio_format_diff(expected, actual)
                );
                return Err(CodecError::InvalidData("FirProcessor input AudioFormat mismatch"));
            }
        } else {
            self.fmt = Some(frame.format());
            self.locked = false;
        }

        if self.states.is_empty() {
            let fmt = self.fmt.expect("format should be set");
            self.states = init_states(fmt, self.taps.len());
        }
        Ok(())
    }

    fn process_frame(&mut self, frame: &dyn AudioFrameView) -> CodecResult<AudioFrame> {
        let fmt = frame.format();
        let nb_samples = frame.nb_samples();
        let plane_count = frame.plane_count();

        let expected_plane_count = AudioFrame::expected_plane_count(&fmt);
        if plane_count != expected_plane_count {
            return Err(CodecError::InvalidData("unexpected plane_count for input AudioFormat"));
        }

        let expected_bytes = AudioFrame::expected_bytes_per_plane(&fmt, nb_samples);
        let channels = fmt.channels() as usize;
        if channels == 0 {
            return Err(CodecError::InvalidData("channels=0"));
        }

        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(expected_plane_count);
        if fmt.is_planar() {
            for c in 0..channels {
                let p = frame
                    .plane(c)
                    .ok_or(CodecError::InvalidData("missing plane in AudioFrameView"))?;
                if p.len() != expected_bytes {
                    return Err(CodecError::InvalidData("unexpected plane byte size"));
                }
                let out = process_planar_bytes(p, fmt.sample_format.sample_type(), &self.taps, &mut self.states[c])?;
                planes.push(out);
            }
        } else {
            let p = frame
                .plane(0)
                .ok_or(CodecError::InvalidData("missing plane in AudioFrameView"))?;
            if p.len() != expected_bytes {
                return Err(CodecError::InvalidData("unexpected plane byte size"));
            }
            let out = process_interleaved_bytes(
                p,
                fmt.sample_format.sample_type(),
                channels,
                &self.taps,
                &mut self.states,
            )?;
            planes.push(out);
        }

        AudioFrame::from_planes(fmt, nb_samples, frame.time_base(), frame.pts(), planes)
            .map_err(|_| CodecError::InvalidData("failed to build AudioFrame from planes"))
    }
}

impl AudioProcessor for FirProcessor {
    fn name(&self) -> &'static str {
        "fir"
    }

    fn input_format(&self) -> Option<AudioFormat> {
        self.fmt
    }

    fn output_format(&self) -> Option<AudioFormat> {
        self.fmt
    }

    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }

        // 严格对齐接口语义：如果输出队列未取空，让调用方先 receive_frame()。
        if !self.out_q.is_empty() {
            return Err(CodecError::Again);
        }

        let Some(frame) = frame else {
            self.flushed = true;
            return Ok(());
        };

        if frame.nb_samples() == 0 {
            return Ok(());
        }

        self.ensure_format(frame)?;
        let out = self.process_frame(frame)?;
        self.out_q.push_back(out);
        Ok(())
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
        for st in &mut self.states {
            for v in st.iter_mut() {
                *v = 0.0;
            }
        }
        if !self.locked {
            self.fmt = None;
            self.states.clear();
        }
        Ok(())
    }
}

fn validate_taps(taps: Vec<f32>) -> CodecResult<Vec<f64>> {
    if taps.is_empty() {
        return Err(CodecError::InvalidData("taps must not be empty"));
    }
    let mut out = Vec::with_capacity(taps.len());
    for &v in &taps {
        if !v.is_finite() {
            return Err(CodecError::InvalidData("taps must be finite"));
        }
        out.push(v as f64);
    }
    Ok(out)
}

fn init_states(fmt: AudioFormat, taps_len: usize) -> Vec<Vec<f64>> {
    let channels = fmt.channels() as usize;
    let delay = taps_len.saturating_sub(1);
    vec![vec![0.0; delay]; channels]
}

#[inline]
fn apply_fir_sample(x: f64, taps: &[f64], state: &mut [f64]) -> f64 {
    let mut acc = taps[0] * x;
    for (k, s) in state.iter().enumerate() {
        acc += taps[k + 1] * *s;
    }
    if !state.is_empty() {
        for i in (1..state.len()).rev() {
            state[i] = state[i - 1];
        }
        state[0] = x;
    }
    if acc.is_finite() {
        acc
    } else {
        0.0
    }
}

fn process_planar_bytes(
    input: &[u8],
    sample_type: SampleType,
    taps: &[f64],
    state: &mut [f64],
) -> CodecResult<Vec<u8>> {
    match sample_type {
        SampleType::U8 => {
            let mut out = Vec::with_capacity(input.len());
            for &b in input {
                let x = (b as f64) - 128.0;
                let y = apply_fir_sample(x, taps, state);
                out.push(clamp_u8_pcm(y + 128.0));
            }
            Ok(out)
        }
        SampleType::I16 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(2) {
                let raw = i16::from_ne_bytes([ch[0], ch[1]]);
                let y = apply_fir_sample(raw as f64, taps, state);
                out.extend_from_slice(&clamp_i16(y).to_ne_bytes());
            }
            Ok(out)
        }
        SampleType::I32 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(4) {
                let raw = i32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]);
                let y = apply_fir_sample(raw as f64, taps, state);
                out.extend_from_slice(&clamp_i32(y).to_ne_bytes());
            }
            Ok(out)
        }
        SampleType::I64 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(8) {
                let raw = i64::from_ne_bytes([
                    ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7],
                ]);
                let y = apply_fir_sample(raw as f64, taps, state);
                out.extend_from_slice(&clamp_i64(y).to_ne_bytes());
            }
            Ok(out)
        }
        SampleType::F32 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(4) {
                let mut x = f32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]) as f64;
                if !x.is_finite() {
                    x = 0.0;
                }
                let y = apply_fir_sample(x, taps, state) as f32;
                let out_v = if y.is_finite() { y } else { 0.0 };
                out.extend_from_slice(&out_v.to_ne_bytes());
            }
            Ok(out)
        }
        SampleType::F64 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(8) {
                let mut x = f64::from_ne_bytes([
                    ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7],
                ]);
                if !x.is_finite() {
                    x = 0.0;
                }
                let y = apply_fir_sample(x, taps, state);
                let out_v = if y.is_finite() { y } else { 0.0 };
                out.extend_from_slice(&out_v.to_ne_bytes());
            }
            Ok(out)
        }
    }
}

fn process_interleaved_bytes(
    input: &[u8],
    sample_type: SampleType,
    channels: usize,
    taps: &[f64],
    states: &mut [Vec<f64>],
) -> CodecResult<Vec<u8>> {
    if channels == 0 {
        return Err(CodecError::InvalidData("channels=0"));
    }
    if states.len() != channels {
        return Err(CodecError::InvalidState("fir states length mismatch"));
    }

    match sample_type {
        SampleType::U8 => {
            let mut out = Vec::with_capacity(input.len());
            let mut idx = 0usize;
            while idx < input.len() {
                for c in 0..channels {
                    let b = input[idx];
                    let x = (b as f64) - 128.0;
                    let y = apply_fir_sample(x, taps, &mut states[c]);
                    out.push(clamp_u8_pcm(y + 128.0));
                    idx += 1;
                }
            }
            Ok(out)
        }
        SampleType::I16 => {
            let mut out = Vec::with_capacity(input.len());
            let mut i = 0usize;
            while i < input.len() {
                for c in 0..channels {
                    let raw = i16::from_ne_bytes([input[i], input[i + 1]]);
                    let y = apply_fir_sample(raw as f64, taps, &mut states[c]);
                    out.extend_from_slice(&clamp_i16(y).to_ne_bytes());
                    i += 2;
                }
            }
            Ok(out)
        }
        SampleType::I32 => {
            let mut out = Vec::with_capacity(input.len());
            let mut i = 0usize;
            while i < input.len() {
                for c in 0..channels {
                    let raw = i32::from_ne_bytes([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    let y = apply_fir_sample(raw as f64, taps, &mut states[c]);
                    out.extend_from_slice(&clamp_i32(y).to_ne_bytes());
                    i += 4;
                }
            }
            Ok(out)
        }
        SampleType::I64 => {
            let mut out = Vec::with_capacity(input.len());
            let mut i = 0usize;
            while i < input.len() {
                for c in 0..channels {
                    let raw = i64::from_ne_bytes([
                        input[i],
                        input[i + 1],
                        input[i + 2],
                        input[i + 3],
                        input[i + 4],
                        input[i + 5],
                        input[i + 6],
                        input[i + 7],
                    ]);
                    let y = apply_fir_sample(raw as f64, taps, &mut states[c]);
                    out.extend_from_slice(&clamp_i64(y).to_ne_bytes());
                    i += 8;
                }
            }
            Ok(out)
        }
        SampleType::F32 => {
            let mut out = Vec::with_capacity(input.len());
            let mut i = 0usize;
            while i < input.len() {
                for c in 0..channels {
                    let mut x = f32::from_ne_bytes([input[i], input[i + 1], input[i + 2], input[i + 3]]) as f64;
                    if !x.is_finite() {
                        x = 0.0;
                    }
                    let y = apply_fir_sample(x, taps, &mut states[c]) as f32;
                    let out_v = if y.is_finite() { y } else { 0.0 };
                    out.extend_from_slice(&out_v.to_ne_bytes());
                    i += 4;
                }
            }
            Ok(out)
        }
        SampleType::F64 => {
            let mut out = Vec::with_capacity(input.len());
            let mut i = 0usize;
            while i < input.len() {
                for c in 0..channels {
                    let mut x = f64::from_ne_bytes([
                        input[i],
                        input[i + 1],
                        input[i + 2],
                        input[i + 3],
                        input[i + 4],
                        input[i + 5],
                        input[i + 6],
                        input[i + 7],
                    ]);
                    if !x.is_finite() {
                        x = 0.0;
                    }
                    let y = apply_fir_sample(x, taps, &mut states[c]);
                    let out_v = if y.is_finite() { y } else { 0.0 };
                    out.extend_from_slice(&out_v.to_ne_bytes());
                    i += 8;
                }
            }
            Ok(out)
        }
    }
}

#[inline]
fn clamp_i64(v: f64) -> i64 {
    if v.is_nan() {
        return 0;
    }
    if v >= i64::MAX as f64 {
        i64::MAX
    } else if v <= i64::MIN as f64 {
        i64::MIN
    } else {
        v.round() as i64
    }
}

#[inline]
fn clamp_i32(v: f64) -> i32 {
    if v.is_nan() {
        return 0;
    }
    if v >= i32::MAX as f64 {
        i32::MAX
    } else if v <= i32::MIN as f64 {
        i32::MIN
    } else {
        v.round() as i32
    }
}

#[inline]
fn clamp_i16(v: f64) -> i16 {
    if v.is_nan() {
        return 0;
    }
    if v >= i16::MAX as f64 {
        i16::MAX
    } else if v <= i16::MIN as f64 {
        i16::MIN
    } else {
        v.round() as i16
    }
}

#[inline]
fn clamp_u8_pcm(v: f64) -> u8 {
    // PCM U8 通常以 128 为“零点”（unsigned offset binary）
    if v.is_nan() {
        return 128;
    }
    if v >= 255.0 {
        255
    } else if v <= 0.0 {
        0
    } else {
        v.round() as u8
    }
}
