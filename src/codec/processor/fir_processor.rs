//! FirProcessor：PCM->PCM FIR 滤波器（流式）
//!
//! - 不改变 AudioFormat
//! - 支持 planar / interleaved
//! - 使用每声道独立的 FIR 状态（共享同一组 taps）

use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView, SampleType};
use std::collections::VecDeque;
#[cfg(feature = "rustfft")]
use std::sync::Arc;

#[cfg(feature = "rustfft")]
use rustfft::{num_complex::Complex, num_traits::Zero, Fft, FftPlanner};

#[cfg(feature = "wide")]
use wide::f64x4;

#[cfg(feature = "rustfft")]
const FFT_MIN_TAPS: usize = 256;
#[cfg(feature = "rustfft")]
const FFT_TARGET_BLOCK: usize = 1024;

pub struct FirProcessor {
    taps: Vec<f64>,
    // 如果为 None：吃到首帧后再锁定格式（并初始化 state）
    fmt: Option<AudioFormat>,
    // 是否由构造参数固定（true => reset 不清空；false => reset 清空推断）
    locked: bool,
    // 每声道 FIR 状态：保存最近的 taps_len-1 个输入样本（x[n-1], x[n-2], ...）
    states: Vec<Vec<f64>>,
    // 每声道环形缓冲区写指针
    state_pos: Vec<usize>,
    #[cfg(feature = "rustfft")]
    fft_state: Option<FftConvState>,
    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

#[cfg(feature = "rustfft")]
struct FftConvState {
    fft_len: usize,
    block_len: usize,
    overlap_len: usize,
    taps_fft: Vec<Complex<f64>>,
    overlap: Vec<Vec<f64>>,
    work: Vec<Vec<Complex<f64>>>,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    scratch_fft: Vec<Complex<f64>>,
    scratch_ifft: Vec<Complex<f64>>,
}

#[cfg(feature = "rustfft")]
impl FftConvState {
    fn new(taps: &[f64], channels: usize, fft_len: usize) -> Self {
        let overlap_len = taps.len().saturating_sub(1);
        let block_len = fft_len.saturating_sub(overlap_len);
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_len);
        let ifft = planner.plan_fft_inverse(fft_len);

        let mut taps_fft = vec![Complex::zero(); fft_len];
        for (i, &v) in taps.iter().enumerate() {
            taps_fft[i].re = v;
        }
        let mut scratch_fft = vec![Complex::zero(); fft.get_inplace_scratch_len()];
        fft.process_with_scratch(&mut taps_fft, &mut scratch_fft);

        let overlap = vec![vec![0.0; overlap_len]; channels];
        let work = vec![vec![Complex::zero(); fft_len]; channels];
        let scratch_ifft = vec![Complex::zero(); ifft.get_inplace_scratch_len()];

        Self {
            fft_len,
            block_len,
            overlap_len,
            taps_fft,
            overlap,
            work,
            fft,
            ifft,
            scratch_fft,
            scratch_ifft,
        }
    }

    fn reset(&mut self) {
        for buf in &mut self.overlap {
            for v in buf.iter_mut() {
                *v = 0.0;
            }
        }
    }

    fn process_channel(&mut self, channel: usize, input: &[f64], output: &mut [f64]) {
        let mut offset = 0usize;
        let scale = 1.0 / (self.fft_len as f64);
        while offset < input.len() {
            let blk = self.block_len.min(input.len() - offset);
            let buf = &mut self.work[channel];
            for v in buf.iter_mut() {
                *v = Complex::zero();
            }
            if self.overlap_len > 0 {
                for i in 0..self.overlap_len {
                    buf[i].re = self.overlap[channel][i];
                }
            }
            for i in 0..blk {
                buf[self.overlap_len + i].re = input[offset + i];
            }
            self.fft.process_with_scratch(buf, &mut self.scratch_fft);
            for i in 0..self.fft_len {
                buf[i] = buf[i] * self.taps_fft[i];
            }
            self.ifft.process_with_scratch(buf, &mut self.scratch_ifft);
            for i in 0..blk {
                output[offset + i] = buf[self.overlap_len + i].re * scale;
            }
            if self.overlap_len > 0 {
                if blk >= self.overlap_len {
                    self.overlap[channel].copy_from_slice(
                        &input[offset + blk - self.overlap_len..offset + blk],
                    );
                } else {
                    self.overlap[channel].copy_within(blk.., 0);
                    let start = self.overlap_len - blk;
                    self.overlap[channel][start..].copy_from_slice(&input[offset..offset + blk]);
                }
            }
            offset += blk;
        }
    }
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
            state_pos: Vec::new(),
            #[cfg(feature = "rustfft")]
            fft_state: None,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    /// 创建并固定输入/输出格式（后续如果输入格式不匹配会报错）。
    pub fn new_with_format(fmt: AudioFormat, taps: Vec<f32>) -> CodecResult<Self> {
        let taps = validate_taps(taps)?;
        let (states, state_pos) = init_states(fmt, taps.len());
        Ok(Self {
            taps,
            fmt: Some(fmt),
            locked: true,
            states,
            state_pos,
            #[cfg(feature = "rustfft")]
            fft_state: None,
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
            let (states, state_pos) = init_states(fmt, self.taps.len());
            self.states = states;
            self.state_pos = state_pos;
        }
        Ok(())
    }

    #[cfg(feature = "rustfft")]
    fn should_use_fft(&self, nb_samples: usize) -> bool {
        let taps_len = self.taps.len();
        taps_len >= FFT_MIN_TAPS || taps_len > nb_samples
    }

    #[cfg(feature = "rustfft")]
    fn ensure_fft_state(&mut self, nb_samples: usize) -> CodecResult<()> {
        if self.fft_state.is_some() || !self.should_use_fft(nb_samples) {
            return Ok(());
        }
        let fmt = self.fmt.ok_or(CodecError::InvalidState("format not set"))?;
        let channels = fmt.channels() as usize;
        if channels == 0 {
            return Err(CodecError::InvalidData("channels=0"));
        }
        let fft_len = select_fft_len(self.taps.len(), nb_samples);
        self.fft_state = Some(FftConvState::new(&self.taps, channels, fft_len));
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

        #[cfg(feature = "rustfft")]
        self.ensure_fft_state(nb_samples)?;

        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(expected_plane_count);
        #[cfg(feature = "rustfft")]
        let use_fft = self.fft_state.is_some();
        #[cfg(not(feature = "rustfft"))]
        let use_fft = false;

        if fmt.is_planar() {
            for c in 0..channels {
                let p = frame
                    .plane(c)
                    .ok_or(CodecError::InvalidData("missing plane in AudioFrameView"))?;
                if p.len() != expected_bytes {
                    return Err(CodecError::InvalidData("unexpected plane byte size"));
                }
                let out = if use_fft {
                    #[cfg(feature = "rustfft")]
                    {
                        let fft_state = self.fft_state.as_mut().expect("fft_state must exist");
                        process_planar_bytes_fft(p, fmt.sample_format.sample_type(), fft_state, c)?
                    }
                    #[cfg(not(feature = "rustfft"))]
                    {
                        unreachable!("fft disabled")
                    }
                } else {
                    process_planar_bytes(
                        p,
                        fmt.sample_format.sample_type(),
                        &self.taps,
                        &mut self.states[c],
                        &mut self.state_pos[c],
                    )?
                };
                planes.push(out);
            }
        } else {
            let p = frame
                .plane(0)
                .ok_or(CodecError::InvalidData("missing plane in AudioFrameView"))?;
            if p.len() != expected_bytes {
                return Err(CodecError::InvalidData("unexpected plane byte size"));
            }
            let out = if use_fft {
                #[cfg(feature = "rustfft")]
                {
                    let fft_state = self.fft_state.as_mut().expect("fft_state must exist");
                    process_interleaved_bytes_fft(
                        p,
                        fmt.sample_format.sample_type(),
                        channels,
                        fft_state,
                    )?
                }
                #[cfg(not(feature = "rustfft"))]
                {
                    unreachable!("fft disabled")
                }
            } else {
                process_interleaved_bytes(
                    p,
                    fmt.sample_format.sample_type(),
                    channels,
                    &self.taps,
                    &mut self.states,
                    &mut self.state_pos,
                )?
            };
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
        for pos in &mut self.state_pos {
            *pos = 0;
        }
        #[cfg(feature = "rustfft")]
        if let Some(state) = &mut self.fft_state {
            state.reset();
        }
        if !self.locked {
            self.fmt = None;
            self.states.clear();
            self.state_pos.clear();
            #[cfg(feature = "rustfft")]
            {
                self.fft_state = None;
            }
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

fn init_states(fmt: AudioFormat, taps_len: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let channels = fmt.channels() as usize;
    let delay = taps_len.saturating_sub(1);
    (vec![vec![0.0; delay]; channels], vec![0; channels])
}

#[cfg(feature = "rustfft")]
fn select_fft_len(taps_len: usize, nb_samples: usize) -> usize {
    let overlap_len = taps_len.saturating_sub(1);
    let target_block = FFT_TARGET_BLOCK.max(nb_samples.min(FFT_TARGET_BLOCK * 4));
    next_pow2(overlap_len + target_block.max(1))
}

#[cfg(feature = "rustfft")]
fn next_pow2(mut n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if usize::BITS == 64 {
        n |= n >> 32;
    }
    n + 1
}

#[cfg(not(feature = "wide"))]
#[inline]
fn dot_fir_tail(taps_tail: &[f64], state: &[f64], state_pos: usize) -> f64 {
    if state.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut idx = if state_pos == 0 {
        state.len() - 1
    } else {
        state_pos - 1
    };
    for k in 0..state.len() {
        acc += taps_tail[k] * state[idx];
        idx = if idx == 0 { state.len() - 1 } else { idx - 1 };
    }
    acc
}

#[cfg(feature = "wide")]
#[inline]
fn dot_fir_tail_simd(taps_tail: &[f64], state: &[f64], state_pos: usize) -> f64 {
    if state.is_empty() {
        return 0.0;
    }
    let mut idx = if state_pos == 0 {
        state.len() - 1
    } else {
        state_pos - 1
    };
    let mut acc = f64x4::ZERO;
    let mut i = 0usize;
    while i + 4 <= state.len() {
        let s0 = state[idx];
        idx = if idx == 0 { state.len() - 1 } else { idx - 1 };
        let s1 = state[idx];
        idx = if idx == 0 { state.len() - 1 } else { idx - 1 };
        let s2 = state[idx];
        idx = if idx == 0 { state.len() - 1 } else { idx - 1 };
        let s3 = state[idx];
        idx = if idx == 0 { state.len() - 1 } else { idx - 1 };
        let tv = f64x4::new([taps_tail[i], taps_tail[i + 1], taps_tail[i + 2], taps_tail[i + 3]]);
        let sv = f64x4::new([s0, s1, s2, s3]);
        acc = acc + tv * sv;
        i += 4;
    }
    let mut sum = acc.to_array().iter().sum::<f64>();
    while i < state.len() {
        sum += taps_tail[i] * state[idx];
        idx = if idx == 0 { state.len() - 1 } else { idx - 1 };
        i += 1;
    }
    sum
}

#[inline]
fn apply_fir_sample(x: f64, taps: &[f64], state: &mut [f64], state_pos: &mut usize) -> f64 {
    let taps_tail = &taps[1..];
    let mut acc = taps[0] * x;
    #[cfg(feature = "wide")]
    {
        acc += dot_fir_tail_simd(taps_tail, state, *state_pos);
    }
    #[cfg(not(feature = "wide"))]
    {
        acc += dot_fir_tail(taps_tail, state, *state_pos);
    }
    if !state.is_empty() {
        let n = state.len();
        state[*state_pos] = x;
        *state_pos = if *state_pos + 1 == n { 0 } else { *state_pos + 1 };
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
    state_pos: &mut usize,
) -> CodecResult<Vec<u8>> {
    match sample_type {
        SampleType::U8 => {
            let mut out = Vec::with_capacity(input.len());
            for &b in input {
                let x = (b as f64) - 128.0;
                let y = apply_fir_sample(x, taps, state, state_pos);
                out.push(clamp_u8_pcm(y + 128.0));
            }
            Ok(out)
        }
        SampleType::I16 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(2) {
                let raw = i16::from_ne_bytes([ch[0], ch[1]]);
                let y = apply_fir_sample(raw as f64, taps, state, state_pos);
                out.extend_from_slice(&clamp_i16(y).to_ne_bytes());
            }
            Ok(out)
        }
        SampleType::I32 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(4) {
                let raw = i32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]);
                let y = apply_fir_sample(raw as f64, taps, state, state_pos);
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
                let y = apply_fir_sample(raw as f64, taps, state, state_pos);
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
                let y = apply_fir_sample(x, taps, state, state_pos) as f32;
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
                let y = apply_fir_sample(x, taps, state, state_pos);
                let out_v = if y.is_finite() { y } else { 0.0 };
                out.extend_from_slice(&out_v.to_ne_bytes());
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "rustfft")]
fn bytes_per_sample(sample_type: SampleType) -> usize {
    match sample_type {
        SampleType::U8 => 1,
        SampleType::I16 => 2,
        SampleType::I32 => 4,
        SampleType::I64 => 8,
        SampleType::F32 => 4,
        SampleType::F64 => 8,
    }
}

#[cfg(feature = "rustfft")]
fn planar_bytes_to_f64(input: &[u8], sample_type: SampleType) -> Vec<f64> {
    match sample_type {
        SampleType::U8 => input.iter().map(|&b| (b as f64) - 128.0).collect(),
        SampleType::I16 => input
            .chunks_exact(2)
            .map(|ch| i16::from_ne_bytes([ch[0], ch[1]]) as f64)
            .collect(),
        SampleType::I32 => input
            .chunks_exact(4)
            .map(|ch| i32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]) as f64)
            .collect(),
        SampleType::I64 => input
            .chunks_exact(8)
            .map(|ch| {
                i64::from_ne_bytes([ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7]]) as f64
            })
            .collect(),
        SampleType::F32 => input
            .chunks_exact(4)
            .map(|ch| {
                let mut x = f32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]) as f64;
                if !x.is_finite() {
                    x = 0.0;
                }
                x
            })
            .collect(),
        SampleType::F64 => input
            .chunks_exact(8)
            .map(|ch| {
                let mut x = f64::from_ne_bytes([ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7]]);
                if !x.is_finite() {
                    x = 0.0;
                }
                x
            })
            .collect(),
    }
}

#[cfg(feature = "rustfft")]
fn planar_f64_to_bytes(samples: &[f64], sample_type: SampleType) -> Vec<u8> {
    match sample_type {
        SampleType::U8 => samples.iter().map(|&v| clamp_u8_pcm(v + 128.0)).collect(),
        SampleType::I16 => {
            let mut out = Vec::with_capacity(samples.len() * 2);
            for &v in samples {
                out.extend_from_slice(&clamp_i16(v).to_ne_bytes());
            }
            out
        }
        SampleType::I32 => {
            let mut out = Vec::with_capacity(samples.len() * 4);
            for &v in samples {
                out.extend_from_slice(&clamp_i32(v).to_ne_bytes());
            }
            out
        }
        SampleType::I64 => {
            let mut out = Vec::with_capacity(samples.len() * 8);
            for &v in samples {
                out.extend_from_slice(&clamp_i64(v).to_ne_bytes());
            }
            out
        }
        SampleType::F32 => {
            let mut out = Vec::with_capacity(samples.len() * 4);
            for &v in samples {
                let vv = if v.is_finite() { v as f32 } else { 0.0 };
                out.extend_from_slice(&vv.to_ne_bytes());
            }
            out
        }
        SampleType::F64 => {
            let mut out = Vec::with_capacity(samples.len() * 8);
            for &v in samples {
                let vv = if v.is_finite() { v } else { 0.0 };
                out.extend_from_slice(&vv.to_ne_bytes());
            }
            out
        }
    }
}

#[cfg(feature = "rustfft")]
fn process_planar_bytes_fft(
    input: &[u8],
    sample_type: SampleType,
    fft_state: &mut FftConvState,
    channel: usize,
) -> CodecResult<Vec<u8>> {
    let samples = planar_bytes_to_f64(input, sample_type);
    let mut out = vec![0.0; samples.len()];
    fft_state.process_channel(channel, &samples, &mut out);
    Ok(planar_f64_to_bytes(&out, sample_type))
}

fn process_interleaved_bytes(
    input: &[u8],
    sample_type: SampleType,
    channels: usize,
    taps: &[f64],
    states: &mut [Vec<f64>],
    state_pos: &mut [usize],
) -> CodecResult<Vec<u8>> {
    if channels == 0 {
        return Err(CodecError::InvalidData("channels=0"));
    }
    if states.len() != channels {
        return Err(CodecError::InvalidState("fir states length mismatch"));
    }
    if state_pos.len() != channels {
        return Err(CodecError::InvalidState("fir state_pos length mismatch"));
    }

    match sample_type {
        SampleType::U8 => {
            let mut out = Vec::with_capacity(input.len());
            let mut idx = 0usize;
            while idx < input.len() {
                for c in 0..channels {
                    let b = input[idx];
                    let x = (b as f64) - 128.0;
                    let y = apply_fir_sample(x, taps, &mut states[c], &mut state_pos[c]);
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
                    let y = apply_fir_sample(raw as f64, taps, &mut states[c], &mut state_pos[c]);
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
                    let y = apply_fir_sample(raw as f64, taps, &mut states[c], &mut state_pos[c]);
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
                    let y = apply_fir_sample(raw as f64, taps, &mut states[c], &mut state_pos[c]);
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
                    let y = apply_fir_sample(x, taps, &mut states[c], &mut state_pos[c]) as f32;
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
                    let y = apply_fir_sample(x, taps, &mut states[c], &mut state_pos[c]);
                    let out_v = if y.is_finite() { y } else { 0.0 };
                    out.extend_from_slice(&out_v.to_ne_bytes());
                    i += 8;
                }
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "rustfft")]
fn process_interleaved_bytes_fft(
    input: &[u8],
    sample_type: SampleType,
    channels: usize,
    fft_state: &mut FftConvState,
) -> CodecResult<Vec<u8>> {
    if channels == 0 {
        return Err(CodecError::InvalidData("channels=0"));
    }
    let bps = bytes_per_sample(sample_type);
    let frame_bytes = bps * channels;
    if frame_bytes == 0 || input.len() % frame_bytes != 0 {
        return Err(CodecError::InvalidData("unexpected interleaved byte size"));
    }
    let nb_samples = input.len() / frame_bytes;
    let mut chans = vec![vec![0.0; nb_samples]; channels];
    let mut idx = 0usize;
    for s in 0..nb_samples {
        for c in 0..channels {
            let v = match sample_type {
                SampleType::U8 => {
                    let b = input[idx];
                    idx += 1;
                    (b as f64) - 128.0
                }
                SampleType::I16 => {
                    let raw = i16::from_ne_bytes([input[idx], input[idx + 1]]);
                    idx += 2;
                    raw as f64
                }
                SampleType::I32 => {
                    let raw = i32::from_ne_bytes([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);
                    idx += 4;
                    raw as f64
                }
                SampleType::I64 => {
                    let raw = i64::from_ne_bytes([
                        input[idx],
                        input[idx + 1],
                        input[idx + 2],
                        input[idx + 3],
                        input[idx + 4],
                        input[idx + 5],
                        input[idx + 6],
                        input[idx + 7],
                    ]);
                    idx += 8;
                    raw as f64
                }
                SampleType::F32 => {
                    let mut x = f32::from_ne_bytes([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]) as f64;
                    idx += 4;
                    if !x.is_finite() {
                        x = 0.0;
                    }
                    x
                }
                SampleType::F64 => {
                    let mut x = f64::from_ne_bytes([
                        input[idx],
                        input[idx + 1],
                        input[idx + 2],
                        input[idx + 3],
                        input[idx + 4],
                        input[idx + 5],
                        input[idx + 6],
                        input[idx + 7],
                    ]);
                    idx += 8;
                    if !x.is_finite() {
                        x = 0.0;
                    }
                    x
                }
            };
            chans[c][s] = v;
        }
    }

    for c in 0..channels {
        let mut out = vec![0.0; nb_samples];
        fft_state.process_channel(c, &chans[c], &mut out);
        chans[c] = out;
    }

    let mut out_bytes = Vec::with_capacity(input.len());
    for s in 0..nb_samples {
        for c in 0..channels {
            let v = chans[c][s];
            match sample_type {
                SampleType::U8 => out_bytes.push(clamp_u8_pcm(v + 128.0)),
                SampleType::I16 => out_bytes.extend_from_slice(&clamp_i16(v).to_ne_bytes()),
                SampleType::I32 => out_bytes.extend_from_slice(&clamp_i32(v).to_ne_bytes()),
                SampleType::I64 => out_bytes.extend_from_slice(&clamp_i64(v).to_ne_bytes()),
                SampleType::F32 => {
                    let vv = if v.is_finite() { v as f32 } else { 0.0 };
                    out_bytes.extend_from_slice(&vv.to_ne_bytes());
                }
                SampleType::F64 => {
                    let vv = if v.is_finite() { v } else { 0.0 };
                    out_bytes.extend_from_slice(&vv.to_ne_bytes());
                }
            }
        }
    }
    Ok(out_bytes)
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
