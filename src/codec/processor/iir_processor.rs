//! IirProcessor：PCM->PCM IIR 滤波器（流式）
//!
//! - 不改变 AudioFormat
//! - 支持 planar / interleaved
//! - 使用每声道独立的 IIR 状态（共享同一组系数）

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
const IIR_FFT_MIN_TAPS: usize = 256;
#[cfg(feature = "rustfft")]
const IIR_FFT_TARGET_BLOCK: usize = 1024;
#[cfg(feature = "rustfft")]
const IIR_FFT_IMPULSE_LEN: usize = 8192;

pub struct IirProcessor {
    b: Vec<f64>,
    a: Vec<f64>,
    // 如果为 None：吃到首帧后再锁定格式（并初始化 state）
    fmt: Option<AudioFormat>,
    // 是否由构造参数固定（true => reset 不清空；false => reset 清空推断）
    locked: bool,
    // 每声道 IIR 状态：x[n-1], x[n-2], ... / y[n-1], y[n-2], ...
    x_states: Vec<Vec<f64>>,
    y_states: Vec<Vec<f64>>,
    // 每声道环形缓冲区写指针
    x_pos: Vec<usize>,
    y_pos: Vec<usize>,
    #[cfg(feature = "rustfft")]
    fft_state: Option<IirFftState>,
    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

#[cfg(feature = "rustfft")]
struct IirFftState {
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
impl IirFftState {
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

impl IirProcessor {
    /// 创建 IIR processor（b 为前向系数，a 为反馈系数；a[0] 会被归一化为 1）。
    pub fn new(b: Vec<f32>, a: Vec<f32>) -> CodecResult<Self> {
        let (b, a) = validate_coeffs(b, a)?;
        Ok(Self {
            b,
            a,
            fmt: None,
            locked: false,
            x_states: Vec::new(),
            y_states: Vec::new(),
            x_pos: Vec::new(),
            y_pos: Vec::new(),
            #[cfg(feature = "rustfft")]
            fft_state: None,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    /// 创建并固定输入/输出格式（后续如果输入格式不匹配会报错）。
    pub fn new_with_format(fmt: AudioFormat, b: Vec<f32>, a: Vec<f32>) -> CodecResult<Self> {
        let (b, a) = validate_coeffs(b, a)?;
        let (x_states, y_states, x_pos, y_pos) = init_states(fmt, b.len(), a.len());
        Ok(Self {
            b,
            a,
            fmt: Some(fmt),
            locked: true,
            x_states,
            y_states,
            x_pos,
            y_pos,
            #[cfg(feature = "rustfft")]
            fft_state: None,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    pub fn b(&self) -> &[f64] {
        &self.b
    }

    pub fn a(&self) -> &[f64] {
        &self.a
    }

    fn ensure_format(&mut self, frame: &dyn AudioFrameView) -> CodecResult<()> {
        if let Some(expected) = self.fmt {
            let actual = frame.format();
            if actual != expected {
                eprintln!(
                    "IirProcessor input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                    crate::common::audio::audio::audio_format_diff(expected, actual)
                );
                return Err(CodecError::InvalidData("IirProcessor input AudioFormat mismatch"));
            }
        } else {
            self.fmt = Some(frame.format());
            self.locked = false;
        }

        if self.x_states.is_empty() && self.y_states.is_empty() {
            let fmt = self.fmt.expect("format should be set");
            let (xs, ys, xp, yp) = init_states(fmt, self.b.len(), self.a.len());
            self.x_states = xs;
            self.y_states = ys;
            self.x_pos = xp;
            self.y_pos = yp;
        }
        Ok(())
    }

    #[cfg(feature = "rustfft")]
    fn should_use_fft(&self, nb_samples: usize) -> bool {
        let taps_len = self.b.len().max(self.a.len());
        taps_len >= IIR_FFT_MIN_TAPS || taps_len > nb_samples
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
        let impulse = compute_impulse_response(&self.b, &self.a, IIR_FFT_IMPULSE_LEN);
        let fft_len = select_fft_len(impulse.len(), nb_samples);
        self.fft_state = Some(IirFftState::new(&impulse, channels, fft_len));
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
                        &self.b,
                        &self.a,
                        &mut self.x_states[c],
                        &mut self.y_states[c],
                        &mut self.x_pos[c],
                        &mut self.y_pos[c],
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
                    &self.b,
                    &self.a,
                    &mut self.x_states,
                    &mut self.y_states,
                    &mut self.x_pos,
                    &mut self.y_pos,
                )?
            };
            planes.push(out);
        }

        AudioFrame::from_planes(fmt, nb_samples, frame.time_base(), frame.pts(), planes)
            .map_err(|_| CodecError::InvalidData("failed to build AudioFrame from planes"))
    }
}

impl AudioProcessor for IirProcessor {
    fn name(&self) -> &'static str {
        "iir"
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
        for st in &mut self.x_states {
            for v in st.iter_mut() {
                *v = 0.0;
            }
        }
        for st in &mut self.y_states {
            for v in st.iter_mut() {
                *v = 0.0;
            }
        }
        for pos in &mut self.x_pos {
            *pos = 0;
        }
        for pos in &mut self.y_pos {
            *pos = 0;
        }
        #[cfg(feature = "rustfft")]
        if let Some(state) = &mut self.fft_state {
            state.reset();
        }
        if !self.locked {
            self.fmt = None;
            self.x_states.clear();
            self.y_states.clear();
            self.x_pos.clear();
            self.y_pos.clear();
            #[cfg(feature = "rustfft")]
            {
                self.fft_state = None;
            }
        }
        Ok(())
    }
}

fn validate_coeffs(b: Vec<f32>, a: Vec<f32>) -> CodecResult<(Vec<f64>, Vec<f64>)> {
    if b.is_empty() {
        return Err(CodecError::InvalidData("b must not be empty"));
    }
    if a.is_empty() {
        return Err(CodecError::InvalidData("a must not be empty"));
    }
    let a0 = a[0] as f64;
    if !a0.is_finite() || a0 == 0.0 {
        return Err(CodecError::InvalidData("a[0] must be finite and non-zero"));
    }
    let mut b_out = Vec::with_capacity(b.len());
    for &v in &b {
        if !v.is_finite() {
            return Err(CodecError::InvalidData("b must be finite"));
        }
        b_out.push(v as f64 / a0);
    }
    let mut a_out = Vec::with_capacity(a.len());
    for &v in &a {
        if !v.is_finite() {
            return Err(CodecError::InvalidData("a must be finite"));
        }
        a_out.push(v as f64 / a0);
    }
    a_out[0] = 1.0;
    Ok((b_out, a_out))
}

fn init_states(
    fmt: AudioFormat,
    b_len: usize,
    a_len: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>, Vec<usize>) {
    let channels = fmt.channels() as usize;
    let x_delay = b_len.saturating_sub(1);
    let y_delay = a_len.saturating_sub(1);
    (
        vec![vec![0.0; x_delay]; channels],
        vec![vec![0.0; y_delay]; channels],
        vec![0; channels],
        vec![0; channels],
    )
}

#[cfg(feature = "rustfft")]
fn compute_impulse_response(b: &[f64], a: &[f64], len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    for n in 0..len {
        let mut acc = if n < b.len() { b[n] } else { 0.0 };
        let y_len = a.len().saturating_sub(1);
        for k in 0..y_len {
            if n > k {
                acc -= a[k + 1] * out[n - k - 1];
            }
        }
        out[n] = if acc.is_finite() { acc } else { 0.0 };
    }
    out
}

#[cfg(feature = "rustfft")]
fn select_fft_len(taps_len: usize, nb_samples: usize) -> usize {
    let overlap_len = taps_len.saturating_sub(1);
    let target_block = IIR_FFT_TARGET_BLOCK.max(nb_samples.min(IIR_FFT_TARGET_BLOCK * 4));
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

#[inline]
fn dot_ring(taps_tail: &[f64], state: &[f64], pos: usize) -> f64 {
    if state.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut idx = if pos == 0 { state.len() - 1 } else { pos - 1 };
    for k in 0..state.len() {
        acc += taps_tail[k] * state[idx];
        idx = if idx == 0 { state.len() - 1 } else { idx - 1 };
    }
    acc
}

#[cfg(feature = "wide")]
#[inline]
fn dot_ring_simd(taps_tail: &[f64], state: &[f64], pos: usize) -> f64 {
    if state.is_empty() {
        return 0.0;
    }
    let mut idx = if pos == 0 { state.len() - 1 } else { pos - 1 };
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
fn apply_iir_sample(
    x: f64,
    b: &[f64],
    a: &[f64],
    x_state: &mut [f64],
    y_state: &mut [f64],
    x_pos: &mut usize,
    y_pos: &mut usize,
) -> f64 {
    let b_tail = &b[1..];
    let a_tail = &a[1..];
    let mut acc = b[0] * x;
    #[cfg(feature = "wide")]
    {
        acc += dot_ring_simd(b_tail, x_state, *x_pos);
        acc -= dot_ring_simd(a_tail, y_state, *y_pos);
    }
    #[cfg(not(feature = "wide"))]
    {
        acc += dot_ring(b_tail, x_state, *x_pos);
        acc -= dot_ring(a_tail, y_state, *y_pos);
    }

    if !x_state.is_empty() {
        let n = x_state.len();
        x_state[*x_pos] = x;
        *x_pos = if *x_pos + 1 == n { 0 } else { *x_pos + 1 };
    }
    if !y_state.is_empty() {
        let n = y_state.len();
        y_state[*y_pos] = if acc.is_finite() { acc } else { 0.0 };
        *y_pos = if *y_pos + 1 == n { 0 } else { *y_pos + 1 };
    }

    if acc.is_finite() { acc } else { 0.0 }
}

fn process_planar_bytes(
    input: &[u8],
    sample_type: SampleType,
    b: &[f64],
    a: &[f64],
    x_state: &mut [f64],
    y_state: &mut [f64],
    x_pos: &mut usize,
    y_pos: &mut usize,
) -> CodecResult<Vec<u8>> {
    match sample_type {
        SampleType::U8 => {
            let mut out = Vec::with_capacity(input.len());
            for &v in input {
                let x = (v as f64) - 128.0;
                let y = apply_iir_sample(x, b, a, x_state, y_state, x_pos, y_pos);
                out.push(clamp_u8_pcm(y + 128.0));
            }
            Ok(out)
        }
        SampleType::I16 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(2) {
                let raw = i16::from_ne_bytes([ch[0], ch[1]]);
                let y = apply_iir_sample(raw as f64, b, a, x_state, y_state, x_pos, y_pos);
                out.extend_from_slice(&clamp_i16(y).to_ne_bytes());
            }
            Ok(out)
        }
        SampleType::I32 => {
            let mut out = Vec::with_capacity(input.len());
            for ch in input.chunks_exact(4) {
                let raw = i32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]);
                let y = apply_iir_sample(raw as f64, b, a, x_state, y_state, x_pos, y_pos);
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
                let y = apply_iir_sample(raw as f64, b, a, x_state, y_state, x_pos, y_pos);
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
                let y = apply_iir_sample(x, b, a, x_state, y_state, x_pos, y_pos) as f32;
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
                let y = apply_iir_sample(x, b, a, x_state, y_state, x_pos, y_pos);
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
    b: &[f64],
    a: &[f64],
    x_states: &mut [Vec<f64>],
    y_states: &mut [Vec<f64>],
    x_pos: &mut [usize],
    y_pos: &mut [usize],
) -> CodecResult<Vec<u8>> {
    if channels == 0 {
        return Err(CodecError::InvalidData("channels=0"));
    }
    if x_states.len() != channels || y_states.len() != channels {
        return Err(CodecError::InvalidState("iir states length mismatch"));
    }
    if x_pos.len() != channels || y_pos.len() != channels {
        return Err(CodecError::InvalidState("iir state_pos length mismatch"));
    }

    match sample_type {
        SampleType::U8 => {
            let mut out = Vec::with_capacity(input.len());
            let mut idx = 0usize;
            while idx < input.len() {
                for c in 0..channels {
                    let v = input[idx];
                    let x = (v as f64) - 128.0;
                    let y = apply_iir_sample(
                        x,
                        b,
                        a,
                        &mut x_states[c],
                        &mut y_states[c],
                        &mut x_pos[c],
                        &mut y_pos[c],
                    );
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
                    let y = apply_iir_sample(
                        raw as f64,
                        b,
                        a,
                        &mut x_states[c],
                        &mut y_states[c],
                        &mut x_pos[c],
                        &mut y_pos[c],
                    );
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
                    let y = apply_iir_sample(
                        raw as f64,
                        b,
                        a,
                        &mut x_states[c],
                        &mut y_states[c],
                        &mut x_pos[c],
                        &mut y_pos[c],
                    );
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
                    let y = apply_iir_sample(
                        raw as f64,
                        b,
                        a,
                        &mut x_states[c],
                        &mut y_states[c],
                        &mut x_pos[c],
                        &mut y_pos[c],
                    );
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
                    let y = apply_iir_sample(
                        x,
                        b,
                        a,
                        &mut x_states[c],
                        &mut y_states[c],
                        &mut x_pos[c],
                        &mut y_pos[c],
                    ) as f32;
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
                    let y = apply_iir_sample(
                        x,
                        b,
                        a,
                        &mut x_states[c],
                        &mut y_states[c],
                        &mut x_pos[c],
                        &mut y_pos[c],
                    );
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
    fft_state: &mut IirFftState,
    channel: usize,
) -> CodecResult<Vec<u8>> {
    let samples = planar_bytes_to_f64(input, sample_type);
    let mut out = vec![0.0; samples.len()];
    fft_state.process_channel(channel, &samples, &mut out);
    Ok(planar_f64_to_bytes(&out, sample_type))
}

#[cfg(feature = "rustfft")]
fn process_interleaved_bytes_fft(
    input: &[u8],
    sample_type: SampleType,
    channels: usize,
    fft_state: &mut IirFftState,
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
