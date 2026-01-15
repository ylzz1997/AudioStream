//! DelayProcessor：PCM->PCM 延迟处理（在开头插入静音）。
//!
//! - 不改变 AudioFormat
//! - delay_ms 以毫秒为单位，按 sample_rate 换算到 samples
//! - 使用 FIFO 保持输入帧边界（输出帧大小跟随输入帧）

use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView, AudioFrameViewMut};
use crate::common::audio::fifo::AudioFifo;
use std::collections::VecDeque;

pub struct DelayProcessor {
    delay_ms: f64,
    // 如果为 None：吃到首帧后再锁定格式
    fmt: Option<AudioFormat>,
    // 是否由构造参数固定（true => reset 不清空；false => reset 清空推断）
    locked: bool,
    // 由 delay_ms + sample_rate 计算得到
    delay_samples: Option<usize>,
    delay_inserted: bool,
    fifo: Option<AudioFifo>,
    out_sizes: VecDeque<usize>,
    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

impl DelayProcessor {
    /// 创建 delay processor（delay_ms 单位毫秒）。
    pub fn new(delay_ms: f64) -> CodecResult<Self> {
        validate_delay_ms(delay_ms)?;
        Ok(Self {
            delay_ms,
            fmt: None,
            locked: false,
            delay_samples: None,
            delay_inserted: false,
            fifo: None,
            out_sizes: VecDeque::new(),
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    /// 创建并固定输入/输出格式（后续如果输入格式不匹配会报错）。
    pub fn new_with_format(fmt: AudioFormat, delay_ms: f64) -> CodecResult<Self> {
        validate_delay_ms(delay_ms)?;
        let delay_samples = compute_delay_samples(delay_ms, fmt.sample_rate)?;
        Ok(Self {
            delay_ms,
            fmt: Some(fmt),
            locked: true,
            delay_samples: Some(delay_samples),
            delay_inserted: false,
            fifo: None,
            out_sizes: VecDeque::new(),
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    pub fn delay_ms(&self) -> f64 {
        self.delay_ms
    }

    pub fn delay_samples_cached(&self) -> Option<usize> {
        self.delay_samples
    }

    fn ensure_format(&mut self, frame: &dyn AudioFrameView) -> CodecResult<()> {
        if let Some(expected) = self.fmt {
            let actual = frame.format();
            if actual != expected {
                eprintln!(
                    "DelayProcessor input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                    crate::common::audio::audio::audio_format_diff(expected, actual)
                );
                return Err(CodecError::InvalidData("DelayProcessor input AudioFormat mismatch"));
            }
        } else {
            self.fmt = Some(frame.format());
            self.locked = false;
        }

        if self.delay_samples.is_none() {
            let fmt = self.fmt.expect("format should be set");
            self.delay_samples = Some(compute_delay_samples(self.delay_ms, fmt.sample_rate)?);
        }

        if self.fifo.is_none() {
            let fmt = self.fmt.expect("format should be set");
            let tb = frame.time_base();
            let fifo = AudioFifo::new(fmt, tb)
                .map_err(|e| CodecError::Other(format!("AudioFifo init failed: {e}")))?;
            self.fifo = Some(fifo);
        }
        Ok(())
    }

    fn insert_initial_silence(&mut self, frame: &dyn AudioFrameView) -> CodecResult<()> {
        if self.delay_inserted {
            return Ok(());
        }
        let delay_samples = self.delay_samples.unwrap_or(0);
        self.delay_inserted = true;
        if delay_samples == 0 {
            return Ok(());
        }

        let fmt = frame.format();
        let mut silence = AudioFrame::new_alloc(fmt, delay_samples)
            .map_err(|_| CodecError::InvalidData("failed to alloc silence frame"))?;
        silence
            .set_time_base(frame.time_base())
            .map_err(|_| CodecError::InvalidData("invalid time_base for silence frame"))?;
        let pts = frame.pts().map(|p| p - delay_samples as i64);
        silence.set_pts(pts);

        let fifo = self.fifo.as_mut().ok_or(CodecError::InvalidState("fifo not initialized"))?;
        fifo.push_frame(&silence)
            .map_err(|e| CodecError::Other(format!("AudioFifo push failed: {e}")))?;
        Ok(())
    }

    fn try_pop_output(&mut self) -> CodecResult<()> {
        let Some(fifo) = self.fifo.as_mut() else {
            return Ok(());
        };
        while let Some(&n) = self.out_sizes.front() {
            if fifo.available_samples() < n {
                break;
            }
            let f = fifo
                .pop_frame(n)
                .map_err(|e| CodecError::Other(format!("AudioFifo pop failed: {e}")))?;
            if let Some(frame) = f {
                self.out_q.push_back(frame);
                self.out_sizes.pop_front();
            } else {
                break;
            }
        }
        Ok(())
    }

    fn flush_remaining(&mut self) -> CodecResult<()> {
        let Some(fifo) = self.fifo.as_mut() else {
            return Ok(());
        };
        if !self.out_sizes.is_empty() {
            self.try_pop_output()?;
        }
        if !self.out_sizes.is_empty() {
            // 仍有未满足的输出请求，留给后续 receive_frame 推进
            return Ok(());
        }
        let remaining = fifo.available_samples();
        if remaining == 0 {
            return Ok(());
        }
        let f = fifo
            .pop_frame(remaining)
            .map_err(|e| CodecError::Other(format!("AudioFifo pop failed: {e}")))?;
        if let Some(frame) = f {
            self.out_q.push_back(frame);
        }
        Ok(())
    }
}

impl AudioProcessor for DelayProcessor {
    fn name(&self) -> &'static str {
        "delay"
    }

    fn input_format(&self) -> Option<AudioFormat> {
        self.fmt
    }

    fn output_format(&self) -> Option<AudioFormat> {
        self.fmt
    }

    fn delay_samples(&self) -> usize {
        self.delay_samples.unwrap_or(0)
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
            self.flush_remaining()?;
            return Ok(());
        };

        if frame.nb_samples() == 0 {
            return Ok(());
        }

        self.ensure_format(frame)?;
        self.insert_initial_silence(frame)?;

        let fifo = self.fifo.as_mut().ok_or(CodecError::InvalidState("fifo not initialized"))?;
        fifo.push_frame(frame)
            .map_err(|e| CodecError::Other(format!("AudioFifo push failed: {e}")))?;
        self.out_sizes.push_back(frame.nb_samples());

        self.try_pop_output()?;
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
        self.out_sizes.clear();
        self.fifo = None;
        self.flushed = false;
        self.delay_inserted = false;
        if !self.locked {
            self.fmt = None;
            self.delay_samples = None;
        }
        Ok(())
    }
}

fn validate_delay_ms(delay_ms: f64) -> CodecResult<()> {
    if !delay_ms.is_finite() || delay_ms < 0.0 {
        return Err(CodecError::InvalidData("delay_ms must be finite and >= 0"));
    }
    Ok(())
}

fn compute_delay_samples(delay_ms: f64, sample_rate: u32) -> CodecResult<usize> {
    if sample_rate == 0 {
        return Err(CodecError::InvalidData("sample_rate must be > 0"));
    }
    let samples_f = delay_ms * sample_rate as f64 / 1000.0;
    if !samples_f.is_finite() || samples_f < 0.0 {
        return Err(CodecError::InvalidData("invalid delay_ms"));
    }
    if samples_f > (usize::MAX as f64) {
        return Err(CodecError::InvalidData("delay_ms too large"));
    }
    Ok(samples_f.round() as usize)
}
