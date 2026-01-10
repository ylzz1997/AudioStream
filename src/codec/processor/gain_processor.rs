//! GainProcessor：PCM->PCM 线性增益（volume）processor
//!
//! - 不改变 AudioFormat（采样率/声道/planar/interleaved/采样类型都保持不变）
//! - 对整数类型做饱和裁剪
//! - 支持 planar / interleaved（按 plane 逐块处理）

use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView};
use crate::function::gain::apply_gain_bytes_inplace;
use std::collections::VecDeque;

pub struct GainProcessor {
    gain: f64,
    // 如果为 None：吃到首帧后再锁定格式（输出格式也随之可得）
    fmt: Option<AudioFormat>,
    // 是否由构造参数固定（true => reset 不清空；false => reset 清空推断）
    locked: bool,
    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

impl GainProcessor {
    /// 创建一个增益 processor（gain 为线性倍率，例如 0.5 / 2.0）。
    pub fn new(gain: f64) -> CodecResult<Self> {
        if !gain.is_finite() {
            return Err(CodecError::InvalidData("gain must be finite"));
        }
        Ok(Self {
            gain,
            fmt: None,
            locked: false,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    /// 创建并固定输入/输出格式（后续如果输入格式不匹配会报错）。
    pub fn new_with_format(fmt: AudioFormat, gain: f64) -> CodecResult<Self> {
        if !gain.is_finite() {
            return Err(CodecError::InvalidData("gain must be finite"));
        }
        Ok(Self {
            gain,
            fmt: Some(fmt),
            locked: true,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    pub fn gain(&self) -> f64 {
        self.gain
    }

    pub fn set_gain(&mut self, gain: f64) -> CodecResult<()> {
        if !gain.is_finite() {
            return Err(CodecError::InvalidData("gain must be finite"));
        }
        self.gain = gain;
        Ok(())
    }

    fn process_frame(&self, frame: &dyn AudioFrameView) -> CodecResult<AudioFrame> {
        let fmt = frame.format();
        let nb_samples = frame.nb_samples();

        // 校验 plane_count 与格式期望一致（避免 silent corruption）。
        let expected_plane_count = AudioFrame::expected_plane_count(&fmt);
        if frame.plane_count() != expected_plane_count {
            return Err(CodecError::InvalidData("unexpected plane_count for input AudioFormat"));
        }

        let expected_bytes = AudioFrame::expected_bytes_per_plane(&fmt, nb_samples);
        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(expected_plane_count);
        for i in 0..expected_plane_count {
            let p = frame
                .plane(i)
                .ok_or(CodecError::InvalidData("missing plane in AudioFrameView"))?;
            if p.len() != expected_bytes {
                return Err(CodecError::InvalidData("unexpected plane byte size"));
            }
            let mut out = p.to_vec();
            apply_gain_bytes_inplace(&mut out, fmt.sample_format.sample_type(), self.gain).map_err(|e| {
                CodecError::Unsupported(match e {
                    crate::function::gain::GainError::InvalidBuffer(_) => "invalid pcm buffer for gain",
                    crate::function::gain::GainError::UnsupportedFormat(_) => "unsupported sample type for gain",
                })
            })?;
            planes.push(out);
        }

        AudioFrame::from_planes(fmt, nb_samples, frame.time_base(), frame.pts(), planes)
            .map_err(|_| CodecError::InvalidData("failed to build AudioFrame from planes"))
    }
}

impl AudioProcessor for GainProcessor {
    fn name(&self) -> &'static str {
        "gain"
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
            // 空帧直接忽略（不产生输出）。
            return Ok(());
        }

        if let Some(expected) = self.fmt {
            let actual_fmt = frame.format();
            if actual_fmt != expected {
                eprintln!(
                    "GainProcessor input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                    crate::common::audio::audio::audio_format_diff(expected, actual_fmt)
                );
                return Err(CodecError::InvalidData("GainProcessor input AudioFormat mismatch"));
            }
        } else {
            // 首帧锁定格式
            self.fmt = Some(frame.format());
        }

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
        // 仅清空“推断出来”的格式；显式指定的 fmt 保持
        if !self.locked {
            self.fmt = None;
        }
        Ok(())
    }
}


