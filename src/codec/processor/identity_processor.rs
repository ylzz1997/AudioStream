//! IdentityProcessor：PCM->PCM “原样透传” processor

use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView};
use std::collections::VecDeque;

pub struct IdentityProcessor {
    expected_fmt: Option<AudioFormat>,
    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

impl IdentityProcessor {
    /// 接受任意输入格式（不做格式约束）。
    pub fn new() -> Self {
        Self {
            expected_fmt: None,
            out_q: VecDeque::new(),
            flushed: false,
        }
    }

    /// 约束输入格式必须等于 `expected_fmt`（否则报错）。
    pub fn new_with_format(expected_fmt: AudioFormat) -> Self {
        Self {
            expected_fmt: Some(expected_fmt),
            out_q: VecDeque::new(),
            flushed: false,
        }
    }

    fn copy_frame(&self, frame: &dyn AudioFrameView) -> CodecResult<AudioFrame> {
        let fmt = frame.format();
        let nb_samples = frame.nb_samples();
        let plane_count = frame.plane_count();

        // 校验 plane_count 与格式期望一致，防御性更强（避免 silent corruption）。
        let expected_plane_count = AudioFrame::expected_plane_count(&fmt);
        if plane_count != expected_plane_count {
            return Err(CodecError::InvalidData("unexpected plane_count for input AudioFormat"));
        }

        // 逐 plane 拷贝
        let expected_bytes = AudioFrame::expected_bytes_per_plane(&fmt, nb_samples);
        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(plane_count);
        for i in 0..plane_count {
            let p = frame
                .plane(i)
                .ok_or(CodecError::InvalidData("missing plane in AudioFrameView"))?;
            if p.len() != expected_bytes {
                return Err(CodecError::InvalidData("unexpected plane byte size"));
            }
            planes.push(p.to_vec());
        }

        AudioFrame::from_planes(fmt, nb_samples, frame.time_base(), frame.pts(), planes)
            .map_err(|_| CodecError::InvalidData("failed to build AudioFrame from planes"))
    }
}

impl Default for IdentityProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioProcessor for IdentityProcessor {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn input_format(&self) -> Option<AudioFormat> {
        self.expected_fmt
    }

    fn output_format(&self) -> Option<AudioFormat> {
        self.expected_fmt
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
            // flush
            self.flushed = true;
            return Ok(());
        };

        if let Some(expected) = self.expected_fmt {
            if frame.format() != expected {
                return Err(CodecError::InvalidData("IdentityProcessor input AudioFormat mismatch"));
            }
        }

        if frame.nb_samples() == 0 {
            // 空帧直接忽略（不产生输出）。
            return Ok(());
        }

        let out = self.copy_frame(frame)?;
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
        Ok(())
    }
}


