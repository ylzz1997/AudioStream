//! CompressorProcessor：PCM->PCM 压缩器processor

use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView};
use crate::function::compressor::{DynamicsCompressor, DynamicsError, DynamicsParams};
use std::collections::VecDeque;

pub struct CompressorProcessor {
    dyns: DynamicsCompressor,
    fmt: AudioFormat,
    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

impl CompressorProcessor {
    /// 创建并固定输入/输出格式（后续如果输入格式不匹配会报错）。
    pub fn new_with_format(fmt: AudioFormat, params: DynamicsParams) -> CodecResult<Self> {
        if (params.sample_rate - (fmt.sample_rate as f32)).abs() > 1e-3 {
            return Err(CodecError::InvalidData(
                "DynamicsParams.sample_rate must match AudioFormat.sample_rate",
            ));
        }
        let channels = fmt.channels() as usize;
        let dyns = DynamicsCompressor::new(params, channels).map_err(map_dyn_err)?;
        Ok(Self {
            dyns,
            fmt,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    pub fn params(&self) -> DynamicsParams {
        self.dyns.params()
    }

    pub fn set_params(&mut self, params: DynamicsParams) -> CodecResult<()> {
        if (params.sample_rate - (self.fmt.sample_rate as f32)).abs() > 1e-3 {
            return Err(CodecError::InvalidData(
                "DynamicsParams.sample_rate must match AudioFormat.sample_rate",
            ));
        }
        self.dyns.set_params(params).map_err(map_dyn_err)
    }

    fn process_frame(&mut self, frame: &dyn AudioFrameView) -> CodecResult<AudioFrame> {
        let fmt = frame.format();
        let nb_samples = frame.nb_samples();

        let expected_plane_count = AudioFrame::expected_plane_count(&fmt);
        if frame.plane_count() != expected_plane_count {
            return Err(CodecError::InvalidData("unexpected plane_count for input AudioFormat"));
        }

        let expected_bytes = AudioFrame::expected_bytes_per_plane(&fmt, nb_samples);
        let channels = fmt.channels() as usize;
        if channels == 0 {
            return Err(CodecError::InvalidData("channels=0"));
        }

        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(expected_plane_count);
        for i in 0..expected_plane_count {
            let p = frame
                .plane(i)
                .ok_or(CodecError::InvalidData("missing plane in AudioFrameView"))?;
            if p.len() != expected_bytes {
                return Err(CodecError::InvalidData("unexpected plane byte size"));
            }
            planes.push(p.to_vec());
        }

        // 就地处理
        if fmt.is_planar() {
            // plane_count == channels
            for c in 0..channels {
                self.dyns
                    .process_planar_channel_bytes_inplace(
                        &mut planes[c],
                        fmt.sample_format.sample_type(),
                        c,
                    )
                    .map_err(map_dyn_err)?;
            }
        } else {
            // plane_count == 1
            self.dyns
                .process_interleaved_bytes_inplace(&mut planes[0], fmt.sample_format.sample_type(), channels)
                .map_err(map_dyn_err)?;
        }

        AudioFrame::from_planes(fmt, nb_samples, frame.time_base(), frame.pts(), planes)
            .map_err(|_| CodecError::InvalidData("failed to build AudioFrame from planes"))
    }
}

impl AudioProcessor for CompressorProcessor {
    fn name(&self) -> &'static str {
        "compressor"
    }

    fn input_format(&self) -> Option<AudioFormat> {
        Some(self.fmt)
    }

    fn output_format(&self) -> Option<AudioFormat> {
        Some(self.fmt)
    }

    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
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

        let actual_fmt = frame.format();
        if actual_fmt != self.fmt {
            eprintln!(
                "CompressorProcessor input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                crate::common::audio::audio::audio_format_diff(self.fmt, actual_fmt)
            );
            return Err(CodecError::InvalidData(
                "CompressorProcessor input AudioFormat mismatch",
            ));
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
        self.dyns.reset();
        Ok(())
    }
}

fn map_dyn_err(e: DynamicsError) -> CodecError {
    match e {
        DynamicsError::InvalidParameter(msg) => CodecError::InvalidData(msg),
        DynamicsError::InvalidBuffer(msg) => CodecError::InvalidData(msg),
        DynamicsError::UnsupportedFormat(msg) => CodecError::Unsupported(msg),
    }
}


