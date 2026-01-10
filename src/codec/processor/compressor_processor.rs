//! CompressorProcessor：PCM->PCM 压缩器processor

use crate::codec::error::{CodecError, CodecResult};
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView};
use crate::function::compressor::{DynamicsCompressor, DynamicsError, DynamicsParams};
use std::collections::VecDeque;

pub struct CompressorProcessor {
    // 如果为 None：吃到首帧后再锁定格式并初始化 dyns
    dyns: Option<DynamicsCompressor>,
    // 如果为 None：吃到首帧后锁定格式（并校验 sample_rate 与 params.sample_rate）
    fmt: Option<AudioFormat>,
    // 初始参数（在 fmt 未锁定前允许修改；锁定后要求 sample_rate 一致）
    params: DynamicsParams,
    // 是否由构造参数固定（true => reset 不清空；false => reset 清空推断）
    locked: bool,
    out_q: VecDeque<AudioFrame>,
    flushed: bool,
}

impl CompressorProcessor {
    /// 不指定输入格式：首帧推断并初始化 compressor。
    pub fn new(params: DynamicsParams) -> CodecResult<Self> {
        Ok(Self {
            dyns: None,
            fmt: None,
            params,
            locked: false,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

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
            dyns: Some(dyns),
            fmt: Some(fmt),
            params,
            locked: true,
            out_q: VecDeque::new(),
            flushed: false,
        })
    }

    pub fn params(&self) -> DynamicsParams {
        self.dyns.as_ref().map(|d| d.params()).unwrap_or(self.params)
    }

    pub fn set_params(&mut self, params: DynamicsParams) -> CodecResult<()> {
        if let Some(fmt) = self.fmt {
            if (params.sample_rate - (fmt.sample_rate as f32)).abs() > 1e-3 {
                return Err(CodecError::InvalidData(
                    "DynamicsParams.sample_rate must match AudioFormat.sample_rate",
                ));
            }
        }
        self.params = params;
        if let Some(d) = self.dyns.as_mut() {
            d.set_params(params).map_err(map_dyn_err)?;
        }
        Ok(())
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

        let Some(dyns) = self.dyns.as_mut() else {
            return Err(CodecError::InvalidState("compressor not initialized"));
        };

        // 就地处理
        if fmt.is_planar() {
            // plane_count == channels
            for c in 0..channels {
                dyns
                    .process_planar_channel_bytes_inplace(
                        &mut planes[c],
                        fmt.sample_format.sample_type(),
                        c,
                    )
                    .map_err(map_dyn_err)?;
            }
        } else {
            // plane_count == 1
            dyns
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
        self.fmt
    }

    fn output_format(&self) -> Option<AudioFormat> {
        self.fmt
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
        if let Some(expected) = self.fmt {
            if actual_fmt != expected {
                eprintln!(
                    "CompressorProcessor input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                    crate::common::audio::audio::audio_format_diff(expected, actual_fmt)
                );
                return Err(CodecError::InvalidData(
                    "CompressorProcessor input AudioFormat mismatch",
                ));
            }
        } else {
            // 首帧锁定格式并初始化 dyns
            if (self.params.sample_rate - (actual_fmt.sample_rate as f32)).abs() > 1e-3 {
                return Err(CodecError::InvalidData(
                    "DynamicsParams.sample_rate must match AudioFormat.sample_rate",
                ));
            }
            let channels = actual_fmt.channels() as usize;
            let dyns = DynamicsCompressor::new(self.params, channels).map_err(map_dyn_err)?;
            self.dyns = Some(dyns);
            self.fmt = Some(actual_fmt);
            self.locked = false;
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
        if let Some(d) = self.dyns.as_mut() {
            d.reset();
        }
        if !self.locked {
            self.dyns = None;
            self.fmt = None;
        }
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


