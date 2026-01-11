use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrameView};

/// “WAV/PCM 编码器”：严格意义上 WAV 是容器，这里提供一个把 PCM 音频帧打包成
/// “原始 PCM packet”的 encoder，方便走统一的 send/receive pipeline。
#[derive(Clone, Debug)]
pub struct WavEncoderConfig {
    /// - Some(fmt)：强制要求输入格式匹配
    /// - None：首帧自动推断并锁定
    pub input_format: Option<AudioFormat>,
}

pub struct WavEncoder {
    cfg: WavEncoderConfig,
    queued: Option<CodecPacket>,
    flushed: bool,
    locked: bool,
}

impl WavEncoder {
    pub fn new(cfg: WavEncoderConfig) -> CodecResult<Self> {
        let locked = cfg.input_format.is_some();
        Ok(Self {
            cfg,
            queued: None,
            flushed: false,
            locked,
        })
    }
}

impl AudioEncoder for WavEncoder {
    fn name(&self) -> &'static str {
        "pcm(wav)"
    }

    fn input_format(&self) -> Option<AudioFormat> {
        self.cfg.input_format
    }

    fn preferred_frame_samples(&self) -> Option<usize> {
        None
    }

    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        if self.queued.is_some() {
            return Err(CodecError::Again);
        }
        if frame.is_none() {
            self.flushed = true;
            return Ok(());
        }
        let frame = frame.unwrap();
        let fmt = frame.format();
        if frame.nb_samples() == 0 {
            // 空帧不锁定格式，也不产生输出
            return Ok(());
        }
        if self.cfg.input_format.is_none() {
            self.cfg.input_format = Some(fmt);
        }
        let expected = self
            .cfg
            .input_format
            .ok_or(CodecError::InvalidState("WavEncoder missing input_format"))?;
        if fmt != expected {
            eprintln!(
                "WavEncoder input AudioFormat mismatch:\n  input_output_format_diffs: {}",
                crate::common::audio::audio::audio_format_diff(expected, fmt)
            );
            return Err(CodecError::InvalidData("input AudioFormat mismatch"));
        }

        // 输出 packet 统一为 interleaved 单 plane 的 PCM bytes
        let bps = fmt.sample_format.bytes_per_sample();
        let ch = fmt.channels() as usize;
        let ns = frame.nb_samples();
        let mut data = Vec::with_capacity(ns * ch * bps);

        if fmt.is_planar() {
            for s in 0..ns {
                for c in 0..ch {
                    let p = frame.plane(c).ok_or(CodecError::InvalidData("missing plane"))?;
                    let off = s * bps;
                    data.extend_from_slice(&p[off..off + bps]);
                }
            }
        } else {
            let p0 = frame.plane(0).ok_or(CodecError::InvalidData("missing plane 0"))?;
            data.extend_from_slice(p0);
        }

        self.queued = Some(CodecPacket {
            data,
            time_base: frame.time_base(),
            pts: frame.pts(),
            dts: None,
            duration: Some(ns as i64),
            flags: crate::codec::packet::PacketFlags::empty(),
        });
        Ok(())
    }

    fn receive_packet(&mut self) -> CodecResult<CodecPacket> {
        if let Some(pkt) = self.queued.take() {
            return Ok(pkt);
        }
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        self.queued = None;
        self.flushed = false;
        if !self.locked {
            self.cfg.input_format = None;
        }
        Ok(())
    }
}

