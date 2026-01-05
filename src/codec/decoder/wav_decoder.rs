use crate::codec::decoder::decoder_interface::AudioDecoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrame};

/// “WAV/PCM 解码器”：把“原始 PCM packet”（interleaved bytes）还原成 AudioFrame。
pub struct WavDecoder {
    output_format: AudioFormat,
    queued: Option<AudioFrame>,
    flushed: bool,
    last_time_base: crate::common::audio::audio::Rational,
}

impl WavDecoder {
    pub fn new(output_format: AudioFormat) -> CodecResult<Self> {
        Ok(Self {
            output_format,
            queued: None,
            flushed: false,
            last_time_base: crate::common::audio::audio::Rational::new(1, output_format.sample_rate as i32),
        })
    }
}

impl AudioDecoder for WavDecoder {
    fn name(&self) -> &'static str {
        "pcm(wav)"
    }

    fn output_format(&self) -> Option<AudioFormat> {
        Some(self.output_format)
    }

    fn send_packet(&mut self, packet: Option<CodecPacket>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        if self.queued.is_some() {
            return Err(CodecError::Again);
        }
        if packet.is_none() {
            self.flushed = true;
            return Ok(());
        }
        let pkt = packet.unwrap();
        self.last_time_base = pkt.time_base;
        let bps = self.output_format.sample_format.bytes_per_sample();
        let ch = self.output_format.channels() as usize;
        if ch == 0 {
            return Err(CodecError::InvalidData("channels=0"));
        }
        if pkt.data.len() % (ch * bps) != 0 {
            return Err(CodecError::InvalidData("packet size not aligned to channels*bps"));
        }
        let nb_samples = pkt.data.len() / (ch * bps);

        let planes = vec![pkt.data];
        let frame = AudioFrame::from_planes(
            self.output_format,
            nb_samples,
            self.last_time_base,
            pkt.pts,
            planes,
        )
        .map_err(|_| CodecError::InvalidData("failed to build AudioFrame"))?;

        self.queued = Some(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> CodecResult<AudioFrame> {
        if let Some(f) = self.queued.take() {
            return Ok(f);
        }
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        self.queued = None;
        self.flushed = false;
        Ok(())
    }
}

