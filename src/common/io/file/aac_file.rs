use crate::codec::error::CodecError;
use crate::codec::packet::CodecPacket;
use crate::common::audio::fifo::AudioFifo;
use crate::common::audio::audio::{AudioFrame, AudioFrameView, Rational};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use crate::codec::encoder::encoder_interface::AudioEncoder;

use crate::common::io::io::{AudioReader, AudioWriter, AudioIOResult, AudioIOError};

/// ADTS 头解析出来的最小信息。
#[derive(Clone, Copy, Debug)]
struct AdtsInfo {
    profile: u8, // audioObjectType - 1 (2 bits)
    sr_idx: u8,  // samplingFrequencyIndex (4 bits)
    ch_cfg: u8,  // channelConfiguration (4 bits)
    header_len: usize,
    frame_len: usize, // header + payload
    num_raw_blocks: u8, // number_of_raw_data_blocks_in_frame (0..=3) => blocks+1
}

fn aac_sampling_rate_from_index(idx: u8) -> Option<u32> {
    // ISO/IEC 14496-3 Table 1.16 – Sampling Frequency Index
    Some(match idx {
        0 => 96_000,
        1 => 88_200,
        2 => 64_000,
        3 => 48_000,
        4 => 44_100,
        5 => 32_000,
        6 => 24_000,
        7 => 22_050,
        8 => 16_000,
        9 => 12_000,
        10 => 11_025,
        11 => 8_000,
        12 => 7_350,
        _ => return None,
    })
}

fn parse_adts_header(buf: &[u8]) -> Option<AdtsInfo> {
    if buf.len() < 7 {
        return None;
    }
    // syncword 12bits: 0xFFF
    if buf[0] != 0xFF || (buf[1] & 0xF0) != 0xF0 {
        return None;
    }
    let protection_absent = (buf[1] & 0x01) != 0;
    let header_len = if protection_absent { 7 } else { 9 };

    let profile = (buf[2] >> 6) & 0x03;
    let sr_idx = (buf[2] >> 2) & 0x0F;
    let ch_cfg = ((buf[2] & 0x01) << 2) | ((buf[3] >> 6) & 0x03);

    let frame_len = (((buf[3] & 0x03) as usize) << 11) | ((buf[4] as usize) << 3) | (((buf[5] >> 5) as usize) & 0x07);
    if frame_len < header_len {
        return None;
    }
    let num_raw_blocks = buf[6] & 0x03;
    Some(AdtsInfo {
        profile,
        sr_idx,
        ch_cfg,
        header_len,
        frame_len,
        num_raw_blocks,
    })
}

fn build_asc_from_adts(info: AdtsInfo) -> [u8; 2] {
    // AudioSpecificConfig (minimal 2 bytes):
    // audioObjectType = profile + 1
    // samplingFrequencyIndex = sr_idx
    // channelConfiguration = ch_cfg
    let aot = info.profile + 1;
    let b0 = (aot << 3) | (info.sr_idx >> 1);
    let b1 = ((info.sr_idx & 0x01) << 7) | ((info.ch_cfg & 0x0F) << 3);
    [b0, b1]
}

fn build_adts_header_from_asc(asc: &[u8], aac_payload_len: usize) -> Option<[u8; 7]> {
    if asc.len() < 2 {
        return None;
    }
    let b0 = asc[0];
    let b1 = asc[1];
    let audio_object_type = (b0 >> 3) & 0x1F;
    let sr_idx = ((b0 & 0x07) << 1) | ((b1 >> 7) & 0x01);
    let ch_cfg = (b1 >> 3) & 0x0F;
    if audio_object_type == 0 || sr_idx == 0x0F {
        return None;
    }

    let profile = audio_object_type.saturating_sub(1) & 0x03;
    let frame_len = 7 + aac_payload_len;
    let mut h = [0u8; 7];
    h[0] = 0xFF;
    h[1] = 0xF1;
    h[2] = (profile << 6) | ((sr_idx & 0x0F) << 2) | ((ch_cfg >> 2) & 0x01);
    h[3] = ((ch_cfg & 0x03) << 6) | (((frame_len >> 11) as u8) & 0x03);
    h[4] = ((frame_len >> 3) as u8) & 0xFF;
    h[5] = (((frame_len & 0x07) as u8) << 5) | 0x1F;
    h[6] = 0xFC;
    Some(h)
}

pub struct AacAdtsWriter {
    file: File,
    encoder: crate::codec::encoder::aac_encoder::AacEncoder,
    asc: Vec<u8>,
    fifo: AudioFifo,
    frame_samples: usize,
}

impl AacAdtsWriter {
    pub fn create<P: AsRef<Path>>(
        path: P,
        cfg: crate::codec::encoder::aac_encoder::AacEncoderConfig,
    ) -> AudioIOResult<Self> {
        let file = File::create(path)?;
        let encoder = crate::codec::encoder::aac_encoder::AacEncoder::new(cfg)?;

        // 真实 FFmpeg backend 下这里应当能拿到 ASC；占位实现下会是 None。
        let asc = encoder
            .audio_specific_config()
            .ok_or(AudioIOError::Codec(CodecError::Unsupported(
                "AAC encoder did not provide ASC/extradata (need ffmpeg backend)",
            )))?;

        let sample_rate = encoder
            .input_format()
            .map(|f| f.sample_rate)
            .unwrap_or(48_000);
        let time_base = Rational::new(1, sample_rate as i32);
        let input_format = encoder
            .input_format()
            .ok_or(AudioIOError::Format("AAC encoder missing input_format"))?;
        let fifo = AudioFifo::new(input_format, time_base)
            .map_err(|_| AudioIOError::Format("failed to create AudioFifo"))?;
        let frame_samples = encoder.preferred_frame_samples().unwrap_or(1024);

        Ok(Self {
            file,
            encoder,
            asc,
            fifo,
            frame_samples,
        })
    }

    pub fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;
        self.fifo
            .push_frame(frame)
            .map_err(|_| AudioIOError::Format("AudioFifo push failed (format mismatch?)"))?;

        while let Some(chunk) = self
            .fifo
            .pop_frame(self.frame_samples)
            .map_err(|_| AudioIOError::Format("AudioFifo pop failed"))?
        {
            self.encoder.send_frame(Some(&chunk))?;
            self.drain_packets()?;
        }
        Ok(())
    }

    pub fn finalize(&mut self) -> AudioIOResult<()> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;

        // FIFO 剩余样本作为“最后一帧”（允许 < frame_samples）送入
        let remain = self.fifo.available_samples();
        if remain > 0 {
            if let Some(last) = self
                .fifo
                .pop_frame(remain)
                .map_err(|_| AudioIOError::Format("AudioFifo pop last failed"))?
            {
                self.encoder.send_frame(Some(&last))?;
                self.drain_packets()?;
            }
        }

        self.encoder.send_frame(None)?;
        loop {
            match self.encoder.receive_packet() {
                Ok(pkt) => self.write_one_packet(&pkt)?,
                Err(CodecError::Again) => continue,
                Err(CodecError::Eof) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    fn drain_packets(&mut self) -> AudioIOResult<()> {
        use crate::codec::encoder::encoder_interface::AudioEncoder;
        loop {
            match self.encoder.receive_packet() {
                Ok(pkt) => self.write_one_packet(&pkt)?,
                Err(CodecError::Again) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    fn write_one_packet(&mut self, pkt: &CodecPacket) -> AudioIOResult<()> {
        let hdr = build_adts_header_from_asc(&self.asc, pkt.data.len()).ok_or(AudioIOError::Format("invalid ASC"))?;
        self.file.write_all(&hdr)?;
        self.file.write_all(&pkt.data)?;
        Ok(())
    }
}

impl AudioWriter for AacAdtsWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        self.write_frame(frame)
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        self.finalize()
    }
}

pub struct AacAdtsReader {
    data: Vec<u8>,
    pos: usize,
    decoder: crate::codec::decoder::aac_decoder::AacDecoder,
    time_base: Rational,
    next_pts_samples: i64,
    flushed: bool,
}

impl AacAdtsReader {
    pub fn open<P: AsRef<Path>>(path: P) -> AudioIOResult<Self> {
        let mut f = File::open(path)?;
        let mut data = Vec::new();
        f.read_to_end(&mut data)?;

        // 先用首个 ADTS 头生成 ASC，初始化解码器。
        let info = find_next_adts(&data, 0)
            .and_then(|(i, _)| parse_adts_header(&data[i..]))
            .ok_or(AudioIOError::Format("no ADTS header found"))?;
        let sr = aac_sampling_rate_from_index(info.sr_idx).ok_or(AudioIOError::Format("invalid samplingFrequencyIndex"))?;
        let asc = build_asc_from_adts(info);

        let decoder = crate::codec::decoder::aac_decoder::AacDecoder::new_with_asc(&asc)?;
        Ok(Self {
            data,
            pos: 0,
            decoder,
            time_base: Rational::new(1, sr as i32),
            next_pts_samples: 0,
            flushed: false,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.time_base.den as u32
    }

    pub fn time_base(&self) -> Rational {
        self.time_base
    }

    pub fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
        use crate::codec::decoder::decoder_interface::AudioDecoder;

        loop {
            // 先尝试从 decoder 取输出
            match self.decoder.receive_frame() {
                Ok(f) => return Ok(Some(f)),
                Err(CodecError::Again) => { /* need more input */ }
                Err(CodecError::Eof) => return Ok(None),
                Err(e) => return Err(e.into()),
            }

            // 没输出就继续送入下一个 ADTS frame；如果没了就 flush
            if let Some(pkt) = self.read_next_packet()? {
                self.decoder.send_packet(Some(pkt))?;
            } else if !self.flushed {
                self.decoder.send_packet(None)?;
                self.flushed = true;
            } else {
                // 已 flush 但还拿不到输出：继续 receive_frame() 的循环会转成 Eof
            }
        }
    }

    fn read_next_packet(&mut self) -> AudioIOResult<Option<CodecPacket>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }

        let (start, info) = match find_next_adts(&self.data, self.pos) {
            Some(v) => v,
            None => {
                self.pos = self.data.len();
                return Ok(None);
            }
        };

        let end = start + info.frame_len;
        if end > self.data.len() {
            self.pos = self.data.len();
            return Ok(None);
        }

        let payload = &self.data[start + info.header_len..end];
        self.pos = end;

        // pts 以 samples 为单位递增：ADTS 允许一个 frame 包含多个 raw data block
        // number_of_raw_data_blocks_in_frame = n => blocks = n+1
        let blocks = (info.num_raw_blocks as i64) + 1;
        let samples = 1024i64 * blocks;
        let pkt = CodecPacket {
            data: payload.to_vec(),
            time_base: self.time_base,
            pts: Some(self.next_pts_samples),
            dts: None,
            duration: Some(samples),
            flags: crate::codec::packet::PacketFlags::empty(),
        };
        self.next_pts_samples += samples;
        Ok(Some(pkt))
    }
}

impl AudioReader for AacAdtsReader {
    fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
        self.next_frame()
    }
}

fn find_next_adts(data: &[u8], mut pos: usize) -> Option<(usize, AdtsInfo)> {
    while pos + 7 <= data.len() {
        if data[pos] == 0xFF && (data[pos + 1] & 0xF0) == 0xF0 {
            if let Some(info) = parse_adts_header(&data[pos..]) {
                return Some((pos, info));
            }
        }
        pos += 1;
    }
    None
}

