// File I/O Interface

use crate::codec::error::CodecError;
use crate::common::audio::audio::{AudioFrame, AudioFrameView};
use core::fmt;
use std::path::Path;

use super::aac_file::{AacAdtsReader, AacAdtsWriter};
use super::mp3_file::{Mp3Reader, Mp3Writer};
use super::opus_file::{OpusOggReader, OpusOggWriter};
use super::wav_file::{WavReader, WavWriter};
use crate::common::io::io::{AudioReader, AudioWriter};

#[derive(Debug)]
pub enum AudioFileError {
    Io(std::io::Error),
    Codec(CodecError),
    Format(&'static str),
}

impl fmt::Display for AudioFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioFileError::Io(e) => write!(f, "io error: {e}"),
            AudioFileError::Codec(e) => write!(f, "codec error: {e}"),
            AudioFileError::Format(msg) => write!(f, "format error: {msg}"),
        }
    }
}

impl std::error::Error for AudioFileError {}

impl From<std::io::Error> for AudioFileError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<CodecError> for AudioFileError {
    fn from(e: CodecError) -> Self {
        Self::Codec(e)
    }
}

pub type AudioFileResult<T> = Result<T, AudioFileError>;

/// 当前支持的“文件封装格式”。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioFileFormat {
    AacAdts,
    Mp3,
    /// 标准 Ogg Opus（播放器可播放的 .opus）
    OpusOgg,
    Wav,
}

/// 当前支持的 codec（可从 `src/codec/` 扩展）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecId {
    Aac,
    Mp3,
    Opus,
    Pcm, // wav 等无压缩 PCM
}

pub enum AudioFileWriteConfig {
    AacAdts(crate::codec::encoder::aac_encoder::AacEncoderConfig),
    Mp3(super::mp3_file::Mp3WriterConfig),
    /// 标准 Ogg Opus（播放器可播放的 .opus）
    OpusOgg(crate::codec::encoder::opus_encoder::OpusEncoderConfig),
    Wav(super::wav_file::WavWriterConfig),
}

pub enum AudioFileReadConfig {
    AacAdts,
    Mp3,
    /// 标准 Ogg Opus（播放器可播放的 .opus）
    OpusOgg,
    Wav,
}

pub enum AudioFileWriter {
    AacAdts(AacAdtsWriter),
    Mp3(Mp3Writer),
    OpusOgg(OpusOggWriter),
    Wav(WavWriter),
}

pub enum AudioFileReader {
    AacAdts(AacAdtsReader),
    Mp3(Mp3Reader),
    OpusOgg(OpusOggReader),
    Wav(WavReader),
}

impl AudioFileWriter {
    pub fn create<P: AsRef<Path>>(path: P, cfg: AudioFileWriteConfig) -> AudioFileResult<Self> {
        match cfg {
            AudioFileWriteConfig::AacAdts(enc_cfg) => Ok(Self::AacAdts(AacAdtsWriter::create(path, enc_cfg)?)),
            AudioFileWriteConfig::Mp3(cfg) => Ok(Self::Mp3(Mp3Writer::create(path, cfg)?)),
            AudioFileWriteConfig::OpusOgg(cfg) => Ok(Self::OpusOgg(OpusOggWriter::create(path, cfg)?)),
            AudioFileWriteConfig::Wav(cfg) => Ok(Self::Wav(WavWriter::create(path, cfg)?)),
        }
    }
}

impl AudioFileReader {
    pub fn open<P: AsRef<Path>>(path: P, cfg: AudioFileReadConfig) -> AudioFileResult<Self> {
        match cfg {
            AudioFileReadConfig::AacAdts => Ok(Self::AacAdts(AacAdtsReader::open(path)?)),
            AudioFileReadConfig::Mp3 => Ok(Self::Mp3(Mp3Reader::open(path)?)),
            AudioFileReadConfig::OpusOgg => Ok(Self::OpusOgg(OpusOggReader::open(path)?)),
            AudioFileReadConfig::Wav => Ok(Self::Wav(WavReader::open(path)?)),
        }
    }
}

impl AudioWriter for AudioFileWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioFileResult<()> {
        match self {
            AudioFileWriter::AacAdts(w) => w.write_frame(frame),
            AudioFileWriter::Mp3(w) => w.write_frame(frame),
            AudioFileWriter::OpusOgg(w) => w.write_frame(frame),
            AudioFileWriter::Wav(w) => w.write_frame(frame),
        }
    }

    fn finalize(&mut self) -> AudioFileResult<()> {
        match self {
            AudioFileWriter::AacAdts(w) => w.finalize(),
            AudioFileWriter::Mp3(w) => w.finalize(),
            AudioFileWriter::OpusOgg(w) => w.finalize(),
            AudioFileWriter::Wav(w) => w.finalize(),
        }
    }
}

impl AudioReader for AudioFileReader {
    fn next_frame(&mut self) -> AudioFileResult<Option<AudioFrame>> {
        match self {
            AudioFileReader::AacAdts(r) => r.next_frame(),
            AudioFileReader::Mp3(r) => r.next_frame(),
            AudioFileReader::OpusOgg(r) => r.next_frame(),
            AudioFileReader::Wav(r) => r.next_frame(),
        }
    }
}
