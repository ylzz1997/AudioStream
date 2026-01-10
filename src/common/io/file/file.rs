// File I/O Interface

use crate::codec::error::CodecError;
use crate::common::audio::audio::{AudioFrame, AudioFrameView};
use core::fmt;
use std::path::Path;

use super::aac_file::{AacAdtsReader, AacAdtsWriter};
use super::flac_file::{FlacReader, FlacWriter};
use super::mp3_file::{Mp3Reader, Mp3Writer};
use super::opus_file::{OpusOggReader, OpusOggWriter};
use super::wav_file::{WavReader, WavWriter};
use crate::common::io::io::{AudioReader, AudioWriter, AudioIOResult, AudioIOError};

impl fmt::Display for AudioIOError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioIOError::Io(e) => write!(f, "io error: {e}"),
            AudioIOError::Codec(e) => write!(f, "codec error: {e}"),
            AudioIOError::Format(msg) => write!(f, "format error: {msg}"),
        }
    }
}

impl std::error::Error for AudioIOError {}

impl From<std::io::Error> for AudioIOError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<CodecError> for AudioIOError {
    fn from(e: CodecError) -> Self {
        Self::Codec(e)
    }
}

/// 当前支持的“文件封装格式”。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioFileFormat {
    AacAdts,
    Flac,
    Mp3,
    /// 标准 Ogg Opus（播放器可播放的 .opus）
    OpusOgg,
    Wav,
}

/// 当前支持的 codec（可从 `src/codec/` 扩展）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecId {
    Aac,
    Flac,
    Mp3,
    Opus,
    Pcm, // wav 等无压缩 PCM
}

pub enum AudioFileWriteConfig {
    AacAdts(super::aac_file::AacAdtsWriterConfig),
    Flac(super::flac_file::FlacWriterConfig),
    Mp3(super::mp3_file::Mp3WriterConfig),
    OpusOgg(super::opus_file::OpusOggWriterConfig),
    Wav(super::wav_file::WavWriterConfig),
}

pub enum AudioFileReadConfig {
    AacAdts,
    Flac,
    Mp3,
    OpusOgg,
    Wav(super::wav_file::WavReaderConfig),
}

pub enum AudioFileWriter {
    AacAdts(AacAdtsWriter),
    Flac(FlacWriter),
    Mp3(Mp3Writer),
    OpusOgg(OpusOggWriter),
    Wav(WavWriter),
}

pub enum AudioFileReader {
    AacAdts(AacAdtsReader),
    Flac(FlacReader),
    Mp3(Mp3Reader),
    OpusOgg(OpusOggReader),
    Wav(WavReader),
}

impl AudioFileWriter {
    pub fn create<P: AsRef<Path>>(path: P, cfg: AudioFileWriteConfig) -> AudioIOResult<Self> {
        match cfg {
            AudioFileWriteConfig::AacAdts(enc_cfg) => Ok(Self::AacAdts(AacAdtsWriter::create(path, enc_cfg)?)),
            AudioFileWriteConfig::Flac(cfg) => Ok(Self::Flac(FlacWriter::create(path, cfg)?)),
            AudioFileWriteConfig::Mp3(cfg) => Ok(Self::Mp3(Mp3Writer::create(path, cfg)?)),
            AudioFileWriteConfig::OpusOgg(cfg) => Ok(Self::OpusOgg(OpusOggWriter::create(path, cfg)?)),
            AudioFileWriteConfig::Wav(cfg) => Ok(Self::Wav(WavWriter::create(path, cfg)?)),
        }
    }
}

impl AudioFileReader {
    pub fn open<P: AsRef<Path>>(path: P, cfg: AudioFileReadConfig) -> AudioIOResult<Self> {
        match cfg {
            AudioFileReadConfig::AacAdts => Ok(Self::AacAdts(AacAdtsReader::open(path)?)),
            AudioFileReadConfig::Flac => Ok(Self::Flac(FlacReader::open(path)?)),
            AudioFileReadConfig::Mp3 => Ok(Self::Mp3(Mp3Reader::open(path)?)),
            AudioFileReadConfig::OpusOgg => Ok(Self::OpusOgg(OpusOggReader::open(path)?)),
            AudioFileReadConfig::Wav(cfg) => Ok(Self::Wav(WavReader::open_with_config(path, cfg)?)),
        }
    }
}

impl AudioWriter for AudioFileWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        match self {
            AudioFileWriter::AacAdts(w) => w.write_frame(frame),
            AudioFileWriter::Flac(w) => w.write_frame(frame),
            AudioFileWriter::Mp3(w) => w.write_frame(frame),
            AudioFileWriter::OpusOgg(w) => w.write_frame(frame),
            AudioFileWriter::Wav(w) => w.write_frame(frame),
        }
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        match self {
            AudioFileWriter::AacAdts(w) => w.finalize(),
            AudioFileWriter::Flac(w) => w.finalize(),
            AudioFileWriter::Mp3(w) => w.finalize(),
            AudioFileWriter::OpusOgg(w) => w.finalize(),
            AudioFileWriter::Wav(w) => w.finalize(),
        }
    }
}

impl AudioReader for AudioFileReader {
    fn next_frame(&mut self) -> AudioIOResult<Option<AudioFrame>> {
        match self {
            AudioFileReader::AacAdts(r) => r.next_frame(),
            AudioFileReader::Flac(r) => r.next_frame(),
            AudioFileReader::Mp3(r) => r.next_frame(),
            AudioFileReader::OpusOgg(r) => r.next_frame(),
            AudioFileReader::Wav(r) => r.next_frame(),
        }
    }
}
