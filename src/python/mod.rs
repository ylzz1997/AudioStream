//! Python bindings (PyO3) for streaming encoder/decoder.
//!
//! 设计目标（对应需求）：
//! - 暴露 Encoder/Decoder
//! - put_frame / get_frame：内部用 AudioFifo 做 chunk 聚合；最后不满一个 chunk 默认不返回，
//!   但 get_frame(force=True) 可强制 flush 剩余样本。

#![allow(unsafe_op_in_unsafe_fn)]

use crate::codec::decoder::aac_decoder::AacDecoder;
use crate::codec::decoder::decoder_interface::AudioDecoder;
use crate::codec::decoder::flac_decoder::FlacDecoder;
use crate::codec::decoder::mp3_decoder::Mp3Decoder;
use crate::codec::decoder::opus_decoder::OpusDecoder;
use crate::codec::decoder::wav_decoder::WavDecoder;
use crate::codec::encoder::aac_encoder::{AacEncoder, AacEncoderConfig};
use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::encoder::flac_encoder::{FlacEncoder, FlacEncoderConfig};
use crate::codec::encoder::mp3_encoder::{Mp3Encoder, Mp3EncoderConfig};
use crate::codec::encoder::opus_encoder::{OpusEncoder, OpusEncoderConfig};
use crate::codec::encoder::wav_encoder::{WavEncoder, WavEncoderConfig};
use crate::codec::error::CodecError;
use crate::codec::packet::{CodecPacket, PacketFlags};
use crate::codec::processor::identity_processor::IdentityProcessor;
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{
    AudioError, AudioFormat as RsAudioFormat, AudioFrameView, ChannelLayout, Rational, SampleFormat, SampleType,
};
use crate::common::audio::fifo::AudioFifo;

use numpy::{Element, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use std::collections::VecDeque;
use tokio::runtime::Runtime;

use crate::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
use crate::pipeline::node::dynamic_node_interface::{DynNode, IdentityNode, ProcessorNode};
use crate::pipeline::node::node_interface::{AsyncPipeline, NodeBuffer, NodeBufferKind};

use crate::codec::processor::resample_processor::ResampleProcessor;
use crate::common::audio::audio::AudioFrameViewMut;
use crate::runner::audio_sink::AudioSink;
use crate::runner::audio_source::AudioSource;
use crate::runner::async_runner_interface::AsyncRunner as RsAsyncRunner;
use crate::runner::auto_runner::AutoRunner;
use crate::runner::error::{RunnerError, RunnerResult};
use crate::common::io::file as rs_file;
use crate::common::io::file::{AudioFileError, AudioFileReader as RsAudioFileReader, AudioFileReadConfig, AudioFileWriter as RsAudioFileWriter, AudioFileWriteConfig};
use crate::common::io::io::{AudioReader, AudioWriter};

fn map_codec_err(e: CodecError) -> PyErr {
    match e {
        CodecError::InvalidData(msg) => PyValueError::new_err(msg),
        CodecError::InvalidState(msg) => PyRuntimeError::new_err(msg),
        CodecError::Unsupported(msg) => PyRuntimeError::new_err(msg),
        CodecError::Other(msg) => PyRuntimeError::new_err(msg),
        CodecError::Again => PyRuntimeError::new_err("codec again (EAGAIN)"),
        CodecError::Eof => PyRuntimeError::new_err("codec eof (EOF)"),
    }
}

fn map_audio_err(e: AudioError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn sample_format_from_type(t: SampleType, planar: bool) -> SampleFormat {
    match t {
        SampleType::U8 => SampleFormat::U8 { planar },
        SampleType::I16 => SampleFormat::I16 { planar },
        SampleType::I32 => SampleFormat::I32 { planar },
        SampleType::I64 => SampleFormat::I64 { planar },
        SampleType::F32 => SampleFormat::F32 { planar },
        SampleType::F64 => SampleFormat::F64 { planar },
    }
}

fn parse_sample_type(s: &str) -> Option<SampleType> {
    match s.to_ascii_lowercase().as_str() {
        "u8" => Some(SampleType::U8),
        "i16" => Some(SampleType::I16),
        "i32" => Some(SampleType::I32),
        "i64" => Some(SampleType::I64),
        "f32" => Some(SampleType::F32),
        "f64" => Some(SampleType::F64),
        _ => None,
    }
}

fn codec_from_str(s: &str) -> Option<&'static str> {
    match s.to_ascii_lowercase().as_str() {
        "wav" | "pcm" => Some("wav"),
        "mp3" => Some("mp3"),
        "aac" => Some("aac"),
        "opus" => Some("opus"),
        "flac" => Some("flac"),
        _ => None,
    }
}

fn warnings_warn(py: Python<'_>, msg: &str) -> PyResult<()> {
    let warnings = py.import_bound("warnings")?;
    warnings.call_method1("warn", (msg,))?;
    Ok(())
}

/// Python 侧 AudioFormat（用于 ConfigStruct 里描述 PCM 格式）。
///
/// 注意：本 Python API 默认使用 planar PCM（numpy shape = (channels, samples)）。
#[pyclass]
#[derive(Clone)]
pub struct AudioFormat {
    #[pyo3(get)]
    pub sample_rate: u32,
    #[pyo3(get)]
    pub channels: u16,
    #[pyo3(get)]
    pub sample_type: String, // "f32"/"i16"/...
    #[pyo3(get)]
    pub planar: bool,
    #[pyo3(get)]
    pub channel_layout_mask: u64,
}

#[pymethods]
impl AudioFormat {
    #[new]
    #[pyo3(signature = (sample_rate, channels, sample_type, planar=true, channel_layout_mask=0))]
    fn new(sample_rate: u32, channels: u16, sample_type: String, planar: bool, channel_layout_mask: u64) -> PyResult<Self> {
        if sample_rate == 0 {
            return Err(PyValueError::new_err("sample_rate 必须 > 0"));
        }
        if channels == 0 {
            return Err(PyValueError::new_err("channels 必须 > 0"));
        }
        if parse_sample_type(&sample_type).is_none() {
            return Err(PyValueError::new_err("sample_type 仅支持: u8/i16/i32/i64/f32/f64"));
        }
        Ok(Self {
            sample_rate,
            channels,
            sample_type,
            planar,
            channel_layout_mask,
        })
    }
}

impl AudioFormat {
    fn to_rs(&self) -> PyResult<RsAudioFormat> {
        let st = parse_sample_type(&self.sample_type)
            .ok_or_else(|| PyValueError::new_err("invalid sample_type"))?;
        let sf = sample_format_from_type(st, self.planar);
        let ch_layout = if self.channel_layout_mask != 0 {
            ChannelLayout {
                channels: self.channels,
                mask: self.channel_layout_mask,
            }
        } else if self.channels == 1 {
            ChannelLayout::mono()
        } else if self.channels == 2 {
            ChannelLayout::stereo()
        } else {
            ChannelLayout::unspecified(self.channels)
        };
        Ok(RsAudioFormat {
            sample_rate: self.sample_rate,
            sample_format: sf,
            channel_layout: ch_layout,
        })
    }

    fn sample_type_rs(&self) -> PyResult<SampleType> {
        parse_sample_type(&self.sample_type).ok_or_else(|| PyValueError::new_err("invalid sample_type"))
    }
}

#[pyclass(name = "WavEncoderConfig")]
#[derive(Clone)]
pub struct WavEncoderConfigPy {
    #[pyo3(get)]
    pub input_format: AudioFormat,
    #[pyo3(get)]
    pub chunk_samples: usize,
}

#[pymethods]
impl WavEncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format, chunk_samples))]
    fn new(input_format: AudioFormat, chunk_samples: usize) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        Ok(Self {
            input_format,
            chunk_samples,
        })
    }
}

#[pyclass(name = "WavDecoderConfig")]
#[derive(Clone)]
pub struct WavDecoderConfigPy {
    #[pyo3(get)]
    pub output_format: AudioFormat,
    #[pyo3(get)]
    pub chunk_samples: usize,
}

#[pymethods]
impl WavDecoderConfigPy {
    #[new]
    #[pyo3(signature = (output_format, chunk_samples))]
    fn new(output_format: AudioFormat, chunk_samples: usize) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        Ok(Self {
            output_format,
            chunk_samples,
        })
    }
}

#[pyclass(name = "Mp3EncoderConfig")]
#[derive(Clone)]
pub struct Mp3EncoderConfigPy {
    #[pyo3(get)]
    pub input_format: AudioFormat,
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub bitrate: Option<u32>,
}

#[pymethods]
impl Mp3EncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format, chunk_samples, bitrate=Some(128_000)))]
    fn new(input_format: AudioFormat, chunk_samples: usize, bitrate: Option<u32>) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        Ok(Self {
            input_format,
            chunk_samples,
            bitrate,
        })
    }
}

#[pyclass(name = "AacEncoderConfig")]
#[derive(Clone)]
pub struct AacEncoderConfigPy {
    #[pyo3(get)]
    pub input_format: AudioFormat,
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub bitrate: Option<u32>,
}

#[pymethods]
impl AacEncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format, chunk_samples, bitrate=None))]
    fn new(input_format: AudioFormat, chunk_samples: usize, bitrate: Option<u32>) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        Ok(Self {
            input_format,
            chunk_samples,
            bitrate,
        })
    }
}

#[pyclass(name = "Mp3DecoderConfig")]
#[derive(Clone)]
pub struct Mp3DecoderConfigPy {
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub packet_time_base_den: i32,
}

#[pymethods]
impl Mp3DecoderConfigPy {
    #[new]
    #[pyo3(signature = (chunk_samples, packet_time_base_den=48000))]
    fn new(chunk_samples: usize, packet_time_base_den: i32) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        if packet_time_base_den <= 0 {
            return Err(PyValueError::new_err("packet_time_base_den 必须 > 0"));
        }
        Ok(Self {
            chunk_samples,
            packet_time_base_den,
        })
    }
}

#[pyclass(name = "AacDecoderConfig")]
#[derive(Clone)]
pub struct AacDecoderConfigPy {
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub packet_time_base_den: i32,
}

#[pymethods]
impl AacDecoderConfigPy {
    #[new]
    #[pyo3(signature = (chunk_samples, packet_time_base_den=48000))]
    fn new(chunk_samples: usize, packet_time_base_den: i32) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        if packet_time_base_den <= 0 {
            return Err(PyValueError::new_err("packet_time_base_den 必须 > 0"));
        }
        Ok(Self {
            chunk_samples,
            packet_time_base_den,
        })
    }
}

#[pyclass(name = "OpusEncoderConfig")]
#[derive(Clone)]
pub struct OpusEncoderConfigPy {
    #[pyo3(get)]
    pub input_format: AudioFormat,
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub bitrate: Option<u32>,
}

#[pymethods]
impl OpusEncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format, chunk_samples, bitrate=Some(96_000)))]
    fn new(input_format: AudioFormat, chunk_samples: usize, bitrate: Option<u32>) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        Ok(Self {
            input_format,
            chunk_samples,
            bitrate,
        })
    }
}

#[pyclass(name = "FlacEncoderConfig")]
#[derive(Clone)]
pub struct FlacEncoderConfigPy {
    #[pyo3(get)]
    pub input_format: AudioFormat,
    #[pyo3(get)]
    pub chunk_samples: usize,
    /// 0..=12（FFmpeg backend 常见语义）；None=默认
    #[pyo3(get)]
    pub compression_level: Option<i32>,
}

#[pymethods]
impl FlacEncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format, chunk_samples, compression_level=None))]
    fn new(input_format: AudioFormat, chunk_samples: usize, compression_level: Option<i32>) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        Ok(Self {
            input_format,
            chunk_samples,
            compression_level,
        })
    }
}

#[pyclass(name = "OpusDecoderConfig")]
#[derive(Clone)]
pub struct OpusDecoderConfigPy {
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub packet_time_base_den: i32,
    /// 可选：Opus extradata（通常为 OpusHead；raw packet 流时可能需要）。
    #[pyo3(get)]
    pub extradata: Option<Vec<u8>>,
}

#[pymethods]
impl OpusDecoderConfigPy {
    #[new]
    #[pyo3(signature = (chunk_samples, packet_time_base_den=48000, extradata=None))]
    fn new(chunk_samples: usize, packet_time_base_den: i32, extradata: Option<Vec<u8>>) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        if packet_time_base_den <= 0 {
            return Err(PyValueError::new_err("packet_time_base_den 必须 > 0"));
        }
        Ok(Self {
            chunk_samples,
            packet_time_base_den,
            extradata,
        })
    }
}

#[pyclass(name = "FlacDecoderConfig")]
#[derive(Clone)]
pub struct FlacDecoderConfigPy {
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub packet_time_base_den: i32,
}

#[pymethods]
impl FlacDecoderConfigPy {
    #[new]
    #[pyo3(signature = (chunk_samples, packet_time_base_den=48000))]
    fn new(chunk_samples: usize, packet_time_base_den: i32) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        if packet_time_base_den <= 0 {
            return Err(PyValueError::new_err("packet_time_base_den 必须 > 0"));
        }
        Ok(Self {
            chunk_samples,
            packet_time_base_den,
        })
    }
}

fn ascontig_cast_2d<'py>(
    py: Python<'py>,
    pcm: &Bound<'py, PyAny>,
    dtype_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import_bound("numpy")?;
    let dtype = np.getattr(dtype_name)?;
    // numpy.ascontiguousarray(a, dtype=...)
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", dtype)?;
    np.getattr("ascontiguousarray")?.call((pcm,), Some(&kwargs))
}

fn ndarray_to_frame_planar<'py, T: Copy + Element>(
    arr_any: &Bound<'py, PyAny>,
    fmt: RsAudioFormat,
) -> PyResult<crate::common::audio::audio::AudioFrame> {
    let arr = arr_any.downcast::<PyArray2<T>>()?;
    let ro = arr.readonly();
    let view = ro.as_array();

    let ch = view.shape()[0];
    let ns = view.shape()[1];
    if ch != fmt.channels() as usize {
        return Err(PyValueError::new_err("pcm 的 channels 维度与 config.input_format.channels 不一致"));
    }

    // 每行视作一个 planar plane；拷贝成 bytes
    let bps = fmt.sample_format.bytes_per_sample();
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(ch);
    for c in 0..ch {
        let row = view.row(c);
        let slice = row
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("pcm 需要是 C contiguous 的 (channels, samples)"))?;
        let bytes_len = ns * bps;
        let mut out = vec![0u8; bytes_len];
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr() as *const u8, out.as_mut_ptr(), bytes_len);
        }
        planes.push(out);
    }

    crate::common::audio::audio::AudioFrame::from_planes(
        fmt,
        ns,
        Rational::new(1, fmt.sample_rate as i32),
        None,
        planes,
    )
    .map_err(map_audio_err)
}

fn ndarray_to_frame_interleaved<'py, T: Copy + Element>(
    arr_any: &Bound<'py, PyAny>,
    fmt: RsAudioFormat,
) -> PyResult<crate::common::audio::audio::AudioFrame> {
    // 约定：interleaved numpy shape = (samples, channels)
    let arr = arr_any.downcast::<PyArray2<T>>()?;
    let ro = arr.readonly();
    let view = ro.as_array();
    let ns = view.shape()[0];
    let ch = view.shape()[1];
    if ch != fmt.channels() as usize {
        return Err(PyValueError::new_err("pcm 的 channels 维度与 format.channels 不一致"));
    }

    let bps = fmt.sample_format.bytes_per_sample();
    let bytes_len = ns * ch * bps;
    let mut out = vec![0u8; bytes_len];

    let slice = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("pcm 需要是 C contiguous 的 (samples, channels)"))?;

    unsafe {
        std::ptr::copy_nonoverlapping(slice.as_ptr() as *const u8, out.as_mut_ptr(), bytes_len);
    }

    crate::common::audio::audio::AudioFrame::from_planes(
        fmt,
        ns,
        Rational::new(1, fmt.sample_rate as i32),
        None,
        vec![out],
    )
    .map_err(map_audio_err)
}

fn packet_time_base_from_den(den: i32) -> Rational {
    Rational::new(1, den)
}

fn map_runner_err(e: crate::runner::error::RunnerError) -> PyErr {
    match e {
        crate::runner::error::RunnerError::Codec(ce) => map_codec_err(ce),
        crate::runner::error::RunnerError::Io(ioe) => PyRuntimeError::new_err(ioe.to_string()),
        crate::runner::error::RunnerError::InvalidData(msg) => PyValueError::new_err(msg),
        crate::runner::error::RunnerError::InvalidState(msg) => PyRuntimeError::new_err(msg),
    }
}

fn pyerr_to_runner_err(e: PyErr) -> RunnerError {
    RunnerError::Codec(CodecError::Other(e.to_string()))
}

fn map_file_err(e: AudioFileError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn push_encoder_from_fifo(
    enc: &mut dyn AudioEncoder,
    fifo: &mut AudioFifo,
    chunk_samples: usize,
    out_q: &mut VecDeque<CodecPacket>,
    force: bool,
) -> Result<(), CodecError> {
    loop {
        if fifo.available_samples() >= chunk_samples {
            // 正常 pop 一个 chunk
            let frame = fifo
                .pop_frame(chunk_samples)
                .map_err(|_| CodecError::InvalidData("failed to pop frame from fifo"))?
                .expect("available_samples checked");
            enc.send_frame(Some(&frame))?;
            loop {
                match enc.receive_packet() {
                    Ok(pkt) => out_q.push_back(pkt),
                    Err(CodecError::Again) => break,
                    Err(e) => return Err(e),
                }
            }
            // 继续循环，看看是否还能再吐更多 output（通常 1:1）
            continue;
        }

        if !force {
            return Ok(());
        }

        let left = fifo.available_samples();
        if left == 0 {
            return Ok(());
        }

        // 强制 flush：把剩余不满 chunk 的 samples 作为“最后一帧”送入 encoder
        let frame = fifo
            .pop_frame(left)
            .map_err(|_| CodecError::InvalidData("failed to pop last frame from fifo"))?
            .expect("left > 0 checked");
        enc.send_frame(Some(&frame))?;
        loop {
            match enc.receive_packet() {
                Ok(pkt) => out_q.push_back(pkt),
                Err(CodecError::Again) => break,
                Err(e) => return Err(e),
            }
        }
        return Ok(());
    }
}

fn ensure_planar_frame(frame: crate::common::audio::audio::AudioFrame) -> Result<crate::common::audio::audio::AudioFrame, AudioError> {
    let fmt = *frame.format_ref();
    if fmt.is_planar() {
        return Ok(frame);
    }

    let st = fmt.sample_format.sample_type();
    let planar_fmt = RsAudioFormat {
        sample_rate: fmt.sample_rate,
        sample_format: sample_format_from_type(st, true),
        channel_layout: fmt.channel_layout,
    };
    let ch = planar_fmt.channels() as usize;
    let ns = frame.nb_samples();
    let bps = planar_fmt.sample_format.bytes_per_sample();
    let src = frame.planes_ref().get(0).ok_or(AudioError::InvalidPlaneCount { expected: 1, actual: 0 })?;
    let mut planes: Vec<Vec<u8>> = vec![vec![0u8; ns * bps]; ch];
    for s in 0..ns {
        for c in 0..ch {
            let src_off = (s * ch + c) * bps;
            let dst_off = s * bps;
            planes[c][dst_off..dst_off + bps].copy_from_slice(&src[src_off..src_off + bps]);
        }
    }
    crate::common::audio::audio::AudioFrame::from_planes(planar_fmt, ns, frame.time_base(), frame.pts(), planes)
}

fn frame_to_numpy_planar<'py>(py: Python<'py>, frame: &crate::common::audio::audio::AudioFrame) -> PyResult<PyObject> {
    let fmt = *frame.format_ref();
    let ch = fmt.channels() as usize;
    let ns = frame.nb_samples();
    let st = fmt.sample_format.sample_type();

    let np = py.import_bound("numpy")?;
    let dtype_name = match st {
        SampleType::U8 => "uint8",
        SampleType::I16 => "int16",
        SampleType::I32 => "int32",
        SampleType::I64 => "int64",
        SampleType::F32 => "float32",
        SampleType::F64 => "float64",
    };
    let dtype = np.getattr(dtype_name)?;
    let arr_any = np.getattr("empty")?.call1(((ch, ns), dtype))?;

    // 逐 dtype 分支写入（避免依赖 numpy 的复杂 slice API）
    macro_rules! fill_planar {
        ($t:ty) => {{
            let arr_t = arr_any.downcast::<PyArray2<$t>>()?;
            let mut rw = unsafe { arr_t.as_array_mut() };
            if fmt.is_planar() {
                for c in 0..ch {
                    let src_b = frame.plane(c).ok_or_else(|| PyRuntimeError::new_err("missing plane"))?;
                    let src = unsafe {
                        std::slice::from_raw_parts(src_b.as_ptr() as *const $t, ns)
                    };
                    for s in 0..ns {
                        rw[[c, s]] = src[s];
                    }
                }
            } else {
                let src_b = frame.plane(0).ok_or_else(|| PyRuntimeError::new_err("missing plane 0"))?;
                let src = unsafe {
                    std::slice::from_raw_parts(src_b.as_ptr() as *const $t, ns * ch)
                };
                for s in 0..ns {
                    for c in 0..ch {
                        rw[[c, s]] = src[s * ch + c];
                    }
                }
            }
            Ok::<PyObject, PyErr>(arr_t.to_object(py))
        }};
    }

    match st {
        SampleType::U8 => fill_planar!(u8),
        SampleType::I16 => fill_planar!(i16),
        SampleType::I32 => fill_planar!(i32),
        SampleType::I64 => fill_planar!(i64),
        SampleType::F32 => fill_planar!(f32),
        SampleType::F64 => fill_planar!(f64),
    }
}

fn frame_to_numpy_interleaved<'py>(py: Python<'py>, frame: &crate::common::audio::audio::AudioFrame) -> PyResult<PyObject> {
    // 约定：interleaved numpy shape = (samples, channels)
    let fmt = *frame.format_ref();
    let ch = fmt.channels() as usize;
    let ns = frame.nb_samples();
    let st = fmt.sample_format.sample_type();

    let np = py.import_bound("numpy")?;
    let dtype_name = match st {
        SampleType::U8 => "uint8",
        SampleType::I16 => "int16",
        SampleType::I32 => "int32",
        SampleType::I64 => "int64",
        SampleType::F32 => "float32",
        SampleType::F64 => "float64",
    };
    let dtype = np.getattr(dtype_name)?;
    let arr_any = np.getattr("empty")?.call1(((ns, ch), dtype))?;

    macro_rules! fill_interleaved {
        ($t:ty) => {{
            let arr_t = arr_any.downcast::<PyArray2<$t>>()?;
            let mut rw = unsafe { arr_t.as_array_mut() };
            if fmt.is_planar() {
                for s in 0..ns {
                    for c in 0..ch {
                        let src_b = frame.plane(c).ok_or_else(|| PyRuntimeError::new_err("missing plane"))?;
                        let src = unsafe { std::slice::from_raw_parts(src_b.as_ptr() as *const $t, ns) };
                        rw[[s, c]] = src[s];
                    }
                }
            } else {
                let src_b = frame.plane(0).ok_or_else(|| PyRuntimeError::new_err("missing plane 0"))?;
                let src = unsafe { std::slice::from_raw_parts(src_b.as_ptr() as *const $t, ns * ch) };
                for s in 0..ns {
                    for c in 0..ch {
                        rw[[s, c]] = src[s * ch + c];
                    }
                }
            }
            Ok::<PyObject, PyErr>(arr_t.to_object(py))
        }};
    }

    match st {
        SampleType::U8 => fill_interleaved!(u8),
        SampleType::I16 => fill_interleaved!(i16),
        SampleType::I32 => fill_interleaved!(i32),
        SampleType::I64 => fill_interleaved!(i64),
        SampleType::F32 => fill_interleaved!(f32),
        SampleType::F64 => fill_interleaved!(f64),
    }
}

fn frame_to_numpy<'py>(py: Python<'py>, frame: &crate::common::audio::audio::AudioFrame, planar: bool) -> PyResult<PyObject> {
    if planar {
        frame_to_numpy_planar(py, frame)
    } else {
        frame_to_numpy_interleaved(py, frame)
    }
}

fn sample_type_to_str(st: SampleType) -> &'static str {
    match st {
        SampleType::U8 => "u8",
        SampleType::I16 => "i16",
        SampleType::I32 => "i32",
        SampleType::I64 => "i64",
        SampleType::F32 => "f32",
        SampleType::F64 => "f64",
    }
}

fn audio_format_from_rs(fmt: RsAudioFormat) -> AudioFormat {
    AudioFormat {
        sample_rate: fmt.sample_rate,
        channels: fmt.channels(),
        sample_type: sample_type_to_str(fmt.sample_format.sample_type()).to_string(),
        planar: fmt.is_planar(),
        channel_layout_mask: fmt.channel_layout.mask,
    }
}

fn node_kind_to_str(k: NodeBufferKind) -> &'static str {
    match k {
        NodeBufferKind::Pcm => "pcm",
        NodeBufferKind::Packet => "packet",
    }
}

fn node_kind_from_str(s: &str) -> Option<NodeBufferKind> {
    match s.to_ascii_lowercase().as_str() {
        "pcm" => Some(NodeBufferKind::Pcm),
        "packet" => Some(NodeBufferKind::Packet),
        _ => None,
    }
}

#[pyclass(name = "Packet")]
#[derive(Clone)]
pub struct PacketPy {
    #[pyo3(get)]
    pub data: Vec<u8>,
    #[pyo3(get)]
    pub time_base_num: i32,
    #[pyo3(get)]
    pub time_base_den: i32,
    #[pyo3(get)]
    pub pts: Option<i64>,
    #[pyo3(get)]
    pub dts: Option<i64>,
    #[pyo3(get)]
    pub duration: Option<i64>,
    /// 原始 flags bitmask（当前库内部 flags 还很小集合；先透传 u32）。
    #[pyo3(get)]
    pub flags: u32,
}

#[pymethods]
impl PacketPy {
    #[new]
    #[pyo3(signature = (data, time_base_num=1, time_base_den=48000, pts=None, dts=None, duration=None, flags=0))]
    fn new(
        data: Vec<u8>,
        time_base_num: i32,
        time_base_den: i32,
        pts: Option<i64>,
        dts: Option<i64>,
        duration: Option<i64>,
        flags: u32,
    ) -> PyResult<Self> {
        if time_base_den == 0 || time_base_num == 0 {
            return Err(PyValueError::new_err("time_base_num/time_base_den 必须非 0"));
        }
        Ok(Self {
            data,
            time_base_num,
            time_base_den,
            pts,
            dts,
            duration,
            flags,
        })
    }
}

impl PacketPy {
    fn to_rs(&self) -> CodecPacket {
        let flags = PacketFlags::from_bits(self.flags);
        CodecPacket {
            data: self.data.clone(),
            time_base: Rational::new(self.time_base_num, self.time_base_den),
            pts: self.pts,
            dts: self.dts,
            duration: self.duration,
            flags,
        }
    }

    fn from_rs(p: &CodecPacket) -> Self {
        Self {
            data: p.data.clone(),
            time_base_num: p.time_base.num,
            time_base_den: p.time_base.den,
            pts: p.pts,
            dts: p.dts,
            duration: p.duration,
            flags: p.flags.bits(),
        }
    }
}

/// Python 侧 NodeBuffer（pcm 或 packet），用于动态 pipeline/runner 交互。
#[pyclass(name = "NodeBuffer")]
pub struct NodeBufferPy {
    inner: Option<NodeBuffer>,
}

#[pymethods]
impl NodeBufferPy {
    /// 构造 PCM buffer：
    ///
    /// - format.planar=True  => numpy shape=(channels, samples)
    /// - format.planar=False => numpy shape=(samples, channels)
    #[staticmethod]
    #[pyo3(signature = (pcm, format, pts=None, time_base_num=None, time_base_den=None))]
    fn pcm(
        py: Python<'_>,
        pcm: &Bound<'_, PyAny>,
        format: AudioFormat,
        pts: Option<i64>,
        time_base_num: Option<i32>,
        time_base_den: Option<i32>,
    ) -> PyResult<Self> {
        let rs_fmt = format.to_rs()?;
        let st = format.sample_type_rs()?;
        let dtype_name = match st {
            SampleType::U8 => "uint8",
            SampleType::I16 => "int16",
            SampleType::I32 => "int32",
            SampleType::I64 => "int64",
            SampleType::F32 => "float32",
            SampleType::F64 => "float64",
        };
        let arr_any = ascontig_cast_2d(py, pcm, dtype_name)?;
        let mut frame = if rs_fmt.is_planar() {
            match st {
                SampleType::U8 => ndarray_to_frame_planar::<u8>(&arr_any, rs_fmt)?,
                SampleType::I16 => ndarray_to_frame_planar::<i16>(&arr_any, rs_fmt)?,
                SampleType::I32 => ndarray_to_frame_planar::<i32>(&arr_any, rs_fmt)?,
                SampleType::I64 => ndarray_to_frame_planar::<i64>(&arr_any, rs_fmt)?,
                SampleType::F32 => ndarray_to_frame_planar::<f32>(&arr_any, rs_fmt)?,
                SampleType::F64 => ndarray_to_frame_planar::<f64>(&arr_any, rs_fmt)?,
            }
        } else {
            match st {
                SampleType::U8 => ndarray_to_frame_interleaved::<u8>(&arr_any, rs_fmt)?,
                SampleType::I16 => ndarray_to_frame_interleaved::<i16>(&arr_any, rs_fmt)?,
                SampleType::I32 => ndarray_to_frame_interleaved::<i32>(&arr_any, rs_fmt)?,
                SampleType::I64 => ndarray_to_frame_interleaved::<i64>(&arr_any, rs_fmt)?,
                SampleType::F32 => ndarray_to_frame_interleaved::<f32>(&arr_any, rs_fmt)?,
                SampleType::F64 => ndarray_to_frame_interleaved::<f64>(&arr_any, rs_fmt)?,
            }
        };
        if let Some(p) = pts {
            frame.set_pts(Some(p));
        }
        if let (Some(n), Some(d)) = (time_base_num, time_base_den) {
            frame.set_time_base(Rational::new(n, d)).map_err(map_audio_err)?;
        }
        Ok(Self {
            inner: Some(NodeBuffer::Pcm(frame)),
        })
    }

    /// 构造 Packet buffer。
    #[staticmethod]
    fn packet(pkt: PacketPy) -> PyResult<Self> {
        Ok(Self {
            inner: Some(NodeBuffer::Packet(pkt.to_rs())),
        })
    }

    #[getter]
    fn kind(&self) -> PyResult<&'static str> {
        let Some(inner) = &self.inner else {
            return Err(PyRuntimeError::new_err("NodeBuffer 已被移动（不可再次使用）"));
        };
        Ok(node_kind_to_str(inner.kind()))
    }

    /// 如果是 pcm，返回 numpy ndarray（shape=(channels,samples)）；否则返回 None。
    fn as_pcm(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let Some(inner) = &self.inner else {
            return Err(PyRuntimeError::new_err("NodeBuffer 已被移动（不可再次使用）"));
        };
        match inner {
            NodeBuffer::Pcm(f) => Ok(Some(frame_to_numpy(py, f, true)?)),
            _ => Ok(None),
        }
    }

    /// 如果是 pcm，返回 numpy ndarray（可选输出 interleaved）；否则返回 None。
    #[pyo3(signature = (layout="planar"))]
    fn as_pcm_with_layout(&self, py: Python<'_>, layout: &str) -> PyResult<Option<PyObject>> {
        let Some(inner) = &self.inner else {
            return Err(PyRuntimeError::new_err("NodeBuffer 已被移动（不可再次使用）"));
        };
        let planar = match layout.to_ascii_lowercase().as_str() {
            "planar" => true,
            "interleaved" => false,
            _ => return Err(PyValueError::new_err("layout 仅支持: planar/interleaved")),
        };
        match inner {
            NodeBuffer::Pcm(f) => Ok(Some(frame_to_numpy(py, f, planar)?)),
            _ => Ok(None),
        }
    }

    /// 如果是 packet，返回 Packet；否则返回 None。
    fn as_packet(&self) -> PyResult<Option<PacketPy>> {
        let Some(inner) = &self.inner else {
            return Err(PyRuntimeError::new_err("NodeBuffer 已被移动（不可再次使用）"));
        };
        match inner {
            NodeBuffer::Packet(p) => Ok(Some(PacketPy::from_rs(p))),
            _ => Ok(None),
        }
    }

    /// 如果是 pcm，返回 (AudioFormat, pts, (time_base_num,time_base_den))；否则返回 None。
    fn pcm_info(&self) -> PyResult<Option<(AudioFormat, Option<i64>, (i32, i32))>> {
        let Some(inner) = &self.inner else {
            return Err(PyRuntimeError::new_err("NodeBuffer 已被移动（不可再次使用）"));
        };
        match inner {
            NodeBuffer::Pcm(f) => {
                let fmt = audio_format_from_rs(*f.format_ref());
                let tb = f.time_base();
                Ok(Some((fmt, f.pts(), (tb.num, tb.den))))
            }
            _ => Ok(None),
        }
    }
}

impl NodeBufferPy {
    fn take_inner(&mut self) -> PyResult<NodeBuffer> {
        self.inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("NodeBuffer 已被移动（不可再次使用）"))
    }
}

#[pyclass(name = "DynNode")]
pub struct DynNodePy {
    inner: Option<Box<dyn DynNode>>,
    in_kind: NodeBufferKind,
    out_kind: NodeBufferKind,
    name: &'static str,
}

impl DynNodePy {
    fn new_boxed(node: Box<dyn DynNode>) -> Self {
        let name = node.name();
        let in_kind = node.input_kind();
        let out_kind = node.output_kind();
        Self {
            inner: Some(node),
            in_kind,
            out_kind,
            name,
        }
    }

    fn take_inner(&mut self) -> PyResult<Box<dyn DynNode>> {
        self.inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("DynNode 已被移动（不可再次使用）"))
    }
}

#[pymethods]
impl DynNodePy {
    #[getter]
    fn name(&self) -> &str {
        self.name
    }

    #[getter]
    fn input_kind(&self) -> &'static str {
        node_kind_to_str(self.in_kind)
    }

    #[getter]
    fn output_kind(&self) -> &'static str {
        node_kind_to_str(self.out_kind)
    }
}

/// 让 `Box<dyn AudioEncoder>` 能接到运行时 pipeline：实现一个 DynNode 适配器。
struct BoxedEncoderNode {
    e: Box<dyn AudioEncoder>,
}

impl DynNode for BoxedEncoderNode {
    fn name(&self) -> &'static str {
        self.e.name()
    }

    fn input_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Pcm
    }

    fn output_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Packet
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> crate::codec::error::CodecResult<()> {
        match input {
            None => self.e.send_frame(None),
            Some(NodeBuffer::Pcm(f)) => self.e.send_frame(Some(&f as &dyn AudioFrameView)),
            Some(_) => Err(CodecError::InvalidData("encoder expects PCM input")),
        }
    }

    fn pull(&mut self) -> crate::codec::error::CodecResult<NodeBuffer> {
        self.e.receive_packet().map(NodeBuffer::Packet)
    }
}

/// 让 `Box<dyn AudioDecoder>` 能接到运行时 pipeline：实现一个 DynNode 适配器。
struct BoxedDecoderNode {
    d: Box<dyn AudioDecoder>,
}

impl DynNode for BoxedDecoderNode {
    fn name(&self) -> &'static str {
        self.d.name()
    }

    fn input_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Packet
    }

    fn output_kind(&self) -> NodeBufferKind {
        NodeBufferKind::Pcm
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> crate::codec::error::CodecResult<()> {
        match input {
            None => self.d.send_packet(None),
            Some(NodeBuffer::Packet(p)) => self.d.send_packet(Some(p)),
            Some(_) => Err(CodecError::InvalidData("decoder expects Packet input")),
        }
    }

    fn pull(&mut self) -> crate::codec::error::CodecResult<NodeBuffer> {
        self.d.receive_frame().map(NodeBuffer::Pcm)
    }
}

/// 创建一个 identity 节点（pcm 或 packet）。
#[pyfunction]
fn make_identity_node(kind: &str) -> PyResult<DynNodePy> {
    let k = node_kind_from_str(kind).ok_or_else(|| PyValueError::new_err("kind 仅支持: pcm/packet"))?;
    Ok(DynNodePy::new_boxed(Box::new(IdentityNode::new(k))))
}

/// 创建一个 resample 节点（PCM->PCM）。
///
/// - `in_format/out_format`：必须完整匹配（包括 planar/sample_type/channels 等）
/// - `out_chunk_samples`：可选，启用输出重分帧（例如 Opus 前常设 960@48k）
/// - `pad_final`：flush 时是否补 0 到 `out_chunk_samples`
#[pyfunction]
#[pyo3(signature = (in_format, out_format, out_chunk_samples=None, pad_final=true))]
fn make_resample_node(in_format: AudioFormat, out_format: AudioFormat, out_chunk_samples: Option<usize>, pad_final: bool) -> PyResult<DynNodePy> {
    let in_fmt = in_format.to_rs()?;
    let out_fmt = out_format.to_rs()?;
    let mut p = ResampleProcessor::new(in_fmt, out_fmt).map_err(map_codec_err)?;
    p.set_output_chunker(out_chunk_samples, pad_final).map_err(map_codec_err)?;
    Ok(DynNodePy::new_boxed(Box::new(ProcessorNode::new(p))))
}

/// 创建一个 encoder 节点（PCM->Packet）。注意：此处的 config 里 `chunk_samples` 会被忽略（由上游分帧决定）。
#[pyfunction]
fn make_encoder_node(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<DynNodePy> {
    let codec_norm = codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac/opus/flac"))?;
    let enc: Box<dyn AudioEncoder> = match codec_norm {
        "wav" => {
            let cfg = config.extract::<WavEncoderConfigPy>()?;
            let input_format = cfg.input_format.to_rs()?;
            Box::new(WavEncoder::new(WavEncoderConfig { input_format }).map_err(map_codec_err)?)
        }
        "mp3" => {
            let cfg = config.extract::<Mp3EncoderConfigPy>()?;
            let input_format = cfg.input_format.to_rs()?;
            Box::new(Mp3Encoder::new(Mp3EncoderConfig { input_format, bitrate: cfg.bitrate }).map_err(map_codec_err)?)
        }
        "aac" => {
            let cfg = config.extract::<AacEncoderConfigPy>()?;
            let input_format = cfg.input_format.to_rs()?;
            Box::new(AacEncoder::new(AacEncoderConfig { input_format, bitrate: cfg.bitrate }).map_err(map_codec_err)?)
        }
        "opus" => {
            let cfg = config.extract::<OpusEncoderConfigPy>()?;
            let input_format = cfg.input_format.to_rs()?;
            Box::new(OpusEncoder::new(OpusEncoderConfig { input_format, bitrate: cfg.bitrate }).map_err(map_codec_err)?)
        }
        "flac" => {
            let cfg = config.extract::<FlacEncoderConfigPy>()?;
            let input_format = cfg.input_format.to_rs()?;
            Box::new(FlacEncoder::new(FlacEncoderConfig { input_format, compression_level: cfg.compression_level }).map_err(map_codec_err)?)
        }
        _ => return Err(PyValueError::new_err("unsupported codec")),
    };
    Ok(DynNodePy::new_boxed(Box::new(BoxedEncoderNode { e: enc })))
}

/// 创建一个 decoder 节点（Packet->PCM）。注意：此处的 config 里 `chunk_samples` 会被忽略（由下游取帧决定）。
#[pyfunction]
fn make_decoder_node(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<DynNodePy> {
    let codec_norm = codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac/opus/flac"))?;
    let dec: Box<dyn AudioDecoder> = match codec_norm {
        "wav" => {
            let cfg = config.extract::<WavDecoderConfigPy>()?;
            let out_fmt = cfg.output_format.to_rs()?;
            Box::new(WavDecoder::new(out_fmt).map_err(map_codec_err)?)
        }
        "mp3" => Box::new(Mp3Decoder::new().map_err(map_codec_err)?),
        "aac" => Box::new(AacDecoder::new().map_err(map_codec_err)?),
        "opus" => {
            let cfg = config.extract::<OpusDecoderConfigPy>()?;
            if let Some(ed) = cfg.extradata.as_ref() {
                Box::new(OpusDecoder::new_with_extradata(ed).map_err(map_codec_err)?)
            } else {
                Box::new(OpusDecoder::new().map_err(map_codec_err)?)
            }
        }
        "flac" => Box::new(FlacDecoder::new().map_err(map_codec_err)?),
        _ => return Err(PyValueError::new_err("unsupported codec")),
    };
    Ok(DynNodePy::new_boxed(Box::new(BoxedDecoderNode { d: dec })))
}

/// Python 侧 AsyncDynPipeline（动态节点列表）。
#[pyclass(name = "AsyncDynPipeline")]
pub struct AsyncDynPipelinePy {
    rt: Runtime,
    p: AsyncDynPipeline,
    in_kind: NodeBufferKind,
    out_kind: NodeBufferKind,
}

#[pymethods]
impl AsyncDynPipelinePy {
    #[new]
    fn new(py: Python<'_>, nodes: Vec<Py<DynNodePy>>) -> PyResult<Self> {
        if nodes.is_empty() {
            return Err(PyValueError::new_err("nodes 不能为空"));
        }
        // 取出节点（move）
        let mut boxed: Vec<Box<dyn DynNode>> = Vec::with_capacity(nodes.len());
        let mut in_kind: Option<NodeBufferKind> = None;
        let mut out_kind: Option<NodeBufferKind> = None;
        for (i, n) in nodes.into_iter().enumerate() {
            let mut nb = n.bind(py).borrow_mut();
            let node_in = nb.in_kind;
            let node_out = nb.out_kind;
            if i == 0 {
                in_kind = Some(node_in);
            }
            out_kind = Some(node_out);
            boxed.push(nb.take_inner()?);
        }

        let rt = Runtime::new().map_err(|e| PyRuntimeError::new_err(format!("tokio Runtime init failed: {e}")))?;
        let _guard = rt.enter();
        let p = AsyncDynPipeline::new(boxed).map_err(map_codec_err)?;

        Ok(Self {
            rt,
            p,
            in_kind: in_kind.unwrap(),
            out_kind: out_kind.unwrap(),
        })
    }

    #[getter]
    fn input_kind(&self) -> &'static str {
        node_kind_to_str(self.in_kind)
    }

    #[getter]
    fn output_kind(&self) -> &'static str {
        node_kind_to_str(self.out_kind)
    }

    fn push(&mut self, py: Python<'_>, buf: Py<NodeBufferPy>) -> PyResult<()> {
        let mut b = buf.bind(py).borrow_mut();
        let inner = b.take_inner()?;
        if inner.kind() != self.in_kind {
            return Err(PyValueError::new_err("NodeBuffer kind 与 pipeline input_kind 不匹配"));
        }
        self.p.push_frame(inner).map_err(map_codec_err)
    }

    fn flush(&mut self) -> PyResult<()> {
        self.p.flush().map_err(map_codec_err)
    }

    fn try_get(&mut self) -> PyResult<Option<NodeBufferPy>> {
        match self.p.try_get_frame() {
            Ok(v) => Ok(Some(NodeBufferPy { inner: Some(v) })),
            Err(CodecError::Again) => Ok(None),
            Err(CodecError::Eof) => Ok(None),
            Err(e) => Err(map_codec_err(e)),
        }
    }

    /// 阻塞等待一个输出（直到拿到一帧或 EOF）。
    fn get(&mut self, py: Python<'_>) -> PyResult<Option<NodeBufferPy>> {
        // Python 侧这是同步阻塞调用：释放 GIL，避免卡住其它 Python 线程
        let fut = self.p.get_frame();
        let res = py.allow_threads(|| {
            let _guard = self.rt.enter();
            self.rt.block_on(fut)
        });
        match res {
            Ok(v) => Ok(Some(NodeBufferPy { inner: Some(v) })),
            Err(CodecError::Eof) => Ok(None),
            Err(e) => Err(map_codec_err(e)),
        }
    }
}

/// 让 Python 对象实现 AudioSource：要求提供 `pull() -> Optional[NodeBuffer]`。
struct PyCallbackSource {
    obj: Py<PyAny>,
}

impl AudioSource for PyCallbackSource {
    type Out = NodeBuffer;

    fn name(&self) -> &'static str {
        "py-callback-source"
    }

    fn pull(&mut self) -> RunnerResult<Option<Self::Out>> {
        Python::with_gil(|py| {
            let o = self.obj.bind(py);
            let ret = o.call_method0("pull").map_err(pyerr_to_runner_err)?;
            if ret.is_none() {
                return Ok(None);
            }
            let nb_py: Py<NodeBufferPy> = ret.extract().map_err(|_| {
                RunnerError::InvalidData("Python source.pull() 必须返回 NodeBuffer 或 None")
            })?;
            let mut nb = nb_py.bind(py).borrow_mut();
            let inner = nb.take_inner().map_err(|e| pyerr_to_runner_err(e))?;
            Ok(Some(inner))
        })
    }
}

/// 让 Python 对象实现 AudioSink：要求提供 `push(buf: NodeBuffer)` + `finalize()`。
struct PyCallbackSink {
    obj: Py<PyAny>,
}

impl AudioSink for PyCallbackSink {
    type In = NodeBuffer;

    fn name(&self) -> &'static str {
        "py-callback-sink"
    }

    fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        Python::with_gil(|py| {
            let o = self.obj.bind(py);
            let nb = Py::new(py, NodeBufferPy { inner: Some(input) }).map_err(pyerr_to_runner_err)?;
            o.call_method1("push", (nb,)).map_err(pyerr_to_runner_err)?;
            Ok(())
        })
    }

    fn finalize(&mut self) -> RunnerResult<()> {
        Python::with_gil(|py| {
            let o = self.obj.bind(py);
            o.call_method0("finalize").map_err(pyerr_to_runner_err)?;
            Ok(())
        })
    }
}

/// Python 侧 AsyncDynRunner（动态节点列表 + Python Source/Sink）。
///
/// - `source` 需要实现：`pull() -> Optional[NodeBuffer]`
/// - `sink` 需要实现：`push(buf: NodeBuffer)` / `finalize()`
#[pyclass(name = "AsyncDynRunner")]
pub struct AsyncDynRunnerPy {
    rt: Runtime,
    runner: AutoRunner<AsyncDynPipeline, PyCallbackSource, PyCallbackSink>,
}

#[pymethods]
impl AsyncDynRunnerPy {
    #[new]
    fn new(py: Python<'_>, source: Py<PyAny>, nodes: Vec<Py<DynNodePy>>, sink: Py<PyAny>) -> PyResult<Self> {
        if nodes.is_empty() {
            return Err(PyValueError::new_err("nodes 不能为空"));
        }
        // move nodes
        let mut boxed: Vec<Box<dyn DynNode>> = Vec::with_capacity(nodes.len());
        for n in nodes.into_iter() {
            let mut nb = n.bind(py).borrow_mut();
            boxed.push(nb.take_inner()?);
        }

        let rt = Runtime::new().map_err(|e| PyRuntimeError::new_err(format!("tokio Runtime init failed: {e}")))?;
        let _guard = rt.enter();
        let pipeline = AsyncDynPipeline::new(boxed).map_err(map_codec_err)?;
        let runner = AutoRunner::new(
            PyCallbackSource { obj: source },
            pipeline,
            PyCallbackSink { obj: sink },
        );
        Ok(Self { rt, runner })
    }

    /// 同步阻塞执行到完成（释放 GIL）。
    fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        let res = py.allow_threads(|| {
            let _guard = self.rt.enter();
            self.rt.block_on(self.runner.execute())
        });
        res.map_err(map_runner_err)
    }
}

fn file_format_from_str(s: &str) -> Option<&'static str> {
    match s.to_ascii_lowercase().as_str() {
        "wav" => Some("wav"),
        "mp3" => Some("mp3"),
        "aac" | "aac_adts" | "adts" => Some("aac_adts"),
        "flac" => Some("flac"),
        "opus" | "opus_ogg" | "ogg_opus" => Some("opus_ogg"),
        _ => None,
    }
}

/// Python 侧 AudioFileReader：可作为 `AsyncDynRunner` 的 source（实现 pull()）。
#[pyclass(name = "AudioFileReader", unsendable)]
pub struct AudioFileReaderPy {
    r: RsAudioFileReader,
}

#[pymethods]
impl AudioFileReaderPy {
    #[new]
    fn new(path: String, format: &str) -> PyResult<Self> {
        let fmt = file_format_from_str(format).ok_or_else(|| PyValueError::new_err("format 仅支持: wav/mp3/aac_adts/flac/opus_ogg"))?;
        let cfg = match fmt {
            "wav" => AudioFileReadConfig::Wav,
            "mp3" => AudioFileReadConfig::Mp3,
            "aac_adts" => AudioFileReadConfig::AacAdts,
            "flac" => AudioFileReadConfig::Flac,
            "opus_ogg" => AudioFileReadConfig::OpusOgg,
            _ => return Err(PyValueError::new_err("unsupported format")),
        };
        let r = RsAudioFileReader::open(path, cfg).map_err(map_file_err)?;
        Ok(Self { r })
    }

    /// 读取下一帧 PCM（numpy），EOF 返回 None。
    fn next_frame(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match AudioReader::next_frame(&mut self.r).map_err(map_file_err)? {
            Some(f) => Ok(Some(frame_to_numpy(py, &f, true)?)),
            None => Ok(None),
        }
    }

    /// `AsyncDynRunner` 兼容：pull() -> Optional[NodeBuffer]（输出为 PCM）。
    fn pull(&mut self, _py: Python<'_>) -> PyResult<Option<NodeBufferPy>> {
        match AudioReader::next_frame(&mut self.r).map_err(map_file_err)? {
            Some(f) => Ok(Some(NodeBufferPy { inner: Some(NodeBuffer::Pcm(f)) })),
            None => Ok(None),
        }
    }
}

/// Python 侧 AudioFileWriter：可作为 `AsyncDynRunner` 的 sink（实现 push/finalize）。
#[pyclass(name = "AudioFileWriter", unsendable)]
pub struct AudioFileWriterPy {
    w: RsAudioFileWriter,
    input_format: RsAudioFormat,
    sample_type: SampleType,
}

#[pymethods]
impl AudioFileWriterPy {
    /// 创建文件写端：
    ///
    /// - wav: 写 PCM16LE wav（内部可接受 f32/i16 等并转换）
    /// - mp3/aac/flac/opus_ogg: 依赖 FFmpeg backend（feature=ffmpeg）
    #[new]
    #[pyo3(signature = (path, format, input_format, bitrate=None, compression_level=None))]
    fn new(path: String, format: &str, input_format: AudioFormat, bitrate: Option<u32>, compression_level: Option<i32>) -> PyResult<Self> {
        let fmt = file_format_from_str(format).ok_or_else(|| PyValueError::new_err("format 仅支持: wav/mp3/aac_adts/flac/opus_ogg"))?;
        let rs_fmt = input_format.to_rs()?;
        let st = input_format.sample_type_rs()?;

        let cfg = match fmt {
            "wav" => {
                let ch = rs_fmt.channels();
                AudioFileWriteConfig::Wav(rs_file::WavWriterConfig::pcm16le(rs_fmt.sample_rate, ch))
            }
            "mp3" => {
                let mut c = rs_file::Mp3WriterConfig::new(rs_fmt);
                if let Some(br) = bitrate {
                    c.encoder.bitrate = Some(br);
                }
                AudioFileWriteConfig::Mp3(c)
            }
            "aac_adts" => {
                let c = AacEncoderConfig { input_format: rs_fmt, bitrate };
                AudioFileWriteConfig::AacAdts(c)
            }
            "flac" => {
                let c = rs_file::FlacWriterConfig { input_format: rs_fmt, compression_level };
                AudioFileWriteConfig::Flac(c)
            }
            "opus_ogg" => {
                // Opus Ogg writer 目前要求：48k + packed/interleaved
                if rs_fmt.sample_rate != 48_000 {
                    return Err(PyValueError::new_err("opus_ogg writer 需要 48kHz input_format（请先重采样）"));
                }
                if rs_fmt.sample_format.is_planar() {
                    return Err(PyValueError::new_err("opus_ogg writer 需要 interleaved samples（input_format.planar=False）"));
                }
                let c = OpusEncoderConfig { input_format: rs_fmt, bitrate };
                AudioFileWriteConfig::OpusOgg(c)
            }
            _ => return Err(PyValueError::new_err("unsupported format")),
        };

        let w = RsAudioFileWriter::create(path, cfg).map_err(map_file_err)?;
        Ok(Self {
            w,
            input_format: rs_fmt,
            sample_type: st,
        })
    }

    /// 直接写入一帧 PCM（numpy）：
    /// - input_format.planar=True  => shape=(channels,samples)
    /// - input_format.planar=False => shape=(samples,channels)
    fn write_pcm(&mut self, py: Python<'_>, pcm: &Bound<'_, PyAny>) -> PyResult<()> {
        let dtype_name = match self.sample_type {
            SampleType::U8 => "uint8",
            SampleType::I16 => "int16",
            SampleType::I32 => "int32",
            SampleType::I64 => "int64",
            SampleType::F32 => "float32",
            SampleType::F64 => "float64",
        };
        let arr_any = ascontig_cast_2d(py, pcm, dtype_name)?;
        let frame = if self.input_format.is_planar() {
            match self.sample_type {
                SampleType::U8 => ndarray_to_frame_planar::<u8>(&arr_any, self.input_format)?,
                SampleType::I16 => ndarray_to_frame_planar::<i16>(&arr_any, self.input_format)?,
                SampleType::I32 => ndarray_to_frame_planar::<i32>(&arr_any, self.input_format)?,
                SampleType::I64 => ndarray_to_frame_planar::<i64>(&arr_any, self.input_format)?,
                SampleType::F32 => ndarray_to_frame_planar::<f32>(&arr_any, self.input_format)?,
                SampleType::F64 => ndarray_to_frame_planar::<f64>(&arr_any, self.input_format)?,
            }
        } else {
            match self.sample_type {
                SampleType::U8 => ndarray_to_frame_interleaved::<u8>(&arr_any, self.input_format)?,
                SampleType::I16 => ndarray_to_frame_interleaved::<i16>(&arr_any, self.input_format)?,
                SampleType::I32 => ndarray_to_frame_interleaved::<i32>(&arr_any, self.input_format)?,
                SampleType::I64 => ndarray_to_frame_interleaved::<i64>(&arr_any, self.input_format)?,
                SampleType::F32 => ndarray_to_frame_interleaved::<f32>(&arr_any, self.input_format)?,
                SampleType::F64 => ndarray_to_frame_interleaved::<f64>(&arr_any, self.input_format)?,
            }
        };
        AudioWriter::write_frame(&mut self.w, &frame as &dyn AudioFrameView).map_err(map_file_err)?;
        Ok(())
    }

    /// `AsyncDynRunner` 兼容：push(buf: NodeBuffer)（仅支持 PCM）。
    fn push(&mut self, py: Python<'_>, buf: Py<NodeBufferPy>) -> PyResult<()> {
        let mut b = buf.bind(py).borrow_mut();
        let inner = b.take_inner()?;
        match inner {
            NodeBuffer::Pcm(f) => {
                AudioWriter::write_frame(&mut self.w, &f as &dyn AudioFrameView).map_err(map_file_err)?;
                Ok(())
            }
            NodeBuffer::Packet(_) => Err(PyValueError::new_err("AudioFileWriter.push 仅支持 PCM（NodeBuffer kind=pcm）")),
        }
    }

    fn finalize(&mut self) -> PyResult<()> {
        AudioWriter::finalize(&mut self.w).map_err(map_file_err)
    }
}

#[pyclass]
pub struct Encoder {
    codec: String,
    chunk_samples: usize,
    input_format: RsAudioFormat,
    sample_type: SampleType,
    fifo: AudioFifo,
    enc: Box<dyn AudioEncoder>,
    out_q: VecDeque<CodecPacket>,
    sent_eof: bool,
}

#[pymethods]
impl Encoder {
    /// 创建编码器：
    /// - codec: "wav" | "mp3" | "aac" | "opus" | "flac"
    /// - config: 对应的 *EncoderConfigPy
    #[new]
    fn new(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let codec_norm = codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac/opus/flac"))?;

        match codec_norm {
            "wav" => {
                let cfg = config.extract::<WavEncoderConfigPy>()?;
                let input_format = cfg.input_format.to_rs()?;
                let sample_type = cfg.input_format.sample_type_rs()?;
                let fifo = AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32)).map_err(map_audio_err)?;
                let enc = Box::new(WavEncoder::new(WavEncoderConfig { input_format }).map_err(map_codec_err)?) as Box<dyn AudioEncoder>;
                Ok(Self {
                    codec: "wav".into(),
                    chunk_samples: cfg.chunk_samples,
                    input_format,
                    sample_type,
                    fifo,
                    enc,
                    out_q: VecDeque::new(),
                    sent_eof: false,
                })
            }
            "mp3" => {
                let cfg = config.extract::<Mp3EncoderConfigPy>()?;
                let input_format = cfg.input_format.to_rs()?;
                let sample_type = cfg.input_format.sample_type_rs()?;
                let fifo = AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32)).map_err(map_audio_err)?;
                let enc_cfg = Mp3EncoderConfig {
                    input_format,
                    bitrate: cfg.bitrate,
                };
                let enc = Box::new(Mp3Encoder::new(enc_cfg).map_err(map_codec_err)?) as Box<dyn AudioEncoder>;
                Ok(Self {
                    codec: "mp3".into(),
                    chunk_samples: cfg.chunk_samples,
                    input_format,
                    sample_type,
                    fifo,
                    enc,
                    out_q: VecDeque::new(),
                    sent_eof: false,
                })
            }
            "aac" => {
                let cfg = config.extract::<AacEncoderConfigPy>()?;
                let input_format = cfg.input_format.to_rs()?;
                let sample_type = cfg.input_format.sample_type_rs()?;
                let fifo = AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32)).map_err(map_audio_err)?;
                let enc_cfg = AacEncoderConfig {
                    input_format,
                    bitrate: cfg.bitrate,
                };
                let enc = Box::new(AacEncoder::new(enc_cfg).map_err(map_codec_err)?) as Box<dyn AudioEncoder>;
                Ok(Self {
                    codec: "aac".into(),
                    chunk_samples: cfg.chunk_samples,
                    input_format,
                    sample_type,
                    fifo,
                    enc,
                    out_q: VecDeque::new(),
                    sent_eof: false,
                })
            }
            "opus" => {
                let cfg = config.extract::<OpusEncoderConfigPy>()?;
                let input_format = cfg.input_format.to_rs()?;
                let sample_type = cfg.input_format.sample_type_rs()?;
                let fifo = AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32)).map_err(map_audio_err)?;
                let enc_cfg = OpusEncoderConfig {
                    input_format,
                    bitrate: cfg.bitrate,
                };
                let enc = Box::new(OpusEncoder::new(enc_cfg).map_err(map_codec_err)?) as Box<dyn AudioEncoder>;
                Ok(Self {
                    codec: "opus".into(),
                    chunk_samples: cfg.chunk_samples,
                    input_format,
                    sample_type,
                    fifo,
                    enc,
                    out_q: VecDeque::new(),
                    sent_eof: false,
                })
            }
            "flac" => {
                let cfg = config.extract::<FlacEncoderConfigPy>()?;
                let input_format = cfg.input_format.to_rs()?;
                let sample_type = cfg.input_format.sample_type_rs()?;
                let fifo = AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32)).map_err(map_audio_err)?;
                let enc_cfg = FlacEncoderConfig {
                    input_format,
                    compression_level: cfg.compression_level,
                };
                let enc = Box::new(FlacEncoder::new(enc_cfg).map_err(map_codec_err)?) as Box<dyn AudioEncoder>;
                Ok(Self {
                    codec: "flac".into(),
                    chunk_samples: cfg.chunk_samples,
                    input_format,
                    sample_type,
                    fifo,
                    enc,
                    out_q: VecDeque::new(),
                    sent_eof: false,
                })
            }
            _ => Err(PyValueError::new_err("unsupported codec")),
        }
    }

    /// 输入一段 PCM：
    /// - input_format.planar=True  => numpy shape=(channels, samples)
    /// - input_format.planar=False => numpy shape=(samples, channels)
    ///
    /// 注意：这里不会立刻保证产生输出；需要 get_frame() 去取。
    fn put_frame(&mut self, py: Python<'_>, pcm: &Bound<'_, PyAny>) -> PyResult<()> {
        let dtype_name = match self.sample_type {
            SampleType::U8 => "uint8",
            SampleType::I16 => "int16",
            SampleType::I32 => "int32",
            SampleType::I64 => "int64",
            SampleType::F32 => "float32",
            SampleType::F64 => "float64",
        };
        let arr_any = ascontig_cast_2d(py, pcm, dtype_name)?;

        let frame = if self.input_format.is_planar() {
            match self.sample_type {
                SampleType::U8 => ndarray_to_frame_planar::<u8>(&arr_any, self.input_format)?,
                SampleType::I16 => ndarray_to_frame_planar::<i16>(&arr_any, self.input_format)?,
                SampleType::I32 => ndarray_to_frame_planar::<i32>(&arr_any, self.input_format)?,
                SampleType::I64 => ndarray_to_frame_planar::<i64>(&arr_any, self.input_format)?,
                SampleType::F32 => ndarray_to_frame_planar::<f32>(&arr_any, self.input_format)?,
                SampleType::F64 => ndarray_to_frame_planar::<f64>(&arr_any, self.input_format)?,
            }
        } else {
            match self.sample_type {
                SampleType::U8 => ndarray_to_frame_interleaved::<u8>(&arr_any, self.input_format)?,
                SampleType::I16 => ndarray_to_frame_interleaved::<i16>(&arr_any, self.input_format)?,
                SampleType::I32 => ndarray_to_frame_interleaved::<i32>(&arr_any, self.input_format)?,
                SampleType::I64 => ndarray_to_frame_interleaved::<i64>(&arr_any, self.input_format)?,
                SampleType::F32 => ndarray_to_frame_interleaved::<f32>(&arr_any, self.input_format)?,
                SampleType::F64 => ndarray_to_frame_interleaved::<f64>(&arr_any, self.input_format)?,
            }
        };
        self.fifo.push_frame(&frame).map_err(map_audio_err)?;
        Ok(())
    }

    /// 取出一个编码后的 frame（bytes）。
    ///
    /// - 默认：如果 FIFO 剩余不够一个 chunk，则返回 None 并 warnings.warn
    /// - force=True：强制把最后不足一个 chunk 的残留也作为最后一帧输出（如果 codec 支持可变帧长）
    #[pyo3(signature = (force=false))]
    fn get_frame(&mut self, py: Python<'_>, force: bool) -> PyResult<Option<Py<PyBytes>>> {
        if self.out_q.is_empty() {
            push_encoder_from_fifo(self.enc.as_mut(), &mut self.fifo, self.chunk_samples, &mut self.out_q, force)
                .map_err(map_codec_err)?;
        }

        if force && self.out_q.is_empty() && self.fifo.available_samples() == 0 && !self.sent_eof {
            self.enc.send_frame(None).map_err(map_codec_err)?;
            loop {
                match self.enc.receive_packet() {
                    Ok(pkt) => self.out_q.push_back(pkt),
                    Err(CodecError::Again) => break,
                    Err(CodecError::Eof) => break,
                    Err(e) => return Err(map_codec_err(e)),
                }
            }
            self.sent_eof = true;
        }

        if let Some(pkt) = self.out_q.pop_front() {
            Ok(Some(PyBytes::new_bound(py, &pkt.data).unbind()))
        } else {
            let left = self.fifo.available_samples();
            if left > 0 && left < self.chunk_samples {
                warnings_warn(py, "Not enough for one chunk: the last frame is incomplete and will not be returned by default; to return it, please call get_frame(force=True)")?;
            }
            Ok(None)
        }
    }

    /// 取出一个编码后的 packet（带 time_base/pts/dts/duration/flags）。
    #[pyo3(signature = (force=false))]
    fn get_packet(&mut self, _py: Python<'_>, force: bool) -> PyResult<Option<PacketPy>> {
        if self.out_q.is_empty() {
            push_encoder_from_fifo(self.enc.as_mut(), &mut self.fifo, self.chunk_samples, &mut self.out_q, force)
                .map_err(map_codec_err)?;
        }

        if force && self.out_q.is_empty() && self.fifo.available_samples() == 0 && !self.sent_eof {
            self.enc.send_frame(None).map_err(map_codec_err)?;
            loop {
                match self.enc.receive_packet() {
                    Ok(pkt) => self.out_q.push_back(pkt),
                    Err(CodecError::Again) => break,
                    Err(CodecError::Eof) => break,
                    Err(e) => return Err(map_codec_err(e)),
                }
            }
            self.sent_eof = true;
        }

        if let Some(pkt) = self.out_q.pop_front() {
            Ok(Some(PacketPy::from_rs(&pkt)))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn codec(&self) -> &str {
        &self.codec
    }

    #[getter]
    fn chunk_samples(&self) -> usize {
        self.chunk_samples
    }

    /// 当前 FIFO 内累计的 PCM 样本数（每声道）。
    ///
    /// - 0：已空
    /// - 0 < x < chunk_samples：最后一个 chunk 未满
    /// - x >= chunk_samples：至少可产出一帧（配合 get_frame()）
    fn pending_samples(&self) -> usize {
        self.fifo.available_samples()
    }

    /// 当前状态：
    /// - "ready": 有可取的编码帧（out_q 非空）或 FIFO 已攒够一个 chunk
    /// - "need_more": FIFO 有残留但不足一个 chunk
    /// - "empty": out_q 为空且 FIFO 也为空
    fn state(&self) -> &'static str {
        if !self.out_q.is_empty() {
            return "ready";
        }
        let left = self.fifo.available_samples();
        if left == 0 {
            "empty"
        } else if left < self.chunk_samples {
            "need_more"
        } else {
            "ready"
        }
    }
}

#[pyclass]
pub struct Decoder {
    codec: String,
    chunk_samples: usize,
    packet_time_base: Rational,
    dec: Box<dyn AudioDecoder>,
    fifo: Option<AudioFifo>, // 输出 FIFO（planar）
}

/// Python 侧 Processor（PCM->PCM）：目前包含 IdentityProcessor / ResampleProcessor。
#[pyclass(name = "Processor")]
pub struct ProcessorPy {
    p: Box<dyn AudioProcessor>,
    in_format: Option<RsAudioFormat>,
    out_format: Option<RsAudioFormat>,
    // 若 input_format 已知，则在 numpy<->frame 时可用来选择 dtype
    in_sample_type: Option<SampleType>,
}

#[pymethods]
impl ProcessorPy {
    /// 创建 identity processor：
    /// - format=None：不做格式约束
    /// - format=AudioFormat：严格要求输入匹配
    #[staticmethod]
    #[pyo3(signature = (format=None))]
    fn identity(format: Option<AudioFormat>) -> PyResult<Self> {
        let p: Box<dyn AudioProcessor> = if let Some(fmt) = format {
            Box::new(IdentityProcessor::new_with_format(fmt.to_rs()?))
        } else {
            Box::new(IdentityProcessor::new().map_err(map_codec_err)?)
        };
        let in_format = p.input_format();
        let out_format = p.output_format();
        let in_sample_type = in_format.map(|f| f.sample_format.sample_type());
        Ok(Self {
            p,
            in_format,
            out_format,
            in_sample_type,
        })
    }

    /// 创建 resample processor（自动选择后端：ffmpeg 优先，否则 linear）。
    ///
    /// 说明：
    /// - in_format/out_format 必须完整匹配（channels/planar/sample_type 等）
    /// - out_chunk_samples/pad_final 可用于“重分帧”（典型：Opus 前 960@48k）
    #[staticmethod]
    #[pyo3(signature = (in_format, out_format, out_chunk_samples=None, pad_final=true))]
    fn resample(in_format: AudioFormat, out_format: AudioFormat, out_chunk_samples: Option<usize>, pad_final: bool) -> PyResult<Self> {
        let in_rs = in_format.to_rs()?;
        let out_rs = out_format.to_rs()?;
        let mut p = ResampleProcessor::new(in_rs, out_rs).map_err(map_codec_err)?;
        p.set_output_chunker(out_chunk_samples, pad_final).map_err(map_codec_err)?;
        let in_format = p.input_format();
        let out_format = p.output_format();
        let in_sample_type = in_format.map(|f| f.sample_format.sample_type());
        Ok(Self {
            p: Box::new(p),
            in_format,
            out_format,
            in_sample_type,
        })
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.p.name()
    }

    /// 输入一帧 PCM（numpy）：
    /// - 若 processor.input_format() 已知，则要求 numpy layout 与 format.planar 一致：
    ///   - planar=True  => (channels, samples)
    ///   - planar=False => (samples, channels)
    /// - 若 input_format=None：暂不支持（需要你显式给 identity(format=...) 或 resample(in_format=...)）
    #[pyo3(signature = (pcm, pts=None))]
    fn put_frame(&mut self, py: Python<'_>, pcm: &Bound<'_, PyAny>, pts: Option<i64>) -> PyResult<()> {
        let in_fmt = self
            .in_format
            .ok_or_else(|| PyValueError::new_err("Processor 输入格式未知：请用 Processor.identity(format=...) 或 Processor.resample(in_format=...)"))?;
        let st = self.in_sample_type.ok_or_else(|| PyRuntimeError::new_err("missing input sample type"))?;
        let dtype_name = match st {
            SampleType::U8 => "uint8",
            SampleType::I16 => "int16",
            SampleType::I32 => "int32",
            SampleType::I64 => "int64",
            SampleType::F32 => "float32",
            SampleType::F64 => "float64",
        };
        let arr_any = ascontig_cast_2d(py, pcm, dtype_name)?;
        let mut frame = if in_fmt.is_planar() {
            match st {
                SampleType::U8 => ndarray_to_frame_planar::<u8>(&arr_any, in_fmt)?,
                SampleType::I16 => ndarray_to_frame_planar::<i16>(&arr_any, in_fmt)?,
                SampleType::I32 => ndarray_to_frame_planar::<i32>(&arr_any, in_fmt)?,
                SampleType::I64 => ndarray_to_frame_planar::<i64>(&arr_any, in_fmt)?,
                SampleType::F32 => ndarray_to_frame_planar::<f32>(&arr_any, in_fmt)?,
                SampleType::F64 => ndarray_to_frame_planar::<f64>(&arr_any, in_fmt)?,
            }
        } else {
            match st {
                SampleType::U8 => ndarray_to_frame_interleaved::<u8>(&arr_any, in_fmt)?,
                SampleType::I16 => ndarray_to_frame_interleaved::<i16>(&arr_any, in_fmt)?,
                SampleType::I32 => ndarray_to_frame_interleaved::<i32>(&arr_any, in_fmt)?,
                SampleType::I64 => ndarray_to_frame_interleaved::<i64>(&arr_any, in_fmt)?,
                SampleType::F32 => ndarray_to_frame_interleaved::<f32>(&arr_any, in_fmt)?,
                SampleType::F64 => ndarray_to_frame_interleaved::<f64>(&arr_any, in_fmt)?,
            }
        };
        if let Some(p) = pts {
            frame.set_pts(Some(p));
        }
        self.p.send_frame(Some(&frame as &dyn AudioFrameView)).map_err(map_codec_err)?;
        Ok(())
    }

    /// flush：通知输入结束（之后继续 get_frame 直到 EOF）。
    fn flush(&mut self) -> PyResult<()> {
        self.p.send_frame(None).map_err(map_codec_err)
    }

    /// 取出一帧输出 PCM：
    /// - 返回 numpy；无输出返回 None
    #[pyo3(signature = (layout="planar"))]
    fn get_frame(&mut self, py: Python<'_>, layout: &str) -> PyResult<Option<PyObject>> {
        let planar = match layout.to_ascii_lowercase().as_str() {
            "planar" => true,
            "interleaved" => false,
            _ => return Err(PyValueError::new_err("layout 仅支持: planar/interleaved")),
        };
        match self.p.receive_frame() {
            Ok(f) => Ok(Some(frame_to_numpy(py, &f, planar)?)),
            Err(CodecError::Again) => Ok(None),
            Err(CodecError::Eof) => Ok(None),
            Err(e) => Err(map_codec_err(e)),
        }
    }

    /// 输出格式（若已知）。
    fn output_format(&self) -> Option<AudioFormat> {
        self.out_format.map(audio_format_from_rs)
    }
}

#[pymethods]
impl Decoder {
    /// 创建解码器：
    /// - codec: "wav" | "mp3" | "aac" | "opus" | "flac"
    /// - config: 对应的 *DecoderConfigPy
    #[new]
    fn new(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let codec_norm = codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac/opus/flac"))?;

        match codec_norm {
            "wav" => {
                let cfg = config.extract::<WavDecoderConfigPy>()?;
                let out_fmt = cfg.output_format.to_rs()?;
                if out_fmt.is_planar() {
                    return Err(PyValueError::new_err("WAV/PCM decoder 的 output_format.planar 必须为 False（packet 为 interleaved bytes）"));
                }
                let packet_time_base = Rational::new(1, out_fmt.sample_rate as i32);
                let dec = Box::new(WavDecoder::new(out_fmt).map_err(map_codec_err)?) as Box<dyn AudioDecoder>;
                Ok(Self {
                    codec: "wav".into(),
                    chunk_samples: cfg.chunk_samples,
                    packet_time_base,
                    dec,
                    fifo: Some(AudioFifo::new(
                        RsAudioFormat {
                            sample_rate: out_fmt.sample_rate,
                            sample_format: sample_format_from_type(out_fmt.sample_format.sample_type(), true),
                            channel_layout: out_fmt.channel_layout,
                        },
                        packet_time_base,
                    ).map_err(map_audio_err)?),
                })
            }
            "mp3" => {
                let cfg = config.extract::<Mp3DecoderConfigPy>()?;
                let packet_time_base = packet_time_base_from_den(cfg.packet_time_base_den);
                let dec = Box::new(Mp3Decoder::new().map_err(map_codec_err)?) as Box<dyn AudioDecoder>;
                Ok(Self {
                    codec: "mp3".into(),
                    chunk_samples: cfg.chunk_samples,
                    packet_time_base,
                    dec,
                    fifo: None,
                })
            }
            "aac" => {
                let cfg = config.extract::<AacDecoderConfigPy>()?;
                let packet_time_base = packet_time_base_from_den(cfg.packet_time_base_den);
                let dec = Box::new(AacDecoder::new().map_err(map_codec_err)?) as Box<dyn AudioDecoder>;
                Ok(Self {
                    codec: "aac".into(),
                    chunk_samples: cfg.chunk_samples,
                    packet_time_base,
                    dec,
                    fifo: None,
                })
            }
            "opus" => {
                let cfg = config.extract::<OpusDecoderConfigPy>()?;
                let packet_time_base = packet_time_base_from_den(cfg.packet_time_base_den);
                let dec: Box<dyn AudioDecoder> = if let Some(ed) = cfg.extradata.as_ref() {
                    Box::new(OpusDecoder::new_with_extradata(ed).map_err(map_codec_err)?)
                } else {
                    Box::new(OpusDecoder::new().map_err(map_codec_err)?)
                };
                Ok(Self {
                    codec: "opus".into(),
                    chunk_samples: cfg.chunk_samples,
                    packet_time_base,
                    dec,
                    fifo: None,
                })
            }
            "flac" => {
                let cfg = config.extract::<FlacDecoderConfigPy>()?;
                let packet_time_base = packet_time_base_from_den(cfg.packet_time_base_den);
                let dec = Box::new(FlacDecoder::new().map_err(map_codec_err)?) as Box<dyn AudioDecoder>;
                Ok(Self {
                    codec: "flac".into(),
                    chunk_samples: cfg.chunk_samples,
                    packet_time_base,
                    dec,
                    fifo: None,
                })
            }
            _ => Err(PyValueError::new_err("unsupported codec")),
        }
    }

    /// 输入一个编码 frame（bytes）。
    fn put_frame(&mut self, _py: Python<'_>, frame: &Bound<'_, PyAny>) -> PyResult<()> {
        let b: Vec<u8> = frame.extract().map_err(|_| PyTypeError::new_err("frame 需要是 bytes"))?;

        let pkt = CodecPacket {
            data: b,
            time_base: self.packet_time_base,
            pts: None,
            dts: None,
            duration: None,
            flags: PacketFlags::empty(),
        };

        self.dec.send_packet(Some(pkt)).map_err(map_codec_err)?;

        loop {
            match self.dec.receive_frame() {
                Ok(f) => {
                    let f = ensure_planar_frame(f).map_err(map_audio_err)?;
                    let fmt = *f.format_ref();
                    let fifo = if let Some(existing) = &mut self.fifo {
                        // 校验 format 一致（若不一致则报错）
                        if existing.format() != fmt {
                            return Err(PyRuntimeError::new_err("decoder output format changed unexpectedly"));
                        }
                        existing
                    } else {
                        self.fifo = Some(AudioFifo::new(fmt, f.time_base()).map_err(map_audio_err)?);
                        self.fifo.as_mut().unwrap()
                    };
                    fifo.push_frame(&f).map_err(map_audio_err)?;
                }
                Err(CodecError::Again) => break,
                Err(e) => return Err(map_codec_err(e)),
            }
        }
        Ok(())
    }

    /// 输入一个 packet（支持 pts/dts/duration/time_base/flags）。
    fn put_packet(&mut self, _py: Python<'_>, pkt: PacketPy) -> PyResult<()> {
        let pkt = pkt.to_rs();
        self.dec.send_packet(Some(pkt)).map_err(map_codec_err)?;

        loop {
            match self.dec.receive_frame() {
                Ok(f) => {
                    let f = ensure_planar_frame(f).map_err(map_audio_err)?;
                    let fmt = *f.format_ref();
                    let fifo = if let Some(existing) = &mut self.fifo {
                        // 校验 format 一致（若不一致则报错）
                        if existing.format() != fmt {
                            return Err(PyRuntimeError::new_err("decoder output format changed unexpectedly"));
                        }
                        existing
                    } else {
                        self.fifo = Some(AudioFifo::new(fmt, f.time_base()).map_err(map_audio_err)?);
                        self.fifo.as_mut().unwrap()
                    };
                    fifo.push_frame(&f).map_err(map_audio_err)?;
                }
                Err(CodecError::Again) => break,
                Err(e) => return Err(map_codec_err(e)),
            }
        }
        Ok(())
    }

    /// 取出一帧 PCM：numpy ndarray，shape=(channels, samples)
    ///
    /// - 默认：如果 FIFO 剩余不够一个 chunk，则返回 None 并 warnings.warn
    /// - force=True：强制把最后不足一个 chunk 的残留也返回（作为最后一帧）
    #[pyo3(signature = (force=false, layout="planar"))]
    fn get_frame(&mut self, py: Python<'_>, force: bool, layout: &str) -> PyResult<Option<PyObject>> {
        let planar = match layout.to_ascii_lowercase().as_str() {
            "planar" => true,
            "interleaved" => false,
            _ => return Err(PyValueError::new_err("layout 仅支持: planar/interleaved")),
        };
        let Some(fifo) = &mut self.fifo else {
            // 还没解出任何帧
            return Ok(None);
        };

        let nb = fifo.available_samples();
        if nb >= self.chunk_samples {
            let f = fifo
                .pop_frame(self.chunk_samples)
                .map_err(map_audio_err)?
                .expect("available checked");
            return Ok(Some(frame_to_numpy(py, &f, planar)?));
        }

        if !force {
            if nb > 0 {
                warnings_warn(py, "不够一个 chunk：最后一帧未满，默认不返回；如需返回请 get_frame(force=True)")?;
            }
            return Ok(None);
        }

        if nb == 0 {
            return Ok(None);
        }
        let f = fifo.pop_frame(nb).map_err(map_audio_err)?.expect("nb>0");
        Ok(Some(frame_to_numpy(py, &f, planar)?))
    }

    /// 取出一帧 PCM + 元信息：返回 (numpy, pts, (time_base_num,time_base_den))
    #[pyo3(signature = (force=false, layout="planar"))]
    fn get_frame_info(&mut self, py: Python<'_>, force: bool, layout: &str) -> PyResult<Option<(PyObject, Option<i64>, (i32, i32))>> {
        let planar = match layout.to_ascii_lowercase().as_str() {
            "planar" => true,
            "interleaved" => false,
            _ => return Err(PyValueError::new_err("layout 仅支持: planar/interleaved")),
        };
        let Some(fifo) = &mut self.fifo else {
            return Ok(None);
        };

        let nb = fifo.available_samples();
        let take = if nb >= self.chunk_samples {
            self.chunk_samples
        } else if force {
            nb
        } else {
            if nb > 0 {
                warnings_warn(py, "不够一个 chunk：最后一帧未满，默认不返回；如需返回请 get_frame(force=True)")?;
            }
            return Ok(None);
        };
        if take == 0 {
            return Ok(None);
        }

        let f = fifo.pop_frame(take).map_err(map_audio_err)?.expect("take>0");
        let pts = f.pts();
        let tb = f.time_base();
        let arr = frame_to_numpy(py, &f, planar)?;
        Ok(Some((arr, pts, (tb.num, tb.den))))
    }

    #[getter]
    fn codec(&self) -> &str {
        &self.codec
    }

    #[getter]
    fn chunk_samples(&self) -> usize {
        self.chunk_samples
    }

    /// 当前输出 FIFO 内累计的 PCM 样本数（每声道）。
    fn pending_samples(&self) -> usize {
        self.fifo.as_ref().map(|f| f.available_samples()).unwrap_or(0)
    }

    /// 当前状态：
    /// - "ready": FIFO 已攒够一个 chunk
    /// - "need_more": FIFO 有残留但不足一个 chunk
    /// - "empty": 还没解出任何 PCM 或已完全取空
    fn state(&self) -> &'static str {
        let Some(fifo) = self.fifo.as_ref() else {
            return "empty";
        };
        let left = fifo.available_samples();
        if left == 0 {
            "empty"
        } else if left < self.chunk_samples {
            "need_more"
        } else {
            "ready"
        }
    }
}

#[pymodule]
fn pyaudiostream(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AudioFormat>()?;
    m.add_class::<WavEncoderConfigPy>()?;
    m.add_class::<WavDecoderConfigPy>()?;
    m.add_class::<Mp3EncoderConfigPy>()?;
    m.add_class::<AacEncoderConfigPy>()?;
    m.add_class::<Mp3DecoderConfigPy>()?;
    m.add_class::<AacDecoderConfigPy>()?;
    m.add_class::<OpusEncoderConfigPy>()?;
    m.add_class::<FlacEncoderConfigPy>()?;
    m.add_class::<OpusDecoderConfigPy>()?;
    m.add_class::<FlacDecoderConfigPy>()?;
    m.add_class::<Encoder>()?;
    m.add_class::<Decoder>()?;
    m.add_class::<PacketPy>()?;
    m.add_class::<NodeBufferPy>()?;
    m.add_class::<ProcessorPy>()?;
    m.add_class::<DynNodePy>()?;
    m.add_class::<AsyncDynPipelinePy>()?;
    m.add_class::<AsyncDynRunnerPy>()?;
    m.add_class::<AudioFileReaderPy>()?;
    m.add_class::<AudioFileWriterPy>()?;
    m.add_function(wrap_pyfunction!(make_identity_node, m)?)?;
    m.add_function(wrap_pyfunction!(make_resample_node, m)?)?;
    m.add_function(wrap_pyfunction!(make_encoder_node, m)?)?;
    m.add_function(wrap_pyfunction!(make_decoder_node, m)?)?;
    Ok(())
}


