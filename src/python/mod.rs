//! Python bindings (PyO3) for streaming encoder/decoder.
//!
//! 设计目标（对应需求）：
//! - 暴露 Encoder/Decoder
//! - put_frame / get_frame：内部用 AudioFifo 做 chunk 聚合；最后不满一个 chunk 默认不返回，
//!   但 get_frame(force=True) 可强制 flush 剩余样本。

#![allow(unsafe_op_in_unsafe_fn)]

use crate::codec::decoder::aac_decoder::AacDecoder;
use crate::codec::decoder::decoder_interface::AudioDecoder;
use crate::codec::decoder::mp3_decoder::Mp3Decoder;
use crate::codec::decoder::wav_decoder::WavDecoder;
use crate::codec::encoder::aac_encoder::{AacEncoder, AacEncoderConfig};
use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::encoder::mp3_encoder::{Mp3Encoder, Mp3EncoderConfig};
use crate::codec::encoder::wav_encoder::{WavEncoder, WavEncoderConfig};
use crate::codec::error::CodecError;
use crate::codec::packet::{CodecPacket, PacketFlags};
use crate::common::audio::audio::{
    AudioError, AudioFormat as RsAudioFormat, AudioFrameView, ChannelLayout, Rational, SampleFormat, SampleType,
};
use crate::common::audio::fifo::AudioFifo;

use numpy::{Element, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

use std::collections::VecDeque;

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

fn packet_time_base_from_den(den: i32) -> Rational {
    Rational::new(1, den)
}

fn push_encoder_from_fifo(enc: &mut dyn AudioEncoder, fifo: &mut AudioFifo, chunk_samples: usize, out_q: &mut VecDeque<Vec<u8>>, force: bool) -> Result<(), CodecError> {
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
                    Ok(pkt) => out_q.push_back(pkt.data),
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
                Ok(pkt) => out_q.push_back(pkt.data),
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

fn frame_to_numpy<'py>(py: Python<'py>, frame: &crate::common::audio::audio::AudioFrame) -> PyResult<PyObject> {
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

#[pyclass]
pub struct Encoder {
    codec: String,
    chunk_samples: usize,
    input_format: RsAudioFormat,
    sample_type: SampleType,
    fifo: AudioFifo,
    enc: Box<dyn AudioEncoder>,
    out_q: VecDeque<Vec<u8>>,
    sent_eof: bool,
}

#[pymethods]
impl Encoder {
    /// 创建编码器：
    /// - codec: "wav" | "mp3" | "aac"
    /// - config: 对应的 *EncoderConfigPy
    #[new]
    fn new(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let codec_norm = codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac"))?;

        match codec_norm {
            "wav" => {
                let cfg = config.extract::<WavEncoderConfigPy>()?;
                if !cfg.input_format.planar {
                    return Err(PyValueError::new_err("Python put_frame 的 numpy shape=(channels,samples)；因此 input_format.planar 必须为 True"));
                }
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
                if !cfg.input_format.planar {
                    return Err(PyValueError::new_err("Python put_frame 的 numpy shape=(channels,samples)；因此 input_format.planar 必须为 True"));
                }
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
                if !cfg.input_format.planar {
                    return Err(PyValueError::new_err("Python put_frame 的 numpy shape=(channels,samples)；因此 input_format.planar 必须为 True"));
                }
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
            _ => Err(PyValueError::new_err("unsupported codec")),
        }
    }

    /// 输入一段 PCM：numpy ndarray，shape=(channels, samples)
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

        // 生成 planar AudioFrame 并 push 进 FIFO
        let frame = match self.sample_type {
            SampleType::U8 => ndarray_to_frame_planar::<u8>(&arr_any, self.input_format)?,
            SampleType::I16 => ndarray_to_frame_planar::<i16>(&arr_any, self.input_format)?,
            SampleType::I32 => ndarray_to_frame_planar::<i32>(&arr_any, self.input_format)?,
            SampleType::I64 => ndarray_to_frame_planar::<i64>(&arr_any, self.input_format)?,
            SampleType::F32 => ndarray_to_frame_planar::<f32>(&arr_any, self.input_format)?,
            SampleType::F64 => ndarray_to_frame_planar::<f64>(&arr_any, self.input_format)?,
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
                    Ok(pkt) => self.out_q.push_back(pkt.data),
                    Err(CodecError::Again) => break,
                    Err(CodecError::Eof) => break,
                    Err(e) => return Err(map_codec_err(e)),
                }
            }
            self.sent_eof = true;
        }

        if let Some(data) = self.out_q.pop_front() {
            Ok(Some(PyBytes::new_bound(py, &data).unbind()))
        } else {
            let left = self.fifo.available_samples();
            if left > 0 && left < self.chunk_samples {
                warnings_warn(py, "Not enough for one chunk: the last frame is incomplete and will not be returned by default; to return it, please call get_frame(force=True)")?;
            }
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

#[pymethods]
impl Decoder {
    /// 创建解码器：
    /// - codec: "wav" | "mp3" | "aac"
    /// - config: 对应的 *DecoderConfigPy
    #[new]
    fn new(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let codec_norm = codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac"))?;

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

    /// 取出一帧 PCM：numpy ndarray，shape=(channels, samples)
    ///
    /// - 默认：如果 FIFO 剩余不够一个 chunk，则返回 None 并 warnings.warn
    /// - force=True：强制把最后不足一个 chunk 的残留也返回（作为最后一帧）
    #[pyo3(signature = (force=false))]
    fn get_frame(&mut self, py: Python<'_>, force: bool) -> PyResult<Option<PyObject>> {
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
            return Ok(Some(frame_to_numpy(py, &f)?));
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
        Ok(Some(frame_to_numpy(py, &f)?))
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
    m.add_class::<Encoder>()?;
    m.add_class::<Decoder>()?;
    Ok(())
}


