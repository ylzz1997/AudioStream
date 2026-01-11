#![allow(unsafe_op_in_unsafe_fn)]

use crate::codec::encoder::aac_encoder::{AacEncoder, AacEncoderConfig};
use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::encoder::flac_encoder::{FlacEncoder, FlacEncoderConfig};
use crate::codec::encoder::mp3_encoder::{Mp3Encoder, Mp3EncoderConfig};
use crate::codec::encoder::opus_encoder::{OpusEncoder, OpusEncoderConfig};
use crate::codec::encoder::wav_encoder::{WavEncoder, WavEncoderConfig};
use crate::codec::error::CodecError;
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFrameView, AudioFrameViewMut, Rational, SampleType};
use crate::common::audio::fifo::AudioFifo;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

use std::collections::VecDeque;

use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::{NodeBuffer, NodeBufferKind};

use crate::python::errors::{map_audio_err, map_codec_err};
use crate::python::format::{
    ascontig_cast_2d, codec_from_str, ndarray_to_frame_interleaved, ndarray_to_frame_planar, warnings_warn, AudioFormat,
};
use crate::python::io::{DynNodePy, PacketPy};

#[pyclass(name = "WavEncoderConfig")]
#[derive(Clone)]
pub struct WavEncoderConfigPy {
    #[pyo3(get)]
    pub input_format: Option<AudioFormat>,
    #[pyo3(get)]
    pub chunk_samples: usize,
}

#[pymethods]
impl WavEncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format=None, chunk_samples=1024))]
    fn new(input_format: Option<AudioFormat>, chunk_samples: usize) -> PyResult<Self> {
        if chunk_samples == 0 {
            return Err(PyValueError::new_err("chunk_samples 必须 > 0"));
        }
        Ok(Self {
            input_format,
            chunk_samples,
        })
    }
}

#[pyclass(name = "Mp3EncoderConfig")]
#[derive(Clone)]
pub struct Mp3EncoderConfigPy {
    #[pyo3(get)]
    pub input_format: Option<AudioFormat>,
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub bitrate: Option<u32>,
}

#[pymethods]
impl Mp3EncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format=None, chunk_samples=1152, bitrate=Some(128_000)))]
    fn new(input_format: Option<AudioFormat>, chunk_samples: usize, bitrate: Option<u32>) -> PyResult<Self> {
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
    pub input_format: Option<AudioFormat>,
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub bitrate: Option<u32>,
}

#[pymethods]
impl AacEncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format=None, chunk_samples=1024, bitrate=None))]
    fn new(input_format: Option<AudioFormat>, chunk_samples: usize, bitrate: Option<u32>) -> PyResult<Self> {
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

#[pyclass(name = "OpusEncoderConfig")]
#[derive(Clone)]
pub struct OpusEncoderConfigPy {
    #[pyo3(get)]
    pub input_format: Option<AudioFormat>,
    #[pyo3(get)]
    pub chunk_samples: usize,
    #[pyo3(get)]
    pub bitrate: Option<u32>,
}

#[pymethods]
impl OpusEncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format=None, chunk_samples=960, bitrate=Some(96_000)))]
    fn new(input_format: Option<AudioFormat>, chunk_samples: usize, bitrate: Option<u32>) -> PyResult<Self> {
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
    pub input_format: Option<AudioFormat>,
    #[pyo3(get)]
    pub chunk_samples: usize,
    /// 0..=12（FFmpeg backend 常见语义）；None=默认
    #[pyo3(get)]
    pub compression_level: Option<i32>,
}

#[pymethods]
impl FlacEncoderConfigPy {
    #[new]
    #[pyo3(signature = (input_format=None, chunk_samples=4096, compression_level=None))]
    fn new(input_format: Option<AudioFormat>, chunk_samples: usize, compression_level: Option<i32>) -> PyResult<Self> {
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

fn warn_if_need_format(py: Python<'_>) -> PyResult<()> {
    warnings_warn(
        py,
        "Encoder constructed with input_format=None: first put_frame must provide format=AudioFormat(...)",
    )
}

#[pyclass]
pub struct Encoder {
    codec: String,
    chunk_samples: usize,
    input_format: Option<crate::common::audio::audio::AudioFormat>,
    sample_type: Option<SampleType>,
    fifo: Option<AudioFifo>,
    enc: Box<dyn AudioEncoder>,
    out_q: VecDeque<CodecPacket>,
    sent_eof: bool,
    locked: bool,
}

#[pymethods]
impl Encoder {
    /// 创建编码器：
    /// - codec: "wav" | "mp3" | "aac" | "opus" | "flac"
    /// - config: 对应的 *EncoderConfigPy
    #[new]
    fn new(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let codec_norm =
            codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac/opus/flac"))?;

        match codec_norm {
            "wav" => {
                let cfg = config.extract::<WavEncoderConfigPy>()?;
                let (input_format, sample_type, fifo) = if let Some(py_fmt) = cfg.input_format.clone() {
                    let input_format = py_fmt.to_rs()?;
                    let sample_type = Some(py_fmt.sample_type_rs()?);
                    let fifo = Some(
                        AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32))
                            .map_err(map_audio_err)?,
                    );
                    (Some(input_format), sample_type, fifo)
                } else {
                    (None, None, None)
                };
                let enc = Box::new(WavEncoder::new(WavEncoderConfig { input_format }).map_err(map_codec_err)?)
                    as Box<dyn AudioEncoder>;
                Ok(Self {
                    codec: "wav".into(),
                    chunk_samples: cfg.chunk_samples,
                    input_format,
                    sample_type,
                    fifo,
                    enc,
                    out_q: VecDeque::new(),
                    sent_eof: false,
                    locked: cfg.input_format.is_some(),
                })
            }
            "mp3" => {
                let cfg = config.extract::<Mp3EncoderConfigPy>()?;
                let (input_format, sample_type, fifo) = if let Some(py_fmt) = cfg.input_format.clone() {
                    let input_format = py_fmt.to_rs()?;
                    let sample_type = Some(py_fmt.sample_type_rs()?);
                    let fifo = Some(
                        AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32))
                            .map_err(map_audio_err)?,
                    );
                    (Some(input_format), sample_type, fifo)
                } else {
                    (None, None, None)
                };
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
                    locked: cfg.input_format.is_some(),
                })
            }
            "aac" => {
                let cfg = config.extract::<AacEncoderConfigPy>()?;
                let (input_format, sample_type, fifo) = if let Some(py_fmt) = cfg.input_format.clone() {
                    let input_format = py_fmt.to_rs()?;
                    let sample_type = Some(py_fmt.sample_type_rs()?);
                    let fifo = Some(
                        AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32))
                            .map_err(map_audio_err)?,
                    );
                    (Some(input_format), sample_type, fifo)
                } else {
                    (None, None, None)
                };
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
                    locked: cfg.input_format.is_some(),
                })
            }
            "opus" => {
                let cfg = config.extract::<OpusEncoderConfigPy>()?;
                let (input_format, sample_type, fifo) = if let Some(py_fmt) = cfg.input_format.clone() {
                    let input_format = py_fmt.to_rs()?;
                    let sample_type = Some(py_fmt.sample_type_rs()?);
                    let fifo = Some(
                        AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32))
                            .map_err(map_audio_err)?,
                    );
                    (Some(input_format), sample_type, fifo)
                } else {
                    (None, None, None)
                };
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
                    locked: cfg.input_format.is_some(),
                })
            }
            "flac" => {
                let cfg = config.extract::<FlacEncoderConfigPy>()?;
                let (input_format, sample_type, fifo) = if let Some(py_fmt) = cfg.input_format.clone() {
                    let input_format = py_fmt.to_rs()?;
                    let sample_type = Some(py_fmt.sample_type_rs()?);
                    let fifo = Some(
                        AudioFifo::new(input_format, Rational::new(1, input_format.sample_rate as i32))
                            .map_err(map_audio_err)?,
                    );
                    (Some(input_format), sample_type, fifo)
                } else {
                    (None, None, None)
                };
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
                    locked: cfg.input_format.is_some(),
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
    #[pyo3(signature = (pcm, pts=None, format=None))]
    fn put_frame(
        &mut self,
        py: Python<'_>,
        pcm: &Bound<'_, PyAny>,
        pts: Option<i64>,
        format: Option<AudioFormat>,
    ) -> PyResult<()> {
        let (rs_fmt, sample_type) = if let Some(fmt_py) = format {
            let rs = fmt_py.to_rs()?;
            let st = fmt_py.sample_type_rs()?;
            // 如果已经初始化过格式，则要求一致
            if let Some(expected) = self.input_format {
                if expected != rs {
                    return Err(PyValueError::new_err("format 与 Encoder 当前锁定的 input_format 不一致"));
                }
            }
            (rs, st)
        } else {
            let rs = self
                .input_format
                .ok_or_else(|| PyValueError::new_err("input_format=None 时，首次 put_frame 必须提供 format"))?;
            let st = self
                .sample_type
                .ok_or_else(|| PyValueError::new_err("encoder not initialized (missing sample_type)"))?;
            (rs, st)
        };

        if self.fifo.is_none() {
            self.fifo = Some(
                AudioFifo::new(rs_fmt, Rational::new(1, rs_fmt.sample_rate as i32)).map_err(map_audio_err)?,
            );
            self.input_format = Some(rs_fmt);
            self.sample_type = Some(sample_type);
        } else if let Some(expected) = self.input_format {
            if expected != rs_fmt {
                return Err(PyValueError::new_err("format 与 Encoder 当前锁定的 input_format 不一致"));
            }
        }

        let dtype_name = match sample_type {
            SampleType::U8 => "uint8",
            SampleType::I16 => "int16",
            SampleType::I32 => "int32",
            SampleType::I64 => "int64",
            SampleType::F32 => "float32",
            SampleType::F64 => "float64",
        };
        let arr_any = ascontig_cast_2d(py, pcm, dtype_name)?;

        let mut frame = if rs_fmt.is_planar() {
            match sample_type {
                SampleType::U8 => ndarray_to_frame_planar::<u8>(&arr_any, rs_fmt)?,
                SampleType::I16 => ndarray_to_frame_planar::<i16>(&arr_any, rs_fmt)?,
                SampleType::I32 => ndarray_to_frame_planar::<i32>(&arr_any, rs_fmt)?,
                SampleType::I64 => ndarray_to_frame_planar::<i64>(&arr_any, rs_fmt)?,
                SampleType::F32 => ndarray_to_frame_planar::<f32>(&arr_any, rs_fmt)?,
                SampleType::F64 => ndarray_to_frame_planar::<f64>(&arr_any, rs_fmt)?,
            }
        } else {
            match sample_type {
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
        let fifo = self.fifo.as_mut().ok_or_else(|| PyRuntimeError::new_err("fifo not initialized"))?;
        fifo.push_frame(&frame).map_err(map_audio_err)?;
        Ok(())
    }

    /// 重置内部状态（清空缓存、回到初始态），可继续接收新的流。
    fn reset(&mut self) -> PyResult<()> {
        self.enc.reset().map_err(map_codec_err)?;
        if let Some(f) = self.fifo.as_mut() {
            f.clear();
        }
        // 若是推断模式（构造时未指定 input_format），reset 后回到“未初始化”
        if !self.locked {
            self.input_format = None;
            self.sample_type = None;
            self.fifo = None;
        }
        self.out_q.clear();
        self.sent_eof = false;
        Ok(())
    }

    /// 取出一个编码后的 frame（bytes）。
    ///
    /// - 默认：如果 FIFO 剩余不够一个 chunk，则返回 None 并 warnings.warn
    /// - force=True：强制把最后不足一个 chunk 的残留也作为最后一帧输出（如果 codec 支持可变帧长）
    #[pyo3(signature = (force=false))]
    fn get_frame(&mut self, py: Python<'_>, force: bool) -> PyResult<Option<Py<PyBytes>>> {
        if self.fifo.is_none() && !self.locked && !force && self.out_q.is_empty() {
            // 仅提示一次性 usage：需要 format 才能 put_frame
            warn_if_need_format(py)?;
        }
        if self.out_q.is_empty() {
            if let Some(fifo) = self.fifo.as_mut() {
                push_encoder_from_fifo(
                    self.enc.as_mut(),
                    fifo,
                    self.chunk_samples,
                    &mut self.out_q,
                    force,
                )
                .map_err(map_codec_err)?;
            }
        }

        if force && self.out_q.is_empty() && self.fifo.as_ref().map(|f| f.available_samples()).unwrap_or(0) == 0 && !self.sent_eof {
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
            let left = self.fifo.as_ref().map(|f| f.available_samples()).unwrap_or(0);
            if left > 0 && left < self.chunk_samples {
                warnings_warn(
                    py,
                    "Not enough for one chunk: the last frame is incomplete and will not be returned by default; to return it, please call get_frame(force=True)",
                )?;
            }
            Ok(None)
        }
    }

    /// 取出一个编码后的 packet（带 time_base/pts/dts/duration/flags）。
    #[pyo3(signature = (force=false))]
    fn get_packet(&mut self, py: Python<'_>, force: bool) -> PyResult<Option<PacketPy>> {
        if self.fifo.is_none() && !self.locked && !force && self.out_q.is_empty() {
            warn_if_need_format(py)?;
        }
        if self.out_q.is_empty() {
            if let Some(fifo) = self.fifo.as_mut() {
                push_encoder_from_fifo(
                    self.enc.as_mut(),
                    fifo,
                    self.chunk_samples,
                    &mut self.out_q,
                    force,
                )
                .map_err(map_codec_err)?;
            }
        }

        if force && self.out_q.is_empty() && self.fifo.as_ref().map(|f| f.available_samples()).unwrap_or(0) == 0 && !self.sent_eof {
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
        self.fifo.as_ref().map(|f| f.available_samples()).unwrap_or(0)
    }

    /// 当前状态：
    /// - "ready": 有可取的编码帧（out_q 非空）或 FIFO 已攒够一个 chunk
    /// - "need_more": FIFO 有残留但不足一个 chunk
    /// - "empty": out_q 为空且 FIFO 也为空
    fn state(&self) -> &'static str {
        if !self.out_q.is_empty() {
            return "ready";
        }
        let left = self.fifo.as_ref().map(|f| f.available_samples()).unwrap_or(0);
        if left == 0 {
            "empty"
        } else if left < self.chunk_samples {
            "need_more"
        } else {
            "ready"
        }
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

/// 创建一个 encoder 节点（PCM->Packet）。注意：此处的 config 里 `chunk_samples` 会被忽略（由上游分帧决定）。
#[pyfunction]
pub fn make_encoder_node(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<DynNodePy> {
    let codec_norm =
        codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac/opus/flac"))?;
    let enc: Box<dyn AudioEncoder> = match codec_norm {
        "wav" => {
            let cfg = config.extract::<WavEncoderConfigPy>()?;
            let input_format = match cfg.input_format {
                Some(f) => Some(f.to_rs()?),
                None => None,
            };
            Box::new(WavEncoder::new(WavEncoderConfig { input_format }).map_err(map_codec_err)?)
        }
        "mp3" => {
            let cfg = config.extract::<Mp3EncoderConfigPy>()?;
            let input_format = match cfg.input_format {
                Some(f) => Some(f.to_rs()?),
                None => None,
            };
            Box::new(Mp3Encoder::new(Mp3EncoderConfig {
                input_format,
                bitrate: cfg.bitrate,
            })
            .map_err(map_codec_err)?)
        }
        "aac" => {
            let cfg = config.extract::<AacEncoderConfigPy>()?;
            let input_format = match cfg.input_format {
                Some(f) => Some(f.to_rs()?),
                None => None,
            };
            Box::new(AacEncoder::new(AacEncoderConfig {
                input_format,
                bitrate: cfg.bitrate,
            })
            .map_err(map_codec_err)?)
        }
        "opus" => {
            let cfg = config.extract::<OpusEncoderConfigPy>()?;
            let input_format = match cfg.input_format {
                Some(f) => Some(f.to_rs()?),
                None => None,
            };
            Box::new(OpusEncoder::new(OpusEncoderConfig {
                input_format,
                bitrate: cfg.bitrate,
            })
            .map_err(map_codec_err)?)
        }
        "flac" => {
            let cfg = config.extract::<FlacEncoderConfigPy>()?;
            let input_format = match cfg.input_format {
                Some(f) => Some(f.to_rs()?),
                None => None,
            };
            Box::new(FlacEncoder::new(FlacEncoderConfig {
                input_format,
                compression_level: cfg.compression_level,
            })
            .map_err(map_codec_err)?)
        }
        _ => return Err(PyValueError::new_err("unsupported codec")),
    };
    Ok(DynNodePy::new_boxed(Box::new(BoxedEncoderNode { e: enc })))
}


