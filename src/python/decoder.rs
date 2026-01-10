#![allow(unsafe_op_in_unsafe_fn)]

use crate::codec::decoder::aac_decoder::AacDecoder;
use crate::codec::decoder::decoder_interface::AudioDecoder;
use crate::codec::decoder::flac_decoder::FlacDecoder;
use crate::codec::decoder::mp3_decoder::Mp3Decoder;
use crate::codec::decoder::opus_decoder::OpusDecoder;
use crate::codec::decoder::wav_decoder::WavDecoder;
use crate::codec::error::CodecError;
use crate::codec::packet::{CodecPacket, PacketFlags};
use crate::common::audio::audio::{AudioFormat as RsAudioFormat, AudioFrameView, Rational};
use crate::common::audio::fifo::AudioFifo;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::{NodeBuffer, NodeBufferKind};

use crate::python::errors::{map_audio_err, map_codec_err};
use crate::python::format::{
    audio_format_from_rs, codec_from_str, ensure_planar_frame, frame_to_numpy, packet_time_base_from_den,
    sample_format_from_type, warnings_warn, AudioFormat,
};
use crate::python::io::{DynNodePy, PacketPy};

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
            return Err(PyValueError::new_err("chunk_samples must be greater than 0"));
        }
        Ok(Self {
            output_format,
            chunk_samples,
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
            return Err(PyValueError::new_err("chunk_samples must be greater than 0"));
        }
        if packet_time_base_den <= 0 {
            return Err(PyValueError::new_err("packet_time_base_den must be greater than 0"));
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
            return Err(PyValueError::new_err("chunk_samples must be greater than 0"));
        }
        if packet_time_base_den <= 0 {
            return Err(PyValueError::new_err("packet_time_base_den must be greater than 0"));
        }
        Ok(Self {
            chunk_samples,
            packet_time_base_den,
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
            return Err(PyValueError::new_err("chunk_samples must be greater than 0"));
        }
        if packet_time_base_den <= 0 {
            return Err(PyValueError::new_err("packet_time_base_den must be greater than 0"));
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
            return Err(PyValueError::new_err("chunk_samples must be greater than 0"));
        }
        if packet_time_base_den <= 0 {
            return Err(PyValueError::new_err("packet_time_base_den must be greater than 0"));
        }
        Ok(Self {
            chunk_samples,
            packet_time_base_den,
        })
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
    /// - codec: "wav" | "mp3" | "aac" | "opus" | "flac"
    /// - config: 对应的 *DecoderConfigPy
    #[new]
    fn new(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let codec_norm =
            codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec 仅支持: wav/mp3/aac/opus/flac"))?;

        match codec_norm {
            "wav" => {
                let cfg = config.extract::<WavDecoderConfigPy>()?;
                let out_fmt = cfg.output_format.to_rs()?;
                if out_fmt.is_planar() {
                    return Err(PyValueError::new_err(
                        "WAV/PCM decoder's output_format.planar must be False (packet is interleaved bytes)",
                    ));
                }
                let packet_time_base = Rational::new(1, out_fmt.sample_rate as i32);
                let dec = Box::new(WavDecoder::new(out_fmt).map_err(map_codec_err)?) as Box<dyn AudioDecoder>;
                Ok(Self {
                    codec: "wav".into(),
                    chunk_samples: cfg.chunk_samples,
                    packet_time_base,
                    dec,
                    fifo: Some(
                        AudioFifo::new(
                            RsAudioFormat {
                                sample_rate: out_fmt.sample_rate,
                                sample_format: sample_format_from_type(out_fmt.sample_format.sample_type(), true),
                                channel_layout: out_fmt.channel_layout,
                            },
                            packet_time_base,
                        )
                        .map_err(map_audio_err)?,
                    ),
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
                            return Err(PyRuntimeError::new_err("decoder's output format changed unexpectedly"));
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
                            return Err(PyRuntimeError::new_err("decoder's output format changed unexpectedly"));
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

    /// 重置内部状态（清空缓存、回到初始态），可继续接收新的流。
    fn reset(&mut self) -> PyResult<()> {
        self.dec.reset().map_err(map_codec_err)?;
        self.fifo = None;
        Ok(())
    }

    /// 取出一帧 PCM：numpy ndarray
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
                warnings_warn(py, "not enough samples: last frame is not full, default not return; if you want to return, please get_frame(force=True)")?;
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
    fn get_frame_info(
        &mut self,
        py: Python<'_>,
        force: bool,
        layout: &str,
    ) -> PyResult<Option<(PyObject, Option<i64>, (i32, i32))>> {
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
                warnings_warn(py, "not enough samples: last frame is not full, default not return; if you want to return, please get_frame(force=True)")?;
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

    /// 输出格式（若已知）。
    fn output_format(&self) -> Option<AudioFormat> {
        self.fifo.as_ref().map(|f| audio_format_from_rs(f.format()))
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

/// 创建一个 decoder 节点（Packet->PCM）。注意：此处的 config 里 `chunk_samples` 会被忽略（由下游取帧决定）。
#[pyfunction]
pub fn make_decoder_node(_py: Python<'_>, codec: &str, config: &Bound<'_, PyAny>) -> PyResult<DynNodePy> {
    let codec_norm =
        codec_from_str(codec).ok_or_else(|| PyValueError::new_err("codec only supports: wav, mp3, aac, opus, flac"))?;
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
        _ => return Err(PyValueError::new_err("unsupported codec: only supports wav, mp3, aac, opus, flac")),
    };
    Ok(DynNodePy::new_boxed(Box::new(BoxedDecoderNode { d: dec })))
}


