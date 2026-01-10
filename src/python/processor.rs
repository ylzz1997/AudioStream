#![allow(unsafe_op_in_unsafe_fn)]

use crate::codec::error::CodecError;
use crate::codec::processor::compressor_processor::CompressorProcessor;
use crate::codec::processor::gain_processor::GainProcessor;
use crate::codec::processor::identity_processor::IdentityProcessor;
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::codec::processor::resample_processor::ResampleProcessor;
use crate::common::audio::audio::{AudioFrameView, AudioFrameViewMut, SampleType};
use crate::function::compressor::DynamicsParams;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::pipeline::node::dynamic_node_interface::ProcessorNode;
use crate::pipeline::node::node_interface::{IdentityNode, NodeBufferKind};

use crate::python::errors::map_codec_err;
use crate::python::format::{
    ascontig_cast_2d, audio_format_from_rs, ndarray_to_frame_interleaved, ndarray_to_frame_planar, frame_to_numpy,
    AudioFormat,
};
use crate::python::io::DynNodePy;

/// Python 侧 Processor（PCM->PCM）：目前包含 IdentityProcessor / ResampleProcessor / GainProcessor / CompressorProcessor。
#[pyclass(name = "Processor")]
pub struct ProcessorPy {
    p: Box<dyn AudioProcessor>,
    in_format: Option<crate::common::audio::audio::AudioFormat>,
    out_format: Option<crate::common::audio::audio::AudioFormat>,
    // 若 input_format 已知，则在 numpy<->frame 时可用来选择 dtype
    in_sample_type: Option<SampleType>,
    // 是否已发送 EOF（flush）
    sent_eof: bool,
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
            sent_eof: false,
        })
    }

    /// 创建 resample processor（自动选择后端：ffmpeg 优先，否则 linear）。
    ///
    /// - in_format/out_format 必须完整匹配（channels/planar/sample_type 等）
    /// - out_chunk_samples/pad_final 可用于“重分帧”
    #[staticmethod]
    #[pyo3(signature = (in_format, out_format, out_chunk_samples=None, pad_final=true))]
    fn resample(
        in_format: AudioFormat,
        out_format: AudioFormat,
        out_chunk_samples: Option<usize>,
        pad_final: bool,
    ) -> PyResult<Self> {
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
            sent_eof: false,
        })
    }

    /// 创建 gain processor（线性增益，PCM->PCM，不改变格式）。
    #[staticmethod]
    #[pyo3(signature = (format, gain))]
    fn gain(format: AudioFormat, gain: f64) -> PyResult<Self> {
        let fmt_rs = format.to_rs()?;
        let p = GainProcessor::new_with_format(fmt_rs, gain).map_err(map_codec_err)?;
        let in_format = p.input_format();
        let out_format = p.output_format();
        let in_sample_type = in_format.map(|f| f.sample_format.sample_type());
        Ok(Self {
            p: Box::new(p),
            in_format,
            out_format,
            in_sample_type,
            sent_eof: false,
        })
    }

    /// 创建 compressor processor（动态压缩 + 扩展/门，PCM->PCM，不改变格式）。
    #[staticmethod]
    #[pyo3(signature = (format, sample_rate, threshold_db, knee_width_db, ratio, expansion_ratio, expansion_threshold_db, attack_time, release_time, master_gain_db))]
    fn compressor(
        format: AudioFormat,
        sample_rate: f32,
        threshold_db: f32,
        knee_width_db: f32,
        ratio: f32,
        expansion_ratio: f32,
        expansion_threshold_db: f32,
        attack_time: f32,
        release_time: f32,
        master_gain_db: f32,
    ) -> PyResult<Self> {
        let fmt_rs = format.to_rs()?;
        let params = DynamicsParams {
            sample_rate,
            threshold_db,
            knee_width_db,
            ratio,
            expansion_ratio,
            expansion_threshold_db,
            attack_time,
            release_time,
            master_gain_db,
        };
        let p = CompressorProcessor::new_with_format(fmt_rs, params).map_err(map_codec_err)?;
        let in_format = p.input_format();
        let out_format = p.output_format();
        let in_sample_type = in_format.map(|f| f.sample_format.sample_type());
        Ok(Self {
            p: Box::new(p),
            in_format,
            out_format,
            in_sample_type,
            sent_eof: false,
        })
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.p.name()
    }

    /// 输入一帧 PCM（numpy）。
    #[pyo3(signature = (pcm, pts=None))]
    fn put_frame(&mut self, py: Python<'_>, pcm: &Bound<'_, PyAny>, pts: Option<i64>) -> PyResult<()> {
        if self.sent_eof {
            return Err(PyRuntimeError::new_err(
                "Processor 已 flush（输入结束），不能再 put_frame；如需继续输入，请新建一个 Processor",
            ));
        }
        let in_fmt = self
            .in_format
            .ok_or_else(|| PyValueError::new_err("Processor 输入格式未知：请用 Processor.identity(format=...) 或 Processor.resample(in_format=...)"))?;
        let st = self
            .in_sample_type
            .ok_or_else(|| PyRuntimeError::new_err("missing input sample type"))?;
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
        self.p
            .send_frame(Some(&frame as &dyn AudioFrameView))
            .map_err(map_codec_err)?;
        Ok(())
    }

    /// flush：通知输入结束（之后继续 get_frame 直到 EOF）。
    fn flush(&mut self) -> PyResult<()> {
        if self.sent_eof {
            return Ok(());
        }
        self.p.send_frame(None).map_err(map_codec_err)?;
        self.sent_eof = true;
        Ok(())
    }

    /// 重置内部状态（清空缓存、回到初始态），可继续接收新的流。
    fn reset(&mut self) -> PyResult<()> {
        self.p.reset().map_err(map_codec_err)?;
        self.sent_eof = false;
        Ok(())
    }

    /// 取出一帧输出 PCM：
    /// - 返回 numpy；无输出返回 None
    #[pyo3(signature = (force=false, layout="planar"))]
    fn get_frame(&mut self, py: Python<'_>, force: bool, layout: &str) -> PyResult<Option<PyObject>> {
        // 方案 B：force=True 触发一次内部 flush/吐尾巴（与 Encoder 的直觉对齐）。
        if force && !self.sent_eof {
            self.flush()?;
        }
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

fn node_kind_from_str(s: &str) -> Option<NodeBufferKind> {
    match s.to_ascii_lowercase().as_str() {
        "pcm" => Some(NodeBufferKind::Pcm),
        "packet" => Some(NodeBufferKind::Packet),
        _ => None,
    }
}

/// pipeline 节点：IdentityNode 的 config
#[pyclass(name = "IdentityNodeConfig")]
#[derive(Clone)]
pub struct IdentityNodeConfigPy {
    /// "pcm" | "packet"
    #[pyo3(get, set)]
    pub kind: String,
}

#[pymethods]
impl IdentityNodeConfigPy {
    #[new]
    fn new(kind: &str) -> Self {
        Self { kind: kind.into() }
    }
}

/// pipeline 节点：ResampleProcessorNode 的 config
#[pyclass(name = "ResampleNodeConfig")]
#[derive(Clone)]
pub struct ResampleNodeConfigPy {
    #[pyo3(get, set)]
    pub in_format: AudioFormat,
    #[pyo3(get, set)]
    pub out_format: AudioFormat,
    #[pyo3(get, set)]
    pub out_chunk_samples: Option<usize>,
    #[pyo3(get, set)]
    pub pad_final: bool,
}

#[pymethods]
impl ResampleNodeConfigPy {
    #[new]
    #[pyo3(signature = (in_format, out_format, out_chunk_samples=None, pad_final=true))]
    fn new(in_format: AudioFormat, out_format: AudioFormat, out_chunk_samples: Option<usize>, pad_final: bool) -> Self {
        Self {
            in_format,
            out_format,
            out_chunk_samples,
            pad_final,
        }
    }
}

/// pipeline 节点：GainProcessorNode 的 config
#[pyclass(name = "GainNodeConfig")]
#[derive(Clone)]
pub struct GainNodeConfigPy {
    #[pyo3(get, set)]
    pub format: AudioFormat,
    #[pyo3(get, set)]
    pub gain: f64,
}

#[pymethods]
impl GainNodeConfigPy {
    #[new]
    fn new(format: AudioFormat, gain: f64) -> Self {
        Self { format, gain }
    }
}

/// pipeline 节点：CompressorProcessorNode 的 config
#[pyclass(name = "CompressorNodeConfig")]
#[derive(Clone)]
pub struct CompressorNodeConfigPy {
    #[pyo3(get, set)]
    pub format: AudioFormat,
    /// None => 使用 format.sample_rate
    #[pyo3(get, set)]
    pub sample_rate: Option<f32>,
    #[pyo3(get, set)]
    pub threshold_db: f32,
    #[pyo3(get, set)]
    pub knee_width_db: f32,
    #[pyo3(get, set)]
    pub ratio: f32,
    #[pyo3(get, set)]
    pub expansion_ratio: f32,
    #[pyo3(get, set)]
    pub expansion_threshold_db: f32,
    #[pyo3(get, set)]
    pub attack_time: f32,
    #[pyo3(get, set)]
    pub release_time: f32,
    #[pyo3(get, set)]
    pub master_gain_db: f32,
}

#[pymethods]
impl CompressorNodeConfigPy {
    #[new]
    #[pyo3(signature = (
        format,
        sample_rate=None,
        threshold_db=-18.0,
        knee_width_db=6.0,
        ratio=4.0,
        expansion_ratio=2.0,
        expansion_threshold_db=-60.0,
        attack_time=0.01,
        release_time=0.10,
        master_gain_db=0.0
    ))]
    fn new(
        format: AudioFormat,
        sample_rate: Option<f32>,
        threshold_db: f32,
        knee_width_db: f32,
        ratio: f32,
        expansion_ratio: f32,
        expansion_threshold_db: f32,
        attack_time: f32,
        release_time: f32,
        master_gain_db: f32,
    ) -> Self {
        Self {
            format,
            sample_rate,
            threshold_db,
            knee_width_db,
            ratio,
            expansion_ratio,
            expansion_threshold_db,
            attack_time,
            release_time,
            master_gain_db,
        }
    }
}

/// 统一的 processor 节点工厂：通过字符串选择 processor，并用 *NodeConfig 提供参数。
#[pyfunction]
pub fn make_processor_node(kind: &str, config: &Bound<'_, PyAny>) -> PyResult<DynNodePy> {
    match kind.to_ascii_lowercase().as_str() {
        "identity" => {
            let cfg = config.extract::<IdentityNodeConfigPy>()?;
            let nk = node_kind_from_str(&cfg.kind)
                .ok_or_else(|| PyValueError::new_err("IdentityNodeConfig.kind 仅支持: pcm/packet"))?;
            Ok(DynNodePy::new_boxed(Box::new(IdentityNode::new(nk))))
        }
        "resample" => {
            let cfg = config.extract::<ResampleNodeConfigPy>()?;
            let in_fmt = cfg.in_format.to_rs()?;
            let out_fmt = cfg.out_format.to_rs()?;
            let mut p = ResampleProcessor::new(in_fmt, out_fmt).map_err(map_codec_err)?;
            p.set_output_chunker(cfg.out_chunk_samples, cfg.pad_final)
                .map_err(map_codec_err)?;
            Ok(DynNodePy::new_boxed(Box::new(ProcessorNode::new(p))))
        }
        "gain" => {
            let cfg = config.extract::<GainNodeConfigPy>()?;
            let fmt = cfg.format.to_rs()?;
            let p = GainProcessor::new_with_format(fmt, cfg.gain).map_err(map_codec_err)?;
            Ok(DynNodePy::new_boxed(Box::new(ProcessorNode::new(p))))
        }
        "compressor" => {
            let cfg = config.extract::<CompressorNodeConfigPy>()?;
            let fmt = cfg.format.to_rs()?;
            let sample_rate = cfg.sample_rate.unwrap_or(cfg.format.sample_rate as f32);
            let params = DynamicsParams {
                sample_rate,
                threshold_db: cfg.threshold_db,
                knee_width_db: cfg.knee_width_db,
                ratio: cfg.ratio,
                expansion_ratio: cfg.expansion_ratio,
                expansion_threshold_db: cfg.expansion_threshold_db,
                attack_time: cfg.attack_time,
                release_time: cfg.release_time,
                master_gain_db: cfg.master_gain_db,
            };
            let p = CompressorProcessor::new_with_format(fmt, params).map_err(map_codec_err)?;
            Ok(DynNodePy::new_boxed(Box::new(ProcessorNode::new(p))))
        }
        _ => Err(PyValueError::new_err("kind only support: identity/resample/gain/compressor")),
    }
}


