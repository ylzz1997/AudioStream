#![allow(unsafe_op_in_unsafe_fn)]

use crate::codec::error::CodecError;
use crate::codec::processor::compressor_processor::CompressorProcessor;
use crate::codec::processor::gain_processor::GainProcessor;
use crate::codec::processor::identity_processor::IdentityProcessor;
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::codec::processor::resample_processor::ResampleProcessor;
use crate::common::audio::audio::{AudioFrameView, AudioFrameViewMut, SampleType};
use crate::function::compressor::DynamicsParams;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

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
        })
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.p.name()
    }

    /// 输入一帧 PCM（numpy）。
    #[pyo3(signature = (pcm, pts=None))]
    fn put_frame(&mut self, py: Python<'_>, pcm: &Bound<'_, PyAny>, pts: Option<i64>) -> PyResult<()> {
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

fn node_kind_from_str(s: &str) -> Option<NodeBufferKind> {
    match s.to_ascii_lowercase().as_str() {
        "pcm" => Some(NodeBufferKind::Pcm),
        "packet" => Some(NodeBufferKind::Packet),
        _ => None,
    }
}

#[pyfunction]
#[pyo3(signature = (in_format, out_format, out_chunk_samples=None, pad_final=true))]
pub fn make_resample_node(
    in_format: AudioFormat,
    out_format: AudioFormat,
    out_chunk_samples: Option<usize>,
    pad_final: bool,
) -> PyResult<DynNodePy> {
    let in_fmt = in_format.to_rs()?;
    let out_fmt = out_format.to_rs()?;
    let mut p = ResampleProcessor::new(in_fmt, out_fmt).map_err(map_codec_err)?;
    p.set_output_chunker(out_chunk_samples, pad_final).map_err(map_codec_err)?;
    Ok(DynNodePy::new_boxed(Box::new(ProcessorNode::new(p))))
}

#[pyfunction]
pub fn make_gain_node(format: AudioFormat, gain: f64) -> PyResult<DynNodePy> {
    let fmt = format.to_rs()?;
    let p = GainProcessor::new_with_format(fmt, gain).map_err(map_codec_err)?;
    Ok(DynNodePy::new_boxed(Box::new(ProcessorNode::new(p))))
}

#[pyfunction]
pub fn make_compressor_node(
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
) -> PyResult<DynNodePy> {
    let fmt = format.to_rs()?;
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
    let p = CompressorProcessor::new_with_format(fmt, params).map_err(map_codec_err)?;
    Ok(DynNodePy::new_boxed(Box::new(ProcessorNode::new(p))))
}

fn get_required<'py, T: FromPyObject<'py>>(d: &'py Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    let v = d
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required key: {key}")))?;
    v.extract::<T>()
}

fn get_optional<'py, T: FromPyObject<'py>>(d: &'py Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        Some(v) => {
            if v.is_none() {
                Ok(None)
            } else {
                Ok(Some(v.extract::<T>()?))
            }
        }
        None => Ok(None),
    }
}

fn get_optional_or<'py, T: FromPyObject<'py>>(d: &'py Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T> {
    Ok(get_optional::<T>(d, key)?.unwrap_or(default))
}

/// 统一的 processor 节点工厂：通过字符串选择 processor，并用 dict 提供参数。
#[pyfunction]
pub fn make_processor_node(kind: &str, config: &Bound<'_, PyAny>) -> PyResult<DynNodePy> {
    let d: Bound<'_, PyDict> = config
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("config 需要是 dict"))?
        .clone();

    match kind.to_ascii_lowercase().as_str() {
        "identity" => {
            let k: String = get_required(&d, "kind")?;
            let nk = node_kind_from_str(&k).ok_or_else(|| PyValueError::new_err("identity.kind 仅支持: pcm/packet"))?;
            Ok(DynNodePy::new_boxed(Box::new(IdentityNode::new(nk))))
        }
        "resample" => {
            let in_format: AudioFormat = get_required(&d, "in_format")?;
            let out_format: AudioFormat = get_required(&d, "out_format")?;
            let out_chunk_samples: Option<usize> = get_optional(&d, "out_chunk_samples")?;
            let pad_final: bool = get_optional_or(&d, "pad_final", true)?;
            make_resample_node(in_format, out_format, out_chunk_samples, pad_final)
        }
        "gain" => {
            let format: AudioFormat = get_required(&d, "format")?;
            let gain: f64 = get_required(&d, "gain")?;
            make_gain_node(format, gain)
        }
        "compressor" => {
            let format: AudioFormat = get_required(&d, "format")?;
            let sample_rate: f32 = match get_optional::<f32>(&d, "sample_rate")? {
                Some(v) => v,
                None => format.sample_rate as f32,
            };
            let threshold_db: f32 = get_required(&d, "threshold_db")?;
            let knee_width_db: f32 = get_required(&d, "knee_width_db")?;
            let ratio: f32 = get_required(&d, "ratio")?;
            let expansion_ratio: f32 = get_required(&d, "expansion_ratio")?;
            let expansion_threshold_db: f32 = get_required(&d, "expansion_threshold_db")?;
            let attack_time: f32 = get_required(&d, "attack_time")?;
            let release_time: f32 = get_required(&d, "release_time")?;
            let master_gain_db: f32 = get_required(&d, "master_gain_db")?;
            make_compressor_node(
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
            )
        }
        _ => Err(PyValueError::new_err("kind only support: identity/resample/gain/compressor")),
    }
}


