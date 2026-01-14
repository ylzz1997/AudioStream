#![allow(unsafe_op_in_unsafe_fn)]

use crate::codec::encoder::aac_encoder::AacEncoderConfig;
use crate::codec::encoder::flac_encoder::FlacEncoderConfig;
use crate::codec::encoder::opus_encoder::OpusEncoderConfig;
use crate::codec::error::CodecError;
use crate::codec::packet::{CodecPacket, PacketFlags};
use crate::common::audio::audio::{
    AudioFormat as RsAudioFormat, AudioFrame, AudioFrameView, AudioFrameViewMut, Rational, SampleType,
};
use crate::common::io::file as rs_file;
use crate::common::io::file::{
    AudioFileReadConfig, AudioFileReader as RsAudioFileReader, AudioFileWriteConfig, AudioFileWriter as RsAudioFileWriter,
};
use crate::common::io::boundwriter::{MultiAudioWriter, ParallelAudioWriter as RsParallelAudioWriter};
use crate::pipeline::sink::{AsyncParallelAudioSink as RsAsyncParallelAudioSink, AsyncPipelineAudioSink as RsAsyncPipelineAudioSink};
use crate::common::io::LineAudioWriter as RsLineAudioWriter;
use crate::common::io::io::{AudioReader, AudioWriter};
use crate::pipeline::forkjoinnode::AsyncForkJoinNode as RsAsyncForkJoinNode;
use crate::pipeline::forkjoinnode::Reduce as RsReduce;
use crate::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::dynamic_node_interface::BoxedProcessorNode;
use crate::pipeline::node::node_interface::{AsyncPipeline, NodeBuffer, NodeBufferKind};
use crate::pipeline::node::node_interface::IdentityNode;
use crate::pipeline::node::tap_node::TapNode as RsTapNode;
use crate::pipeline::sink::audio_sink::{AudioSink, AsyncAudioSink};
use crate::pipeline::source::audio_source::AudioSource;
use crate::runner::async_auto_runner::AsyncAutoRunner;
use crate::runner::async_runner_interface::AsyncRunner;
use crate::runner::error::{RunnerError, RunnerResult};

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::sync::mpsc as std_mpsc;
use tokio::runtime::{Builder, Runtime};

use crate::python::errors::{map_codec_err, map_file_err, map_runner_err, pyerr_to_codec_err, pyerr_to_runner_err};
use crate::python::format::{
    ascontig_cast_2d, audio_format_from_rs, frame_to_numpy, ndarray_to_frame_interleaved, ndarray_to_frame_planar,
    AudioFormat,
};
use crate::python::processor::ProcessorPy;

// 上面 import 里用到的 `node_kind_from_str` 在 format.rs 不存在；
// 这里用私有实现，避免模块间循环依赖。

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

fn map_audio_err_to_codec_err(e: crate::common::audio::audio::AudioError) -> CodecError {
    CodecError::Other(e.to_string())
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
    pub(crate) fn to_rs(&self) -> CodecPacket {
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

    pub(crate) fn from_rs(p: &CodecPacket) -> Self {
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
    pub(crate) inner: Option<NodeBuffer>,
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
            frame.set_time_base(Rational::new(n, d))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
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

    /// 如果是 pcm，返回 numpy ndarray（默认 planar）；否则返回 None。
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
    pub(crate) fn take_inner(&mut self) -> PyResult<NodeBuffer> {
        self.inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("NodeBuffer 已被移动（不可再次使用）"))
    }
}

// ---------------------------
// Rust 预置 Reduce（Python 可用）
// ---------------------------

fn ensure_non_empty_items(items: &[NodeBuffer], name: &'static str) -> crate::codec::error::CodecResult<()> {
    if items.is_empty() {
        return Err(CodecError::InvalidData(match name {
            "sum" => "reduce_sum expects non-empty items",
            "product" => "reduce_product expects non-empty items",
            "mean" => "reduce_mean expects non-empty items",
            "max" => "reduce_max expects non-empty items",
            "min" => "reduce_min expects non-empty items",
            "concat" => "reduce_concat expects non-empty items",
            "xor" => "reduce_xor expects non-empty items",
            _ => "reduce expects non-empty items",
        }));
    }
    Ok(())
}

fn pcm_validate_same(items: &[AudioFrame]) -> crate::codec::error::CodecResult<(RsAudioFormat, usize, Rational, Option<i64>)> {
    let f0 = items[0].format();
    let n0 = items[0].nb_samples();
    let tb0 = items[0].time_base();
    let pts0 = items[0].pts();

    for f in items.iter().skip(1) {
        if f.format() != f0 {
            return Err(CodecError::InvalidData("PCM reduce requires same AudioFormat across branches"));
        }
        if f.nb_samples() != n0 {
            return Err(CodecError::InvalidData("PCM reduce requires same nb_samples across branches"));
        }
        if f.time_base() != tb0 {
            return Err(CodecError::InvalidData("PCM reduce requires same time_base across branches"));
        }
    }
    Ok((f0, n0, tb0, pts0))
}

fn pcm_validate_same_for_concat(items: &[AudioFrame]) -> crate::codec::error::CodecResult<(RsAudioFormat, Rational, Option<i64>)> {
    let f0 = items[0].format();
    let tb0 = items[0].time_base();
    let pts0 = items[0].pts();
    for f in items.iter().skip(1) {
        if f.format() != f0 {
            return Err(CodecError::InvalidData("PCM concat requires same AudioFormat across branches"));
        }
        if f.time_base() != tb0 {
            return Err(CodecError::InvalidData("PCM concat requires same time_base across branches"));
        }
    }
    Ok((f0, tb0, pts0))
}

#[inline]
fn pcm_sample_to_f64(sample_type: SampleType, bytes: &[u8]) -> f64 {
    match sample_type {
        SampleType::U8 => (bytes[0] as f64 - 128.0) / 128.0,
        SampleType::I16 => {
            let v = i16::from_ne_bytes([bytes[0], bytes[1]]) as f64;
            v / 32768.0
        }
        SampleType::I32 => {
            let v = i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64;
            v / 2147483648.0
        }
        SampleType::I64 => {
            let v = i64::from_ne_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as f64;
            v / 9223372036854775808.0
        }
        SampleType::F32 => {
            let v = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            v as f64
        }
        SampleType::F64 => f64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

#[inline]
fn pcm_write_sample_from_f64(sample_type: SampleType, v: f64, out: &mut [u8]) {
    match sample_type {
        SampleType::U8 => {
            let x = (v.clamp(-1.0, 1.0) * 128.0 + 128.0).round();
            let x = x.clamp(0.0, 255.0) as u8;
            out[0] = x;
        }
        SampleType::I16 => {
            let x = (v.clamp(-1.0, 1.0) * 32768.0).round();
            let x = x.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
            out.copy_from_slice(&x.to_ne_bytes());
        }
        SampleType::I32 => {
            let x = (v.clamp(-1.0, 1.0) * 2147483648.0).round();
            let x = x.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            out.copy_from_slice(&x.to_ne_bytes());
        }
        SampleType::I64 => {
            let x = (v.clamp(-1.0, 1.0) * 9223372036854775808.0).round();
            let x = x.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
            out.copy_from_slice(&x.to_ne_bytes());
        }
        SampleType::F32 => {
            let x = v as f32;
            out.copy_from_slice(&x.to_ne_bytes());
        }
        SampleType::F64 => out.copy_from_slice(&v.to_ne_bytes()),
    }
}

fn reduce_pcm_any_sum(frames: &[AudioFrame], weight: Option<&[f64]>) -> crate::codec::error::CodecResult<AudioFrame> {
    let (fmt, nb_samples, tb, pts) = pcm_validate_same(frames)?;
    let plane_count = frames[0].plane_count();
    let bps = fmt.sample_format.bytes_per_sample();
    let st = fmt.sample_format.sample_type();

    if let Some(ws) = weight {
        if ws.len() != frames.len() {
            return Err(CodecError::InvalidData("weight length mismatch"));
        }
    }

    let mut out = AudioFrame::new_alloc(fmt, nb_samples).map_err(map_audio_err_to_codec_err)?;
    out.set_time_base(tb).map_err(map_audio_err_to_codec_err)?;
    out.set_pts(pts);

    for pi in 0..plane_count {
        let out_plane = out
            .plane_mut(pi)
            .ok_or(CodecError::InvalidState("invalid plane index"))?;
        for (si, out_chunk) in out_plane.chunks_exact_mut(bps).enumerate() {
            let base = si * bps;
            let mut acc: f64 = 0.0;
            for (bi, f) in frames.iter().enumerate() {
                let p = f.plane(pi).ok_or(CodecError::InvalidState("invalid plane index"))?;
                let v = pcm_sample_to_f64(st, &p[base..base + bps]);
                let w = weight.map(|ws| ws[bi]).unwrap_or(1.0);
                acc += v * w;
            }
            pcm_write_sample_from_f64(st, acc, out_chunk);
        }
    }
    Ok(out)
}

fn reduce_pcm_any_product(frames: &[AudioFrame], weight: Option<&[f64]>) -> crate::codec::error::CodecResult<AudioFrame> {
    let (fmt, nb_samples, tb, pts) = pcm_validate_same(frames)?;
    let plane_count = frames[0].plane_count();
    let bps = fmt.sample_format.bytes_per_sample();
    let st = fmt.sample_format.sample_type();

    if let Some(ws) = weight {
        if ws.len() != frames.len() {
            return Err(CodecError::InvalidData("weight length mismatch"));
        }
    }

    let mut out = AudioFrame::new_alloc(fmt, nb_samples).map_err(map_audio_err_to_codec_err)?;
    out.set_time_base(tb).map_err(map_audio_err_to_codec_err)?;
    out.set_pts(pts);

    for pi in 0..plane_count {
        let out_plane = out
            .plane_mut(pi)
            .ok_or(CodecError::InvalidState("invalid plane index"))?;
        for (si, out_chunk) in out_plane.chunks_exact_mut(bps).enumerate() {
            let base = si * bps;
            let mut acc: f64 = 1.0;
            for (bi, f) in frames.iter().enumerate() {
                let p = f.plane(pi).ok_or(CodecError::InvalidState("invalid plane index"))?;
                let v = pcm_sample_to_f64(st, &p[base..base + bps]);
                let w = weight.map(|ws| ws[bi]).unwrap_or(1.0);
                acc *= v * w;
            }
            pcm_write_sample_from_f64(st, acc, out_chunk);
        }
    }
    Ok(out)
}

fn reduce_pcm_any_mean(frames: &[AudioFrame]) -> crate::codec::error::CodecResult<AudioFrame> {
    let n = frames.len() as f64;
    let (fmt, nb_samples, tb, pts) = pcm_validate_same(frames)?;
    let plane_count = frames[0].plane_count();
    let bps = fmt.sample_format.bytes_per_sample();
    let st = fmt.sample_format.sample_type();

    let mut out = AudioFrame::new_alloc(fmt, nb_samples).map_err(map_audio_err_to_codec_err)?;
    out.set_time_base(tb).map_err(map_audio_err_to_codec_err)?;
    out.set_pts(pts);

    for pi in 0..plane_count {
        let out_plane = out
            .plane_mut(pi)
            .ok_or(CodecError::InvalidState("invalid plane index"))?;
        for (si, out_chunk) in out_plane.chunks_exact_mut(bps).enumerate() {
            let base = si * bps;
            let mut acc: f64 = 0.0;
            for f in frames.iter() {
                let p = f.plane(pi).ok_or(CodecError::InvalidState("invalid plane index"))?;
                acc += pcm_sample_to_f64(st, &p[base..base + bps]);
            }
            pcm_write_sample_from_f64(st, acc / n, out_chunk);
        }
    }
    Ok(out)
}

fn reduce_pcm_any_max(frames: &[AudioFrame]) -> crate::codec::error::CodecResult<AudioFrame> {
    let (fmt, nb_samples, tb, pts) = pcm_validate_same(frames)?;
    let plane_count = frames[0].plane_count();
    let bps = fmt.sample_format.bytes_per_sample();
    let st = fmt.sample_format.sample_type();

    let mut out = AudioFrame::new_alloc(fmt, nb_samples).map_err(map_audio_err_to_codec_err)?;
    out.set_time_base(tb).map_err(map_audio_err_to_codec_err)?;
    out.set_pts(pts);

    for pi in 0..plane_count {
        let out_plane = out
            .plane_mut(pi)
            .ok_or(CodecError::InvalidState("invalid plane index"))?;
        for (si, out_chunk) in out_plane.chunks_exact_mut(bps).enumerate() {
            let base = si * bps;
            let mut best: f64 = f64::NEG_INFINITY;
            for f in frames.iter() {
                let p = f.plane(pi).ok_or(CodecError::InvalidState("invalid plane index"))?;
                let v = pcm_sample_to_f64(st, &p[base..base + bps]);
                if v > best {
                    best = v;
                }
            }
            pcm_write_sample_from_f64(st, best, out_chunk);
        }
    }
    Ok(out)
}

fn reduce_pcm_any_min(frames: &[AudioFrame]) -> crate::codec::error::CodecResult<AudioFrame> {
    let (fmt, nb_samples, tb, pts) = pcm_validate_same(frames)?;
    let plane_count = frames[0].plane_count();
    let bps = fmt.sample_format.bytes_per_sample();
    let st = fmt.sample_format.sample_type();

    let mut out = AudioFrame::new_alloc(fmt, nb_samples).map_err(map_audio_err_to_codec_err)?;
    out.set_time_base(tb).map_err(map_audio_err_to_codec_err)?;
    out.set_pts(pts);

    for pi in 0..plane_count {
        let out_plane = out
            .plane_mut(pi)
            .ok_or(CodecError::InvalidState("invalid plane index"))?;
        for (si, out_chunk) in out_plane.chunks_exact_mut(bps).enumerate() {
            let base = si * bps;
            let mut best: f64 = f64::INFINITY;
            for f in frames.iter() {
                let p = f.plane(pi).ok_or(CodecError::InvalidState("invalid plane index"))?;
                let v = pcm_sample_to_f64(st, &p[base..base + bps]);
                if v < best {
                    best = v;
                }
            }
            pcm_write_sample_from_f64(st, best, out_chunk);
        }
    }
    Ok(out)
}

fn reduce_pcm_any_concat(frames: &[AudioFrame]) -> crate::codec::error::CodecResult<AudioFrame> {
    if frames.is_empty() {
        return Err(CodecError::InvalidData("concat reduce expects non-empty items"));
    }
    let (fmt, tb, pts) = pcm_validate_same_for_concat(frames)?;
    let plane_count = frames[0].plane_count();
    let total_nb_samples: usize = frames.iter().map(|f| f.nb_samples()).sum();

    let mut out = AudioFrame::new_alloc(fmt, total_nb_samples).map_err(map_audio_err_to_codec_err)?;
    out.set_time_base(tb).map_err(map_audio_err_to_codec_err)?;
    out.set_pts(pts);

    for pi in 0..plane_count {
        let out_plane = out
            .plane_mut(pi)
            .ok_or(CodecError::InvalidState("invalid plane index"))?;
        let mut cursor: usize = 0;
        for f in frames.iter() {
            let p = f.plane(pi).ok_or(CodecError::InvalidState("invalid plane index"))?;
            out_plane[cursor..cursor + p.len()].copy_from_slice(p);
            cursor += p.len();
        }
    }
    Ok(out)
}

fn reduce_nodebuffer_pcm(
    items: &[NodeBuffer],
    op: &'static str,
    weight_f64: Option<&[f64]>,
) -> crate::codec::error::CodecResult<NodeBuffer> {
    ensure_non_empty_items(items, op)?;
    let mut frames: Vec<AudioFrame> = Vec::with_capacity(items.len());
    for b in items {
        match b {
            NodeBuffer::Pcm(f) => frames.push(f.clone()),
            _ => return Err(CodecError::InvalidData("PCM reduce expects NodeBuffer(pcm)")),
        }
    }

    let out = match op {
        "sum" => reduce_pcm_any_sum(&frames, weight_f64)?,
        "product" => reduce_pcm_any_product(&frames, weight_f64)?,
        "mean" => {
            if weight_f64.is_some() {
                return Err(CodecError::InvalidData("mean reduce does not support weight"));
            }
            reduce_pcm_any_mean(&frames)?
        }
        "max" => {
            if weight_f64.is_some() {
                return Err(CodecError::InvalidData("max reduce does not support weight"));
            }
            reduce_pcm_any_max(&frames)?
        }
        "min" => {
            if weight_f64.is_some() {
                return Err(CodecError::InvalidData("min reduce does not support weight"));
            }
            reduce_pcm_any_min(&frames)?
        }
        _ => return Err(CodecError::Unsupported("unsupported pcm reduce op")),
    };
    Ok(NodeBuffer::Pcm(out))
}

fn reduce_nodebuffer_pcm_concat(items: &[NodeBuffer]) -> crate::codec::error::CodecResult<NodeBuffer> {
    ensure_non_empty_items(items, "concat")?;
    let mut frames: Vec<AudioFrame> = Vec::with_capacity(items.len());
    for b in items {
        match b {
            NodeBuffer::Pcm(f) => frames.push(f.clone()),
            _ => return Err(CodecError::InvalidData("concat reduce expects NodeBuffer(pcm)")),
        }
    }
    let out = reduce_pcm_any_concat(&frames)?;
    Ok(NodeBuffer::Pcm(out))
}

fn reduce_nodebuffer_packet_concat(items: &[NodeBuffer]) -> crate::codec::error::CodecResult<NodeBuffer> {
    ensure_non_empty_items(items, "concat")?;
    let mut packets: Vec<CodecPacket> = Vec::with_capacity(items.len());
    for b in items {
        match b {
            NodeBuffer::Packet(p) => packets.push(p.clone()),
            _ => return Err(CodecError::InvalidData("concat reduce expects NodeBuffer(packet)")),
        }
    }
    let tb = packets[0].time_base;
    let mut data: Vec<u8> = Vec::new();
    for p in packets.iter() {
        if p.time_base != tb {
            return Err(CodecError::InvalidData("concat reduce requires same packet time_base"));
        }
        data.extend_from_slice(&p.data);
    }
    Ok(NodeBuffer::Packet(CodecPacket::new(data, tb)))
}

fn reduce_nodebuffer_packet_xor(items: &[NodeBuffer]) -> crate::codec::error::CodecResult<NodeBuffer> {
    ensure_non_empty_items(items, "xor")?;
    let mut packets: Vec<CodecPacket> = Vec::with_capacity(items.len());
    for b in items {
        match b {
            NodeBuffer::Packet(p) => packets.push(p.clone()),
            _ => return Err(CodecError::InvalidData("xor reduce expects NodeBuffer(packet)")),
        }
    }
    let tb = packets[0].time_base;
    let len = packets[0].data.len();
    for p in packets.iter() {
        if p.time_base != tb {
            return Err(CodecError::InvalidData("xor reduce requires same packet time_base"));
        }
        if p.data.len() != len {
            return Err(CodecError::InvalidData("xor reduce requires same packet length"));
        }
    }
    let mut out = vec![0u8; len];
    for p in packets.iter() {
        for i in 0..len {
            out[i] ^= p.data[i];
        }
    }
    Ok(NodeBuffer::Packet(CodecPacket::new(out, tb)))
}

#[pyclass(name = "ReduceSum")]
#[derive(Default)]
pub struct ReduceSumPy {
    weight: Option<Vec<f64>>,
}

#[pymethods]
impl ReduceSumPy {
    #[new]
    #[pyo3(signature = (weight=None))]
    fn new(weight: Option<Vec<f64>>) -> Self {
        Self { weight }
    }

    fn __call__(&self, py: Python<'_>, items: Vec<Py<NodeBufferPy>>) -> PyResult<NodeBufferPy> {
        let mut rs_items: Vec<NodeBuffer> = Vec::with_capacity(items.len());
        for it in items {
            let mut b = it.bind(py).borrow_mut();
            rs_items.push(b.take_inner()?);
        }
        let out = reduce_nodebuffer_pcm(&rs_items, "sum", self.weight.as_deref()).map_err(map_codec_err)?;
        Ok(NodeBufferPy { inner: Some(out) })
    }
}

#[pyclass(name = "ReduceProduct")]
#[derive(Default)]
pub struct ReduceProductPy {
    weight: Option<Vec<f64>>,
}

#[pymethods]
impl ReduceProductPy {
    #[new]
    #[pyo3(signature = (weight=None))]
    fn new(weight: Option<Vec<f64>>) -> Self {
        Self { weight }
    }

    fn __call__(&self, py: Python<'_>, items: Vec<Py<NodeBufferPy>>) -> PyResult<NodeBufferPy> {
        let mut rs_items: Vec<NodeBuffer> = Vec::with_capacity(items.len());
        for it in items {
            let mut b = it.bind(py).borrow_mut();
            rs_items.push(b.take_inner()?);
        }
        let out = reduce_nodebuffer_pcm(&rs_items, "product", self.weight.as_deref()).map_err(map_codec_err)?;
        Ok(NodeBufferPy { inner: Some(out) })
    }
}

#[pyclass(name = "ReduceMean")]
#[derive(Default)]
pub struct ReduceMeanPy;

#[pymethods]
impl ReduceMeanPy {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __call__(&self, py: Python<'_>, items: Vec<Py<NodeBufferPy>>) -> PyResult<NodeBufferPy> {
        let mut rs_items: Vec<NodeBuffer> = Vec::with_capacity(items.len());
        for it in items {
            let mut b = it.bind(py).borrow_mut();
            rs_items.push(b.take_inner()?);
        }
        let out = reduce_nodebuffer_pcm(&rs_items, "mean", None).map_err(map_codec_err)?;
        Ok(NodeBufferPy { inner: Some(out) })
    }
}

#[pyclass(name = "ReduceMax")]
#[derive(Default)]
pub struct ReduceMaxPy;

#[pymethods]
impl ReduceMaxPy {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __call__(&self, py: Python<'_>, items: Vec<Py<NodeBufferPy>>) -> PyResult<NodeBufferPy> {
        let mut rs_items: Vec<NodeBuffer> = Vec::with_capacity(items.len());
        for it in items {
            let mut b = it.bind(py).borrow_mut();
            rs_items.push(b.take_inner()?);
        }
        let out = reduce_nodebuffer_pcm(&rs_items, "max", None).map_err(map_codec_err)?;
        Ok(NodeBufferPy { inner: Some(out) })
    }
}

#[pyclass(name = "ReduceMin")]
#[derive(Default)]
pub struct ReduceMinPy;

#[pymethods]
impl ReduceMinPy {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __call__(&self, py: Python<'_>, items: Vec<Py<NodeBufferPy>>) -> PyResult<NodeBufferPy> {
        let mut rs_items: Vec<NodeBuffer> = Vec::with_capacity(items.len());
        for it in items {
            let mut b = it.bind(py).borrow_mut();
            rs_items.push(b.take_inner()?);
        }
        let out = reduce_nodebuffer_pcm(&rs_items, "min", None).map_err(map_codec_err)?;
        Ok(NodeBufferPy { inner: Some(out) })
    }
}

#[pyclass(name = "ReduceConcat")]
#[derive(Default)]
pub struct ReduceConcatPy;

#[pymethods]
impl ReduceConcatPy {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __call__(&self, py: Python<'_>, items: Vec<Py<NodeBufferPy>>) -> PyResult<NodeBufferPy> {
        let mut rs_items: Vec<NodeBuffer> = Vec::with_capacity(items.len());
        for it in items {
            let mut b = it.bind(py).borrow_mut();
            rs_items.push(b.take_inner()?);
        }
        // concat 同时支持 packet 与 pcm（要求 items 的 kind 一致）
        let out = match rs_items.first() {
            Some(NodeBuffer::Packet(_)) => reduce_nodebuffer_packet_concat(&rs_items).map_err(map_codec_err)?,
            Some(NodeBuffer::Pcm(_)) => reduce_nodebuffer_pcm_concat(&rs_items).map_err(map_codec_err)?,
            None => return Err(PyValueError::new_err("reduce_concat expects non-empty items")),
        };
        Ok(NodeBufferPy { inner: Some(out) })
    }
}

#[pyclass(name = "ReduceXor")]
#[derive(Default)]
pub struct ReduceXorPy;

#[pymethods]
impl ReduceXorPy {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __call__(&self, py: Python<'_>, items: Vec<Py<NodeBufferPy>>) -> PyResult<NodeBufferPy> {
        let mut rs_items: Vec<NodeBuffer> = Vec::with_capacity(items.len());
        for it in items {
            let mut b = it.bind(py).borrow_mut();
            rs_items.push(b.take_inner()?);
        }
        let out = reduce_nodebuffer_packet_xor(&rs_items).map_err(map_codec_err)?;
        Ok(NodeBufferPy { inner: Some(out) })
    }
}

#[pyclass(name = "DynNode")]
pub struct DynNodePy {
    inner: Option<Box<dyn DynNode>>,
    pub(crate) in_kind: NodeBufferKind,
    pub(crate) out_kind: NodeBufferKind,
    name: &'static str,
}

impl DynNodePy {
    pub(crate) fn new_boxed(node: Box<dyn DynNode>) -> Self {
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

    pub(crate) fn take_inner(&mut self) -> PyResult<Box<dyn DynNode>> {
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

/// 让 Python 对象实现 DynNode。
struct PyCallbackNode {
    obj: Py<PyAny>,
    in_kind: NodeBufferKind,
    out_kind: NodeBufferKind,
    name: &'static str,
    flushed: bool,
}

impl PyCallbackNode {
    fn new(obj: Py<PyAny>, in_kind: NodeBufferKind, out_kind: NodeBufferKind, name: String) -> Self {
        // DynNode::name 需要 &'static str；这里把 name 泄漏到进程生命周期（模块卸载前都安全）。
        let leaked: &'static str = Box::leak(name.into_boxed_str());
        Self {
            obj,
            in_kind,
            out_kind,
            name: leaked,
            flushed: false,
        }
    }
}

impl DynNode for PyCallbackNode {
    fn name(&self) -> &'static str {
        self.name
    }
    fn input_kind(&self) -> NodeBufferKind {
        self.in_kind
    }
    fn output_kind(&self) -> NodeBufferKind {
        self.out_kind
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> crate::codec::error::CodecResult<()> {
        Python::with_gil(|py| {
            let o = self.obj.bind(py);
            match input {
                None => {
                    self.flushed = true;
                    // 优先调用 flush()（若存在），否则退化为 push(None)
                    match o.call_method0("flush") {
                        Ok(_) => Ok(()),
                        Err(e) => {
                            if e.is_instance_of::<PyAttributeError>(py) {
                                o.call_method1("push", (Option::<Py<NodeBufferPy>>::None,))
                                    .map(|_| ())
                                    .map_err(pyerr_to_codec_err)
                            } else {
                                Err(pyerr_to_codec_err(e))
                            }
                        }
                    }
                }
                Some(buf) => {
                    if buf.kind() != self.in_kind {
                        return Err(CodecError::InvalidData("Python node input kind mismatch"));
                    }
                    let nb = Py::new(py, NodeBufferPy { inner: Some(buf) }).map_err(pyerr_to_codec_err)?;
                    o.call_method1("push", (nb,)).map(|_| ()).map_err(pyerr_to_codec_err)
                }
            }
        })
    }

    fn pull(&mut self) -> crate::codec::error::CodecResult<NodeBuffer> {
        Python::with_gil(|py| {
            let o = self.obj.bind(py);
            let ret = o.call_method0("pull").map_err(pyerr_to_codec_err)?;
            if ret.is_none() {
                return Err(if self.flushed { CodecError::Eof } else { CodecError::Again });
            }
            let nb_py: Py<NodeBufferPy> = ret.extract().map_err(|_| CodecError::InvalidData("Python node.pull() 必须返回 NodeBuffer 或 None"))?;
            let mut nb = nb_py.bind(py).borrow_mut();
            let inner = nb
                .take_inner()
                .map_err(|_| CodecError::InvalidState("Python node.pull() 返回的 NodeBuffer 已被移动（不可再次使用）"))?;
            if inner.kind() != self.out_kind {
                return Err(CodecError::InvalidData("Python node output kind mismatch"));
            }
            Ok(inner)
        })
    }

    fn reset(&mut self) -> crate::codec::error::CodecResult<()> {
        Python::with_gil(|py| {
            self.flushed = false;
            let o = self.obj.bind(py);
            // reset() 是可选的；不存在则忽略
            match o.call_method0("reset") {
                Ok(_) => Ok(()),
                Err(e) => {
                    if e.is_instance_of::<PyAttributeError>(py) {
                        Ok(())
                    } else {
                        Err(pyerr_to_codec_err(e))
                    }
                }
            }
        })
    }
}

/// python 侧的 Node 基类
#[pyclass(name = "Node", subclass)]
pub struct NodeBase {}

#[pymethods]
impl NodeBase {
    #[new]
    fn new() -> Self {
        Self {}
    }
}

/// 仅用于类型提示/继承的空基类
#[pyclass(name = "AudioSource", subclass)]
pub struct AudioSourceBase {}

#[pymethods]
impl AudioSourceBase {
    #[new]
    fn new() -> Self {
        Self {}
    }
}

/// 仅用于类型提示/继承的空基类
#[pyclass(name = "AudioSink", subclass)]
pub struct AudioSinkBase {}

#[pymethods]
impl AudioSinkBase {
    #[new]
    fn new() -> Self {
        Self {}
    }
}

#[pyfunction]
pub fn make_identity_node(kind: &str) -> PyResult<DynNodePy> {
    let k = node_kind_from_str(kind).ok_or_else(|| PyValueError::new_err("kind 仅支持: pcm/packet"))?;
    Ok(DynNodePy::new_boxed(Box::new(IdentityNode::new(k))))
}

/// 创建一个 Tap 节点（tee）：把输入 **透传给下游** 的同时，**复制一份** 交给一个 Python sink 处理。
///
/// - `sink`：需要实现 `push(buf: NodeBuffer)` + `finalize()` 的对象（与 Runner 的 sink 约定一致）
/// - `kind`：tap 的输入/输出 kind（必须与相邻节点匹配），仅支持 "pcm"/"packet"
#[pyfunction]
#[pyo3(signature = (sink, kind="pcm"))]
pub fn make_tap_node(sink: Py<PyAny>, kind: &str) -> PyResult<DynNodePy> {
    let k = node_kind_from_str(kind).ok_or_else(|| PyValueError::new_err("kind 仅支持: pcm/packet"))?;
    let s = PyCallbackSink { obj: sink };
    Ok(DynNodePy::new_boxed(Box::new(RsTapNode::new(k, s))))
}

/// 创建一个 Python 自定义节点（DynNode），可用于 `AsyncDynPipeline`/`AsyncDynRunner` 的 nodes 列表。
#[pyfunction]
#[pyo3(signature = (obj, input_kind, output_kind, name="py-node".to_string()))]
pub fn make_python_node(obj: Py<PyAny>, input_kind: &str, output_kind: &str, name: String) -> PyResult<DynNodePy> {
    let in_k = node_kind_from_str(input_kind).ok_or_else(|| PyValueError::new_err("input_kind 仅支持: pcm/packet"))?;
    let out_k = node_kind_from_str(output_kind).ok_or_else(|| PyValueError::new_err("output_kind 仅支持: pcm/packet"))?;
    Ok(DynNodePy::new_boxed(Box::new(PyCallbackNode::new(obj, in_k, out_k, name))))
}

/// Python 侧 reduce 回调：`reduce(items: list[NodeBuffer]) -> NodeBuffer`
struct PyReduceFn {
    obj: Py<PyAny>,
    out_kind: NodeBufferKind,
}

impl PyReduceFn {
    fn new(obj: Py<PyAny>, out_kind: NodeBufferKind) -> Self {
        Self { obj, out_kind }
    }
}

impl RsReduce<NodeBuffer> for PyReduceFn {
    fn reduce(&self, items: &[NodeBuffer]) -> crate::codec::error::CodecResult<NodeBuffer> {
        Python::with_gil(|py| {
            let f = self.obj.bind(py);
            let mut py_items: Vec<Py<NodeBufferPy>> = Vec::with_capacity(items.len());
            for v in items.iter() {
                // items 只借用，构造 Python 侧 NodeBuffer 需 clone
                let nb = Py::new(py, NodeBufferPy { inner: Some(v.clone()) }).map_err(pyerr_to_codec_err)?;
                py_items.push(nb);
            }
            let list = PyList::new_bound(py, py_items);
            let ret = f.call1((list,)).map_err(pyerr_to_codec_err)?;
            if ret.is_none() {
                return Err(CodecError::InvalidData("Python reduce() 必须返回 NodeBuffer（不可为 None）"));
            }
            let nb_py: Py<NodeBufferPy> = ret
                .extract()
                .map_err(|_| CodecError::InvalidData("Python reduce() 必须返回 NodeBuffer"))?;
            let mut nb = nb_py.bind(py).borrow_mut();
            let inner = nb.take_inner().map_err(|_| CodecError::InvalidState("Python reduce() 返回的 NodeBuffer 已被移动（不可再次使用）"))?;
            if inner.kind() != self.out_kind {
                return Err(CodecError::InvalidData("Python reduce() 返回的 NodeBuffer kind 与分支 output_kind 不匹配"));
            }
            Ok(inner)
        })
    }
}

/// 延迟初始化的 async fork-join 节点：等真正运行在 tokio runtime 内时再构建分支 `AsyncDynPipeline`。
struct LazyAsyncForkJoinNode {
    name: &'static str,
    in_kind: NodeBufferKind,
    out_kind: NodeBufferKind,
    branches: Option<Vec<Vec<Box<dyn DynNode>>>>,
    reducer: Option<PyReduceFn>,
    inner: Option<RsAsyncForkJoinNode<PyReduceFn>>,
}

impl LazyAsyncForkJoinNode {
    fn new(
        name: String,
        in_kind: NodeBufferKind,
        out_kind: NodeBufferKind,
        branches: Vec<Vec<Box<dyn DynNode>>>,
        reducer: PyReduceFn,
    ) -> Self {
        let leaked: &'static str = Box::leak(name.into_boxed_str());
        Self {
            name: leaked,
            in_kind,
            out_kind,
            branches: Some(branches),
            reducer: Some(reducer),
            inner: None,
        }
    }

    fn ensure_init(&mut self) -> crate::codec::error::CodecResult<()> {
        if self.inner.is_some() {
            return Ok(());
        }
        // 必须在 tokio runtime 上下文中运行（AsyncDynPipeline::new 会 spawn_blocking）
        tokio::runtime::Handle::try_current()
            .map_err(|_| CodecError::InvalidState("AsyncForkJoinNode 需要在 tokio runtime 内运行（请将该节点用于 AsyncDynPipeline/AsyncDynRunner）"))?;

        let branches = self
            .branches
            .take()
            .ok_or(CodecError::InvalidState("fork-join branches already moved"))?;

        let mut ps = Vec::with_capacity(branches.len());
        for nodes in branches {
            let p = AsyncDynPipeline::new(nodes)?;
            ps.push(p);
        }
        let reducer = self
            .reducer
            .take()
            .ok_or(CodecError::InvalidState("fork-join reducer already moved"))?;
        let node = RsAsyncForkJoinNode::new(ps, reducer)?;
        self.inner = Some(node);
        Ok(())
    }
}

impl DynNode for LazyAsyncForkJoinNode {
    fn name(&self) -> &'static str {
        self.name
    }
    fn input_kind(&self) -> NodeBufferKind {
        self.in_kind
    }
    fn output_kind(&self) -> NodeBufferKind {
        self.out_kind
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> crate::codec::error::CodecResult<()> {
        self.ensure_init()?;
        self.inner
            .as_mut()
            .ok_or(CodecError::InvalidState("fork-join not initialized"))?
            .push(input)
    }

    fn pull(&mut self) -> crate::codec::error::CodecResult<NodeBuffer> {
        self.ensure_init()?;
        self.inner
            .as_mut()
            .ok_or(CodecError::InvalidState("fork-join not initialized"))?
            .pull()
    }

    fn reset(&mut self) -> crate::codec::error::CodecResult<()> {
        // 若还没初始化（从未运行过），直接恢复初始状态即可
        if self.inner.is_none() {
            return Ok(());
        }
        self.inner
            .as_mut()
            .ok_or(CodecError::InvalidState("fork-join not initialized"))?
            .reset()
    }
}

/// 构造一个可用于 `AsyncDynPipeline(nodes=[...])` 的 async fork-join 节点。
///
/// - `pipelines`: 多条分支，每条是 `DynNode` 列表（会被 move/搬空）
/// - `reduce`: Python 回调 `reduce(items: list[NodeBuffer]) -> NodeBuffer`
#[pyfunction]
#[pyo3(signature = (pipelines, reduce, name="async-fork-join".to_string()))]
pub fn make_async_fork_join_node(
    py: Python<'_>,
    pipelines: Vec<Vec<Py<DynNodePy>>>,
    reduce: Py<PyAny>,
    name: String,
) -> PyResult<DynNodePy> {
    if pipelines.is_empty() {
        return Err(PyValueError::new_err("pipelines 不能为空"));
    }
    if !reduce.bind(py).is_callable() {
        return Err(PyValueError::new_err("reduce 必须是可调用对象（callable）"));
    }

    let mut branches: Vec<Vec<Box<dyn DynNode>>> = Vec::with_capacity(pipelines.len());
    let mut in_kind: Option<NodeBufferKind> = None;
    let mut out_kind: Option<NodeBufferKind> = None;

    for (bi, nodes) in pipelines.into_iter().enumerate() {
        if nodes.is_empty() {
            return Err(PyValueError::new_err("pipelines 中每个分支都必须至少包含 1 个 DynNode"));
        }
        let nlen = nodes.len();
        let mut branch: Vec<Box<dyn DynNode>> = Vec::with_capacity(nodes.len());
        for (i, n) in nodes.into_iter().enumerate() {
            let mut nb = n.bind(py).borrow_mut();
            let node_in = nb.in_kind;
            let node_out = nb.out_kind;

            // 记录/校验分支内 kind 连接：相邻节点 out == next in
            if i > 0 {
                let prev_out = branch
                    .last()
                    .expect("i>0 implies non-empty")
                    .output_kind();
                if prev_out != node_in {
                    return Err(PyValueError::new_err("分支 pipeline 内相邻节点 kind 不匹配"));
                }
            }

            // 记录/校验所有分支的入口/出口 kind 一致
            if i == 0 {
                if bi == 0 {
                    in_kind = Some(node_in);
                } else if in_kind != Some(node_in) {
                    return Err(PyValueError::new_err("各分支 pipeline 的 input_kind 必须一致"));
                }
            }
            if i == nlen - 1 {
                if bi == 0 {
                    out_kind = Some(node_out);
                } else if out_kind != Some(node_out) {
                    return Err(PyValueError::new_err("各分支 pipeline 的 output_kind 必须一致"));
                }
            }

            branch.push(nb.take_inner()?);
        }
        branches.push(branch);
    }

    let in_kind = in_kind.ok_or_else(|| PyValueError::new_err("pipelines 不能为空"))?;
    let out_kind = out_kind.ok_or_else(|| PyValueError::new_err("pipelines 不能为空"))?;
    let reducer = PyReduceFn::new(reduce, out_kind);

    Ok(DynNodePy::new_boxed(Box::new(LazyAsyncForkJoinNode::new(
        name, in_kind, out_kind, branches, reducer,
    ))))
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

        // Python 侧对象（尤其是 #[pyclass(unsendable)]）不能跨线程使用。
        // 这里必须使用 current-thread runtime，确保所有 tokio 任务都在同一 OS 线程上 poll。
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio Runtime init failed: {e}")))?;
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

    /// 推入一帧输入。
    #[pyo3(signature = (buf=None))]
    fn push(&mut self, py: Python<'_>, buf: Option<Py<NodeBufferPy>>) -> PyResult<()> {
        let Some(buf) = buf else {
            return self.p.flush().map_err(map_codec_err);
        };

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

    /// reset：从起点向终点 reset，直到完成。
    ///
    /// - force=false：不强制打断正在处理的 flow（等它处理到边界后再 reset）
    /// - force=true：强制 reset（丢弃内部缓存/残留）
    #[pyo3(signature = (force=false))]
    fn reset(&mut self, py: Python<'_>, force: bool) -> PyResult<()> {
        let fut = self.p.reset(force);
        let res = py.allow_threads(|| {
            let _guard = self.rt.enter();
            self.rt.block_on(fut)
        });
        res.map_err(map_codec_err)
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
            let nb_py: Py<NodeBufferPy> = ret.extract().map_err(|_| RunnerError::InvalidData("Python source.pull() 必须返回 NodeBuffer 或 None"))?;
            let mut nb = nb_py.bind(py).borrow_mut();
            let inner = nb.take_inner().map_err(pyerr_to_runner_err)?;
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

/// Python 侧 AsyncPipelineAudioSink（builder）：`processors (PCM->PCM)* -> writer`，每段 processor 都是并行 stage（tokio + spawn_blocking）。
///
/// 注意：
/// - 该对象本身不提供 `push/finalize` 的直接写入能力；
/// - 设计为**传入 `AsyncDynRunner` 作为 sink**，Runner 会把它 move 到 Rust runtime 中执行。
#[pyclass(name = "AsyncPipelineAudioSink", unsendable)]
pub struct AsyncPipelineAudioSinkPy {
    parts: Option<AsyncPipelineAudioSinkParts>,
    handle_capacity: usize,
    handle: Option<Py<AsyncPipelineAudioSinkHandlePy>>,
}

struct AsyncPipelineAudioSinkParts {
    nodes: Vec<Box<dyn DynNode>>,
    writer: Box<dyn AudioWriter + Send>,
    queue_capacity: usize,
}

#[pymethods]
impl AsyncPipelineAudioSinkPy {
    /// 构造：绑定一个最终 writer，并可选绑定一组 processors（按顺序执行）。
    ///
    /// 绑定后传入的 `AudioFileWriter` / `Processor` 会被“搬空”，不可再使用。
    #[new]
    #[pyo3(signature = (writer, nodes, queue_capacity=8, handle_capacity=32))]
    fn new(
        py: Python<'_>,
        writer: Py<AudioFileWriterPy>,
        nodes: Vec<PyObject>,
        queue_capacity: usize,
        handle_capacity: usize,
    ) -> PyResult<Self> {
        let rs_writer = {
            let mut ww = writer.bind(py).borrow_mut();
            ww.take_rs_writer()?
        };

        let mut ns: Vec<Box<dyn DynNode>> = Vec::with_capacity(nodes.len());
        for obj in nodes.into_iter() {
            let any = obj.bind(py);
            // 1) DynNode
            if let Ok(n) = any.extract::<Py<DynNodePy>>() {
                let mut nb = n.bind(py).borrow_mut();
                ns.push(nb.take_inner()?);
                continue;
            }
            // 2) Processor -> auto wrap as DynNode (PCM->PCM)
            if let Ok(p) = any.extract::<Py<ProcessorPy>>() {
                let mut pp = p.bind(py).borrow_mut();
                let rs_p = pp.take_rs_processor()?;
                ns.push(Box::new(BoxedProcessorNode::new(rs_p)));
                continue;
            }
            return Err(PyValueError::new_err(
                "nodes 仅支持 DynNode 或 Processor（Encoder/Decoder 请使用 make_encoder_node/make_decoder_node）",
            ));
        }

        Ok(Self {
            parts: Some(AsyncPipelineAudioSinkParts {
                nodes: ns,
                writer: Box::new(rs_writer),
                queue_capacity,
            }),
            handle_capacity: handle_capacity.max(1),
            handle: None,
        })
    }

    /// 启动一个可直接 `push/finalize` 的句柄（内部后台线程运行 tokio runtime）。
    ///
    /// - `handle_capacity`: 句柄输入队列容量（>=1）。越大吞吐更高、延迟/内存越大；满了会阻塞 `push()`（背压）。
    #[pyo3(signature = (handle_capacity=32))]
    fn start(&mut self, handle_capacity: usize) -> PyResult<AsyncPipelineAudioSinkHandlePy> {
        let parts = self
            .parts
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("AsyncPipelineAudioSink is already taken"))?;

        let cap = handle_capacity.max(1);
        let (tx, rx) = std_mpsc::sync_channel::<SinkHandleCmd>(cap);
        let err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let err2 = err.clone();

        let join = std::thread::spawn(move || {
            let rt = match Builder::new_current_thread().enable_all().build() {
                Ok(rt) => rt,
                Err(e) => {
                    let mut g = err2.lock().expect("err_store poisoned");
                    *g = Some(format!("tokio Runtime init failed: {e}"));
                    return;
                }
            };
            let _guard = rt.enter();

            // Build the real Rust async sink inside a *running* runtime context.
            // This is required because `AsyncPipelineAudioSink::new` spawns tasks immediately.
            let mut sink = match rt.block_on(async move {
                RsAsyncPipelineAudioSink::new(parts.nodes, parts.writer, parts.queue_capacity)
            }) {
                Ok(v) => v,
                Err(e) => {
                    let mut g = err2.lock().expect("err_store poisoned");
                    *g = Some(e.to_string());
                    return;
                }
            };

            while let Ok(cmd) = rx.recv() {
                match cmd {
                    SinkHandleCmd::Data(buf) => {
                        if let Err(e) = rt.block_on(sink.push(buf)) {
                            let mut g = err2.lock().expect("err_store poisoned");
                            *g = Some(e.to_string());
                            break;
                        }
                    }
                    SinkHandleCmd::Finalize => {
                        if let Err(e) = rt.block_on(sink.finalize()) {
                            let mut g = err2.lock().expect("err_store poisoned");
                            *g = Some(e.to_string());
                        }
                        break;
                    }
                }
            }
        });

        Ok(AsyncPipelineAudioSinkHandlePy {
            tx: Some(tx),
            join: Some(join),
            err,
        })
    }

    /// stop(): 关闭内部 handle（等价 finalize + join）。
    fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(h) = self.handle.take() else {
            return Ok(());
        };
        let mut hh = h.bind(py).borrow_mut();
        hh.finalize()
    }

    /// 支持 `with`：进入时自动 start，返回 handle。
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(h) = &self.handle {
            return Ok(h.to_object(py));
        }
        let h = self.start(self.handle_capacity)?;
        let h = Py::new(py, h)?;
        self.handle = Some(h.clone_ref(py));
        Ok(h.to_object(py))
    }

    /// 支持 `with`：退出时自动 stop（不吞异常）。
    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<PyObject>,
        _exc: Option<PyObject>,
        _tb: Option<PyObject>,
    ) -> PyResult<bool> {
        let _ = self.stop(py);
        Ok(false)
    }

    /// 为了类型兼容（AudioSink），这里提供同名方法，但不支持直接调用。
    fn push(&mut self, _py: Python<'_>, _buf: Py<NodeBufferPy>) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "AsyncPipelineAudioSink cannot be pushed directly；please pass it to AsyncDynRunner as a sink",
        ))
    }

    /// 为了类型兼容（AudioSink），这里提供同名方法，但不支持直接调用。
    fn finalize(&mut self, _py: Python<'_>) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "AsyncPipelineAudioSink cannot be finalized directly；please pass it to AsyncDynRunner as a sink",
        ))
    }
}

impl AsyncPipelineAudioSinkPy {
    /// 仅取出构建参数（move），不在当前线程构建/spawn。
    ///
    /// 真实的 Rust async sink 必须在 tokio runtime 上下文中构建（会 spawn stage 任务）。
    fn take_parts(&mut self) -> PyResult<AsyncPipelineAudioSinkParts> {
        self.parts
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("AsyncPipelineAudioSink is already taken"))
    }
}

enum SinkHandleCmd {
    Data(NodeBuffer),
    Finalize,
}

/// 可直接 `push/finalize` 的句柄：内部用后台线程 + tokio runtime 驱动真正的 async sink。
#[pyclass(name = "AsyncPipelineAudioSinkHandle", unsendable)]
pub struct AsyncPipelineAudioSinkHandlePy {
    tx: Option<std_mpsc::SyncSender<SinkHandleCmd>>,
    join: Option<std::thread::JoinHandle<()>>,
    err: Arc<Mutex<Option<String>>>,
}

#[pymethods]
impl AsyncPipelineAudioSinkHandlePy {
    /// push(buf: NodeBuffer)
    fn push(&mut self, py: Python<'_>, buf: Py<NodeBufferPy>) -> PyResult<()> {
        if let Some(msg) = self.err.lock().ok().and_then(|g| g.clone()) {
            return Err(PyRuntimeError::new_err(msg));
        }

        let inner_buf = {
            let mut b = buf.bind(py).borrow_mut();
            b.take_inner()?
        };

        let Some(tx) = self.tx.as_ref() else {
            return Err(PyRuntimeError::new_err("AsyncPipelineAudioSinkHandle is closed"));
        };
        tx.send(SinkHandleCmd::Data(inner_buf))
            .map_err(|_| PyRuntimeError::new_err("AsyncPipelineAudioSinkHandle channel closed"))?;
        Ok(())
    }

    fn finalize(&mut self) -> PyResult<()> {
        if self.tx.is_none() && self.join.is_none() {
            return Ok(());
        }
        if let Some(msg) = self.err.lock().ok().and_then(|g| g.clone()) {
            return Err(PyRuntimeError::new_err(msg));
        }

        if let Some(tx) = self.tx.take() {
            let _ = tx.send(SinkHandleCmd::Finalize);
        }
        if let Some(j) = self.join.take() {
            j.join()
                .map_err(|_| PyRuntimeError::new_err("AsyncPipelineAudioSinkHandle worker panicked"))?;
        }
        if let Some(msg) = self.err.lock().ok().and_then(|g| g.clone()) {
            return Err(PyRuntimeError::new_err(msg));
        }
        Ok(())
    }
}

impl Drop for AsyncPipelineAudioSinkHandlePy {
    fn drop(&mut self) {
        // Best-effort shutdown; ignore errors during GC.
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(SinkHandleCmd::Finalize);
        }
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Python 侧 AsyncParallelAudioSink（builder）：fan-out 多个 sink 并发执行（tokio spawn）。
///
/// 注意：
/// - 该对象本身不提供 `push/finalize` 的直接写入能力；
/// - 设计为**传入 `AsyncDynRunner` 作为 sink**，Runner 会把它 move 到 Rust runtime 中执行。
#[pyclass(name = "AsyncParallelAudioSink", unsendable)]
pub struct AsyncParallelAudioSinkPy {
    sinks: Option<Vec<PyObject>>,
    handle_capacity: usize,
    handle: Option<Py<AsyncParallelAudioSinkHandlePy>>,
}

#[pymethods]
impl AsyncParallelAudioSinkPy {
    #[new]
    #[pyo3(signature = (sinks, handle_capacity=32))]
    fn new(sinks: Vec<PyObject>, handle_capacity: usize) -> PyResult<Self> {
        if sinks.is_empty() {
            return Err(PyValueError::new_err("sinks 不能为空"));
        }
        Ok(Self {
            sinks: Some(sinks),
            handle_capacity: handle_capacity.max(1),
            handle: None,
        })
    }

    fn push(&mut self, _py: Python<'_>, _buf: Py<NodeBufferPy>) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "AsyncParallelAudioSink 不能直接 push；请把它作为 sink 传入 AsyncDynRunner",
        ))
    }

    fn finalize(&mut self, _py: Python<'_>) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "AsyncParallelAudioSink 不能直接 finalize；请把它作为 sink 传入 AsyncDynRunner",
        ))
    }

    /// 启动一个可直接 `push/finalize` 的句柄（内部后台线程运行 tokio runtime）。
    ///
    /// - `handle_capacity`: 句柄输入队列容量（>=1）。越大吞吐更高、延迟/内存越大；满了会阻塞 `push()`（背压）。
    #[pyo3(signature = (handle_capacity=32))]
    fn start(&mut self, py: Python<'_>, handle_capacity: usize) -> PyResult<AsyncParallelAudioSinkHandlePy> {
        // 在当前线程（持有 GIL）只“取出构建参数/对象引用”，不要在这里构建会 spawn 的 Rust sink。
        enum SinkSpec {
            Pipeline(AsyncPipelineAudioSinkParts),
            Parallel(Vec<SinkSpec>),
            Py(Py<PyAny>),
        }

        fn collect_specs(py: Python<'_>, obj: PyObject) -> PyResult<Vec<SinkSpec>> {
            let any = obj.bind(py);

            // 1) AsyncPipelineAudioSink -> take parts (build later in runtime thread)
            if let Ok(p) = any.extract::<Py<AsyncPipelineAudioSinkPy>>() {
                let mut pp = p.bind(py).borrow_mut();
                let parts = pp.take_parts()?;
                return Ok(vec![SinkSpec::Pipeline(parts)]);
            }

            // 2) Nested AsyncParallelAudioSink -> collect nested specs
            if let Ok(p) = any.extract::<Py<AsyncParallelAudioSinkPy>>() {
                let mut pp = p.bind(py).borrow_mut();
                let nested = pp.take_sinks()?;
                let mut outs: Vec<SinkSpec> = Vec::new();
                for n in nested {
                    outs.extend(collect_specs(py, n)?);
                }
                return Ok(vec![SinkSpec::Parallel(outs)]);
            }

            // 3) Fallback: treat as Python callback sink object (push/finalize)
            let py_obj: Py<PyAny> = obj.extract(py)?;
            Ok(vec![SinkSpec::Py(py_obj)])
        }

        let sinks = self.take_sinks()?;
        let mut specs: Vec<SinkSpec> = Vec::new();
        for s in sinks {
            specs.extend(collect_specs(py, s)?);
        }
        if specs.is_empty() {
            return Err(PyValueError::new_err("sinks 不能为空"));
        }

        let cap = handle_capacity.max(1);
        let (tx, rx) = std_mpsc::sync_channel::<SinkHandleCmd>(cap);
        let err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let err2 = err.clone();

        let join = std::thread::spawn(move || {
            let rt = match Builder::new_current_thread().enable_all().build() {
                Ok(rt) => rt,
                Err(e) => {
                    let mut g = err2.lock().expect("err_store poisoned");
                    *g = Some(format!("tokio Runtime init failed: {e}"));
                    return;
                }
            };
            let _guard = rt.enter();

            // Build all sinks inside ONE running runtime context (no nested block_on).
            let mut par = match rt.block_on(async move {
                fn build(spec: SinkSpec) -> Result<Box<dyn AsyncAudioSink<In = NodeBuffer> + Send>, RunnerError> {
                    match spec {
                        SinkSpec::Pipeline(parts) => {
                            let rs = RsAsyncPipelineAudioSink::new(parts.nodes, parts.writer, parts.queue_capacity)?;
                            Ok(Box::new(rs))
                        }
                        SinkSpec::Parallel(children) => {
                            let mut par = RsAsyncParallelAudioSink::<NodeBuffer>::with_capacity(children.len());
                            for c in children {
                                par.bind(build(c)?);
                            }
                            Ok(Box::new(par))
                        }
                        SinkSpec::Py(obj) => Ok(Box::new(PyCallbackSink { obj })),
                    }
                }

                let mut par = RsAsyncParallelAudioSink::<NodeBuffer>::with_capacity(specs.len());
                for s in specs {
                    par.bind(build(s)?);
                }
                Ok::<_, RunnerError>(par)
            }) {
                Ok(v) => v,
                Err(e) => {
                    let mut g = err2.lock().expect("err_store poisoned");
                    *g = Some(e.to_string());
                    return;
                }
            };

            while let Ok(cmd) = rx.recv() {
                match cmd {
                    SinkHandleCmd::Data(buf) => {
                        if let Err(e) = rt.block_on(par.push(buf)) {
                            let mut g = err2.lock().expect("err_store poisoned");
                            *g = Some(e.to_string());
                            break;
                        }
                    }
                    SinkHandleCmd::Finalize => {
                        if let Err(e) = rt.block_on(par.finalize()) {
                            let mut g = err2.lock().expect("err_store poisoned");
                            *g = Some(e.to_string());
                        }
                        break;
                    }
                }
            }
        });

        Ok(AsyncParallelAudioSinkHandlePy {
            tx: Some(tx),
            join: Some(join),
            err,
        })
    }

    /// stop(): 关闭内部 handle（等价 finalize + join）。
    fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(h) = self.handle.take() else {
            return Ok(());
        };
        let mut hh = h.bind(py).borrow_mut();
        hh.finalize()
    }

    /// 支持 `with`：进入时自动 start，返回 handle。
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(h) = &self.handle {
            return Ok(h.to_object(py));
        }
        let h = self.start(py, self.handle_capacity)?;
        let h = Py::new(py, h)?;
        self.handle = Some(h.clone_ref(py));
        Ok(h.to_object(py))
    }

    /// 支持 `with`：退出时自动 stop（不吞异常）。
    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<PyObject>,
        _exc: Option<PyObject>,
        _tb: Option<PyObject>,
    ) -> PyResult<bool> {
        let _ = self.stop(py);
        Ok(false)
    }
}

impl AsyncParallelAudioSinkPy {
    fn take_sinks(&mut self) -> PyResult<Vec<PyObject>> {
        self.sinks
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("AsyncParallelAudioSink is already taken"))
    }
}

/// 可直接 `push/finalize` 的句柄：驱动 `AsyncParallelAudioSink`（后台线程 + tokio runtime）。
#[pyclass(name = "AsyncParallelAudioSinkHandle", unsendable)]
pub struct AsyncParallelAudioSinkHandlePy {
    tx: Option<std_mpsc::SyncSender<SinkHandleCmd>>,
    join: Option<std::thread::JoinHandle<()>>,
    err: Arc<Mutex<Option<String>>>,
}

#[pymethods]
impl AsyncParallelAudioSinkHandlePy {
    fn push(&mut self, py: Python<'_>, buf: Py<NodeBufferPy>) -> PyResult<()> {
        if let Some(msg) = self.err.lock().ok().and_then(|g| g.clone()) {
            return Err(PyRuntimeError::new_err(msg));
        }
        let inner_buf = {
            let mut b = buf.bind(py).borrow_mut();
            b.take_inner()?
        };
        let Some(tx) = self.tx.as_ref() else {
            return Err(PyRuntimeError::new_err("AsyncParallelAudioSinkHandle is closed"));
        };
        tx.send(SinkHandleCmd::Data(inner_buf))
            .map_err(|_| PyRuntimeError::new_err("AsyncParallelAudioSinkHandle channel closed"))?;
        Ok(())
    }

    fn finalize(&mut self) -> PyResult<()> {
        if self.tx.is_none() && self.join.is_none() {
            return Ok(());
        }
        if let Some(msg) = self.err.lock().ok().and_then(|g| g.clone()) {
            return Err(PyRuntimeError::new_err(msg));
        }
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(SinkHandleCmd::Finalize);
        }
        if let Some(j) = self.join.take() {
            j.join()
                .map_err(|_| PyRuntimeError::new_err("AsyncParallelAudioSinkHandle worker panicked"))?;
        }
        if let Some(msg) = self.err.lock().ok().and_then(|g| g.clone()) {
            return Err(PyRuntimeError::new_err(msg));
        }
        Ok(())
    }
}

impl Drop for AsyncParallelAudioSinkHandlePy {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(SinkHandleCmd::Finalize);
        }
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// `AsyncDynRunnerPy` 内部使用的 sink：
/// - 默认：调用 Python 回调对象（同步）
/// - 特判：可直接使用 Rust 的 async sink（例如 `AsyncPipelineAudioSink`），避免 Python 回调开销
enum AnyNodeBufferAsyncSink {
    Py(PyCallbackSink),
    RustPipeline(RsAsyncPipelineAudioSink),
    RustParallel(RsAsyncParallelAudioSink<NodeBuffer>),
}

#[async_trait]
impl AsyncAudioSink for AnyNodeBufferAsyncSink {
    type In = NodeBuffer;

    fn name(&self) -> &'static str {
        match self {
            AnyNodeBufferAsyncSink::Py(s) => AudioSink::name(s),
            AnyNodeBufferAsyncSink::RustPipeline(s) => s.name(),
            AnyNodeBufferAsyncSink::RustParallel(s) => s.name(),
        }
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        match self {
            AnyNodeBufferAsyncSink::Py(s) => AudioSink::push(s, input),
            AnyNodeBufferAsyncSink::RustPipeline(s) => s.push(input).await,
            AnyNodeBufferAsyncSink::RustParallel(s) => s.push(input).await,
        }
    }

    async fn finalize(&mut self) -> RunnerResult<()> {
        match self {
            AnyNodeBufferAsyncSink::Py(s) => AudioSink::finalize(s),
            AnyNodeBufferAsyncSink::RustPipeline(s) => s.finalize().await,
            AnyNodeBufferAsyncSink::RustParallel(s) => s.finalize().await,
        }
    }
}

/// Python 侧 AsyncDynRunner（动态节点列表 + Python Source/Sink）。
#[pyclass(name = "AsyncDynRunner")]
pub struct AsyncDynRunnerPy {
    rt: Runtime,
    runner: AsyncAutoRunner<AsyncDynPipeline, PyCallbackSource, AnyNodeBufferAsyncSink>,
}

#[pymethods]
impl AsyncDynRunnerPy {
    #[new]
    fn new(py: Python<'_>, source: Py<PyAny>, nodes: Vec<Py<DynNodePy>>, sink: Py<PyAny>) -> PyResult<Self> {
        if nodes.is_empty() {
            return Err(PyValueError::new_err("nodes 不能为空"));
        }
        let mut boxed: Vec<Box<dyn DynNode>> = Vec::with_capacity(nodes.len());
        for n in nodes.into_iter() {
            let mut nb = n.bind(py).borrow_mut();
            boxed.push(nb.take_inner()?);
        }

        // Python 侧对象（尤其是 #[pyclass(unsendable)]）不能跨线程使用。
        // AsyncAutoRunner 内部会 spawn 任务；使用 current-thread runtime 可避免被调度到其他 worker 线程。
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio Runtime init failed: {e}")))?;
        let pipeline = rt
            .block_on(async { AsyncDynPipeline::new(boxed) })
            .map_err(map_codec_err)?;

        // sink: 优先识别 Rust 的 AsyncPipelineAudioSink / AsyncParallelAudioSink（可真正 async + 背压）
        let sink_any = {
            let any = sink.bind(py);
            if let Ok(w) = any.extract::<Py<AsyncPipelineAudioSinkPy>>() {
                let mut ww = w.bind(py).borrow_mut();
                let parts = ww.take_parts()?;
                let rs_sink = rt
                    .block_on(async move { RsAsyncPipelineAudioSink::new(parts.nodes, parts.writer, parts.queue_capacity) })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                AnyNodeBufferAsyncSink::RustPipeline(rs_sink)
            } else if let Ok(w) = any.extract::<Py<AsyncParallelAudioSinkPy>>() {
                let mut ww = w.bind(py).borrow_mut();
                let inner_sinks = ww.take_sinks()?;
                let mut par = RsAsyncParallelAudioSink::<NodeBuffer>::with_capacity(inner_sinks.len());

                for s_obj in inner_sinks {
                    let s_any = s_obj.bind(py);

                    // 1) Rust async pipeline sink
                    if let Ok(p) = s_any.extract::<Py<AsyncPipelineAudioSinkPy>>() {
                        let mut pp = p.bind(py).borrow_mut();
                        let parts = pp.take_parts()?;
                        let rs = rt
                            .block_on(async move { RsAsyncPipelineAudioSink::new(parts.nodes, parts.writer, parts.queue_capacity) })
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        par.bind(Box::new(rs));
                        continue;
                    }

                    // 2) Nested async-parallel sink
                    if let Ok(p) = s_any.extract::<Py<AsyncParallelAudioSinkPy>>() {
                        let mut pp = p.bind(py).borrow_mut();
                        let nested = pp.take_sinks()?;
                        let mut nested_par = RsAsyncParallelAudioSink::<NodeBuffer>::with_capacity(nested.len());
                        for nn in nested {
                            let nn_any = nn.bind(py);
                            if let Ok(nn_p) = nn_any.extract::<Py<AsyncPipelineAudioSinkPy>>() {
                                let mut nnp = nn_p.bind(py).borrow_mut();
                                let parts = nnp.take_parts()?;
                                let rs = rt
                                    .block_on(async move { RsAsyncPipelineAudioSink::new(parts.nodes, parts.writer, parts.queue_capacity) })
                                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                                nested_par.bind(Box::new(rs));
                            } else {
                                let nn_py: Py<PyAny> = nn.extract(py)?;
                                nested_par.bind(Box::new(AnyNodeBufferAsyncSink::Py(PyCallbackSink { obj: nn_py })));
                            }
                        }
                        par.bind(Box::new(nested_par));
                        continue;
                    }

                    // 3) Fallback: Python callback sink object
                    let s_py: Py<PyAny> = s_obj.extract(py)?;
                    par.bind(Box::new(AnyNodeBufferAsyncSink::Py(PyCallbackSink { obj: s_py })));
                }

                AnyNodeBufferAsyncSink::RustParallel(par)
            } else {
                AnyNodeBufferAsyncSink::Py(PyCallbackSink { obj: sink })
            }
        };

        let runner = AsyncAutoRunner::new(
            PyCallbackSource { obj: source },
            pipeline,
            sink_any,
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
    #[pyo3(signature = (path, format, chunk_samples=None))]
    fn new(path: String, format: &str, chunk_samples: Option<usize>) -> PyResult<Self> {
        let fmt = file_format_from_str(format).ok_or_else(|| PyValueError::new_err("format 仅支持: wav/mp3/aac_adts/flac/opus_ogg"))?;
        let cfg = match fmt {
            "wav" => {
                let cs = chunk_samples.unwrap_or(1024).max(1);
                AudioFileReadConfig::Wav(rs_file::WavReaderConfig { chunk_samples: cs })
            }
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
    #[pyo3(signature = (layout="planar"))]
    fn next_frame(&mut self, py: Python<'_>, layout: &str) -> PyResult<Option<PyObject>> {
        let planar = match layout.to_ascii_lowercase().as_str() {
            "planar" => true,
            "interleaved" => false,
            _ => return Err(PyValueError::new_err("layout 仅支持: planar/interleaved")),
        };
        match AudioReader::next_frame(&mut self.r).map_err(map_file_err)? {
            Some(f) => Ok(Some(frame_to_numpy(py, &f, planar)?)),
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
    path: String,
    format: String,
    bitrate: Option<u32>,
    compression_level: Option<i32>,
    wav_output_format: Option<String>,
    w: Option<RsAudioFileWriter>,
    input_format: Option<RsAudioFormat>,
    sample_type: Option<SampleType>,
}

#[pymethods]
impl AudioFileWriterPy {
    #[new]
    #[pyo3(signature = (path, format, input_format=None, bitrate=None, compression_level=None, wav_output_format=None))]
    fn new(
        path: String,
        format: &str,
        input_format: Option<AudioFormat>,
        bitrate: Option<u32>,
        compression_level: Option<i32>,
        wav_output_format: Option<String>,
    ) -> PyResult<Self> {
        let fmt = file_format_from_str(format).ok_or_else(|| PyValueError::new_err("format 仅支持: wav/mp3/aac_adts/flac/opus_ogg"))?;
        let (w, rs_fmt, st) = if let Some(input_format) = input_format {
            let rs_fmt = input_format.to_rs()?;
            let st = input_format.sample_type_rs()?;
            let cfg = build_file_writer_cfg(fmt, rs_fmt, bitrate, compression_level, wav_output_format.as_deref())?;
            let w = RsAudioFileWriter::create(path.clone(), cfg).map_err(map_file_err)?;
            (Some(w), Some(rs_fmt), Some(st))
        } else {
            (None, None, None)
        };

        Ok(Self {
            path,
            format: fmt.to_string(),
            bitrate,
            compression_level,
            wav_output_format,
            w,
            input_format: rs_fmt,
            sample_type: st,
        })
    }

    /// 直接写入一帧 PCM（numpy）
    fn write_pcm(&mut self, py: Python<'_>, pcm: &Bound<'_, PyAny>) -> PyResult<()> {
        let Some(sample_type) = self.sample_type else {
            return Err(PyValueError::new_err(
                "input_format=None 时无法从裸 numpy 推断 sample_rate/format；请在构造时提供 input_format，或改用 push(NodeBuffer) 让首帧自动推断",
            ));
        };
        let input_format = self
            .input_format
            .ok_or_else(|| PyValueError::new_err("writer not initialized"))?;

        let dtype_name = match sample_type {
            SampleType::U8 => "uint8",
            SampleType::I16 => "int16",
            SampleType::I32 => "int32",
            SampleType::I64 => "int64",
            SampleType::F32 => "float32",
            SampleType::F64 => "float64",
        };
        let arr_any = ascontig_cast_2d(py, pcm, dtype_name)?;
        let frame = if input_format.is_planar() {
            match sample_type {
                SampleType::U8 => ndarray_to_frame_planar::<u8>(&arr_any, input_format)?,
                SampleType::I16 => ndarray_to_frame_planar::<i16>(&arr_any, input_format)?,
                SampleType::I32 => ndarray_to_frame_planar::<i32>(&arr_any, input_format)?,
                SampleType::I64 => ndarray_to_frame_planar::<i64>(&arr_any, input_format)?,
                SampleType::F32 => ndarray_to_frame_planar::<f32>(&arr_any, input_format)?,
                SampleType::F64 => ndarray_to_frame_planar::<f64>(&arr_any, input_format)?,
            }
        } else {
            match sample_type {
                SampleType::U8 => ndarray_to_frame_interleaved::<u8>(&arr_any, input_format)?,
                SampleType::I16 => ndarray_to_frame_interleaved::<i16>(&arr_any, input_format)?,
                SampleType::I32 => ndarray_to_frame_interleaved::<i32>(&arr_any, input_format)?,
                SampleType::I64 => ndarray_to_frame_interleaved::<i64>(&arr_any, input_format)?,
                SampleType::F32 => ndarray_to_frame_interleaved::<f32>(&arr_any, input_format)?,
                SampleType::F64 => ndarray_to_frame_interleaved::<f64>(&arr_any, input_format)?,
            }
        };
        let w = self.w.as_mut().ok_or_else(|| PyValueError::new_err("writer not initialized"))?;
        AudioWriter::write_frame(w, &frame as &dyn AudioFrameView).map_err(map_file_err)?;
        Ok(())
    }

    /// `AsyncDynRunner` 兼容：push(buf: NodeBuffer)（仅支持 PCM）。
    fn push(&mut self, py: Python<'_>, buf: Py<NodeBufferPy>) -> PyResult<()> {
        let mut b = buf.bind(py).borrow_mut();
        let inner = b.take_inner()?;
        match inner {
            NodeBuffer::Pcm(f) => {
                if self.w.is_none() {
                    let rs_fmt = *f.format_ref();
                    let st = rs_fmt.sample_format.sample_type();
                    let cfg = build_file_writer_cfg(
                        &self.format,
                        rs_fmt,
                        self.bitrate,
                        self.compression_level,
                        self.wav_output_format.as_deref(),
                    )?;
                    let w = RsAudioFileWriter::create(self.path.clone(), cfg).map_err(map_file_err)?;
                    self.w = Some(w);
                    self.input_format = Some(rs_fmt);
                    self.sample_type = Some(st);
                } else if let Some(expected) = self.input_format {
                    if *f.format_ref() != expected {
                        return Err(PyRuntimeError::new_err("PCM format is not consistent with the input_format of the writer"));
                    }
                }

                let w = self.w.as_mut().ok_or_else(|| PyValueError::new_err("writer not initialized"))?;
                AudioWriter::write_frame(w, &f as &dyn AudioFrameView).map_err(map_file_err)?;
                Ok(())
            }
            NodeBuffer::Packet(_) => Err(PyValueError::new_err("AudioFileWriter.push 仅支持 PCM（NodeBuffer kind=pcm）")),
        }
    }

    fn finalize(&mut self) -> PyResult<()> {
        let Some(w) = self.w.as_mut() else {
            return Ok(());
        };
        AudioWriter::finalize(w).map_err(map_file_err)
    }
}

impl AudioFileWriterPy {
    /// 把内部的 Rust `AudioFileWriter` 取出（move）。
    ///
    /// 用于把多个文件 writer 绑定到 `ParallelAudioWriter`。
    /// 被取出后，这个 Python 对象将不再可用（再调用会报错/无效果）。
    fn take_rs_writer(&mut self) -> PyResult<RsAudioFileWriter> {
        self.w
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("AudioFileWriter is not initialized or already taken by ParallelAudioWriter"))
    }
}

/// Python 侧 ParallelAudioWriter：把多个 Rust `AudioFileWriter` 绑定成一个，并行写出。
///
/// 注意：当前仅支持绑定 *已初始化* 的 `AudioFileWriter`（构造时传入 input_format 的那种）。
#[pyclass(name = "ParallelAudioWriter", unsendable)]
pub struct ParallelAudioWriterPy {
    inner: RsParallelAudioWriter,
}

#[pymethods]
impl ParallelAudioWriterPy {
    /// 构造：从多个 `AudioFileWriter` 绑定得到一个并行 writer。
    ///
    /// 绑定后传入的 `AudioFileWriter` 会被“搬空”，不可再使用。
    #[new]
    fn new(py: Python<'_>, writers: Vec<PyObject>) -> PyResult<Self> {
        let mut inner = RsParallelAudioWriter::with_capacity(writers.len());
        for w in writers {
            let boxed = take_any_audio_writer(py, w)?;
            inner.bind(boxed);
        }
        Ok(Self { inner })
    }

    /// 追加绑定一个 `AudioFileWriter`。
    fn bind(&mut self, py: Python<'_>, writer: PyObject) -> PyResult<()> {
        let boxed = take_any_audio_writer(py, writer)?;
        self.inner.bind(boxed);
        Ok(())
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// `AsyncDynRunner` 兼容：push(buf: NodeBuffer)（仅支持 PCM）。
    fn push(&mut self, py: Python<'_>, buf: Py<NodeBufferPy>) -> PyResult<()> {
        let inner_buf = {
            let mut b = buf.bind(py).borrow_mut();
            b.take_inner()?
        };

        match inner_buf {
            NodeBuffer::Pcm(f) => {
                py.allow_threads(|| self.inner.write_frame(&f as &dyn AudioFrameView))
                    .map_err(map_file_err)?;
                Ok(())
            }
            NodeBuffer::Packet(_) => Err(PyValueError::new_err("ParallelAudioWriter.push 仅支持 PCM（NodeBuffer kind=pcm）")),
        }
    }

    fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| AudioWriter::finalize(&mut self.inner)).map_err(map_file_err)
    }
}

fn take_any_audio_writer(py: Python<'_>, obj: PyObject) -> PyResult<Box<dyn AudioWriter + Send>> {
    let any = obj.bind(py);

    // 1) AudioFileWriter
    if let Ok(w) = any.extract::<Py<AudioFileWriterPy>>() {
        let mut ww = w.bind(py).borrow_mut();
        let rs_w = ww.take_rs_writer()?;
        return Ok(Box::new(rs_w));
    }

    // 2) LineAudioWriter
    if let Ok(w) = any.extract::<Py<LineAudioWriterPy>>() {
        let mut ww = w.bind(py).borrow_mut();
        let rs_w = ww.take_rs_writer()?;
        return Ok(Box::new(rs_w));
    }

    Err(PyValueError::new_err(
        "ParallelAudioWriter only supports AudioFileWriter / LineAudioWriter",
    ))
}

/// Python 侧 LineAudioWriter：`processors (PCM->PCM)* -> AudioWriter`。
///
/// - 目前 final writer 仅支持 `AudioFileWriter`（会被 move/搬空）。
/// - processors 支持 `Processor`（会被 move/搬空）。
#[pyclass(name = "LineAudioWriter", unsendable)]
pub struct LineAudioWriterPy {
    inner: Option<RsLineAudioWriter>,
}

#[pymethods]
impl LineAudioWriterPy {
    /// 构造：绑定一个最终 writer，并可选绑定一组 processors（按顺序执行）。
    ///
    /// 绑定后传入的 `AudioFileWriter` / `Processor` 会被“搬空”，不可再使用。
    #[new]
    #[pyo3(signature = (writer, processors=None))]
    fn new(py: Python<'_>, writer: Py<AudioFileWriterPy>, processors: Option<Vec<Py<ProcessorPy>>>) -> PyResult<Self> {
        let rs_writer = {
            let mut ww = writer.bind(py).borrow_mut();
            ww.take_rs_writer()?
        };

        let mut ps: Vec<Box<dyn crate::codec::processor::processor_interface::AudioProcessor>> = Vec::new();
        if let Some(procs) = processors {
            ps.reserve(procs.len());
            for p in procs {
                let mut pp = p.bind(py).borrow_mut();
                ps.push(pp.take_rs_processor()?);
            }
        }

        Ok(Self {
            inner: Some(RsLineAudioWriter::with_processors(ps, Box::new(rs_writer))),
        })
    }

    /// 追加绑定一个 processor（追加到链末端）。
    ///
    /// 传入的 `Processor` 会被“搬空”，不可再使用。
    fn add_processor(&mut self, py: Python<'_>, p: Py<ProcessorPy>) -> PyResult<()> {
        let mut pp = p.bind(py).borrow_mut();
        let rs_p = pp.take_rs_processor()?;
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("LineAudioWriter is already taken"))?;
        inner.push_processor(rs_p);
        Ok(())
    }

    /// `AsyncDynRunner` 兼容：push(buf: NodeBuffer)（仅支持 PCM）。
    fn push(&mut self, py: Python<'_>, buf: Py<NodeBufferPy>) -> PyResult<()> {
        let inner_buf = {
            let mut b = buf.bind(py).borrow_mut();
            b.take_inner()?
        };

        match inner_buf {
            NodeBuffer::Pcm(f) => {
                let inner = self
                    .inner
                    .as_mut()
                    .ok_or_else(|| PyRuntimeError::new_err("LineAudioWriter is already taken"))?;
                // 末端 writer 现在要求 Send，可安全 allow_threads
                py.allow_threads(|| inner.write_frame(&f as &dyn AudioFrameView))
                    .map_err(map_file_err)?;
                Ok(())
            }
            NodeBuffer::Packet(_) => Err(PyValueError::new_err("LineAudioWriter.push 仅支持 PCM（NodeBuffer kind=pcm）")),
        }
    }

    fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(());
        };
        py.allow_threads(|| AudioWriter::finalize(inner)).map_err(map_file_err)
    }
}

impl LineAudioWriterPy {
    /// 把内部 Rust `LineAudioWriter` 取出（move）。
    ///
    /// 用于把 `LineAudioWriter` 绑定到 `ParallelAudioWriter`。
    /// 被取出后，这个 Python 对象将不再可用。
    fn take_rs_writer(&mut self) -> PyResult<RsLineAudioWriter> {
        self.inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("LineAudioWriter is already taken"))
    }
}

fn build_file_writer_cfg(
    fmt: &str,
    rs_fmt: RsAudioFormat,
    bitrate: Option<u32>,
    compression_level: Option<i32>,
    wav_output_format: Option<&str>,
) -> PyResult<AudioFileWriteConfig> {
    Ok(match fmt {
        "wav" => {
            let out_fmt = match wav_output_format {
                None | Some("pcm16le") => rs_file::WavOutputSampleFormat::Pcm16Le,
                Some("f32le") => rs_file::WavOutputSampleFormat::Float32Le,
                Some(_) => return Err(PyValueError::new_err("wav_output_format 仅支持: pcm16le/f32le")),
            };
            let c = match out_fmt {
                rs_file::WavOutputSampleFormat::Pcm16Le => rs_file::WavWriterConfig::pcm16le(rs_fmt),
                rs_file::WavOutputSampleFormat::Float32Le => rs_file::WavWriterConfig::f32le(rs_fmt),
            };
            AudioFileWriteConfig::Wav(c)
        }
        "mp3" => {
            let mut enc_cfg = crate::codec::encoder::mp3_encoder::Mp3EncoderConfig::new(rs_fmt);
            if let Some(br) = bitrate {
                enc_cfg.bitrate = Some(br);
            }
            AudioFileWriteConfig::Mp3(rs_file::Mp3WriterConfig { encoder: enc_cfg })
        }
        "aac_adts" => AudioFileWriteConfig::AacAdts(rs_file::AacAdtsWriterConfig {
            encoder: AacEncoderConfig {
                input_format: Some(rs_fmt),
                bitrate,
            },
        }),
        "flac" => AudioFileWriteConfig::Flac(rs_file::FlacWriterConfig {
            encoder: FlacEncoderConfig {
                input_format: Some(rs_fmt),
                compression_level,
            },
        }),
        "opus_ogg" => {
            if rs_fmt.sample_rate != 48_000 {
                return Err(PyValueError::new_err("opus_ogg writer 需要 48kHz input_format（请先重采样）"));
            }
            if rs_fmt.sample_format.is_planar() {
                return Err(PyValueError::new_err("opus_ogg writer 需要 interleaved samples（input_format.planar=False）"));
            }
            AudioFileWriteConfig::OpusOgg(rs_file::OpusOggWriterConfig {
                encoder: OpusEncoderConfig {
                    input_format: Some(rs_fmt),
                    bitrate,
                },
            })
        }
        _ => return Err(PyValueError::new_err("unsupported format")),
    })
}


