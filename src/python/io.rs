#![allow(unsafe_op_in_unsafe_fn)]

use crate::codec::encoder::aac_encoder::AacEncoderConfig;
use crate::codec::encoder::opus_encoder::OpusEncoderConfig;
use crate::codec::error::CodecError;
use crate::codec::packet::{CodecPacket, PacketFlags};
use crate::common::audio::audio::{AudioFormat as RsAudioFormat, AudioFrameView, AudioFrameViewMut, Rational, SampleType};
use crate::common::io::file as rs_file;
use crate::common::io::file::{
    AudioFileReadConfig, AudioFileReader as RsAudioFileReader, AudioFileWriteConfig, AudioFileWriter as RsAudioFileWriter,
};
use crate::common::io::io::{AudioReader, AudioWriter};
use crate::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::{AsyncPipeline, NodeBuffer, NodeBufferKind};
use crate::pipeline::node::node_interface::IdentityNode;
use crate::runner::audio_sink::AudioSink;
use crate::runner::audio_source::AudioSource;
use crate::runner::async_auto_runner::AsyncAutoRunner;
use crate::runner::async_runner_interface::AsyncRunner;
use crate::runner::error::{RunnerError, RunnerResult};

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tokio::runtime::Runtime;

use crate::python::errors::{map_codec_err, map_file_err, map_runner_err, pyerr_to_codec_err, pyerr_to_runner_err};
use crate::python::format::{
    ascontig_cast_2d, audio_format_from_rs, frame_to_numpy, ndarray_to_frame_interleaved, ndarray_to_frame_planar,
    AudioFormat,
};

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

/// 创建一个 Python 自定义节点（DynNode），可用于 `AsyncDynPipeline`/`AsyncDynRunner` 的 nodes 列表。
#[pyfunction]
#[pyo3(signature = (obj, input_kind, output_kind, name="py-node".to_string()))]
pub fn make_python_node(obj: Py<PyAny>, input_kind: &str, output_kind: &str, name: String) -> PyResult<DynNodePy> {
    let in_k = node_kind_from_str(input_kind).ok_or_else(|| PyValueError::new_err("input_kind 仅支持: pcm/packet"))?;
    let out_k = node_kind_from_str(output_kind).ok_or_else(|| PyValueError::new_err("output_kind 仅支持: pcm/packet"))?;
    Ok(DynNodePy::new_boxed(Box::new(PyCallbackNode::new(obj, in_k, out_k, name))))
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

/// Python 侧 AsyncDynRunner（动态节点列表 + Python Source/Sink）。
#[pyclass(name = "AsyncDynRunner")]
pub struct AsyncDynRunnerPy {
    rt: Runtime,
    runner: AsyncAutoRunner<AsyncDynPipeline, PyCallbackSource, PyCallbackSink>,
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

        let rt = Runtime::new().map_err(|e| PyRuntimeError::new_err(format!("tokio Runtime init failed: {e}")))?;
        let _guard = rt.enter();
        let pipeline = AsyncDynPipeline::new(boxed).map_err(map_codec_err)?;
        let runner = AsyncAutoRunner::new(
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
    w: RsAudioFileWriter,
    input_format: RsAudioFormat,
    sample_type: SampleType,
}

#[pymethods]
impl AudioFileWriterPy {
    #[new]
    #[pyo3(signature = (path, format, input_format, bitrate=None, compression_level=None))]
    fn new(
        path: String,
        format: &str,
        input_format: AudioFormat,
        bitrate: Option<u32>,
        compression_level: Option<i32>,
    ) -> PyResult<Self> {
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

    /// 直接写入一帧 PCM（numpy）
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


