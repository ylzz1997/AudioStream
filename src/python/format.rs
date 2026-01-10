#![allow(unsafe_op_in_unsafe_fn)]

use crate::common::audio::audio::{
    AudioError, AudioFormat as RsAudioFormat, AudioFrame, AudioFrameView, ChannelLayout, Rational,
    SampleFormat, SampleType,
};

use numpy::{Element, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::python::errors::map_audio_err;

pub(crate) fn sample_format_from_type(t: SampleType, planar: bool) -> SampleFormat {
    match t {
        SampleType::U8 => SampleFormat::U8 { planar },
        SampleType::I16 => SampleFormat::I16 { planar },
        SampleType::I32 => SampleFormat::I32 { planar },
        SampleType::I64 => SampleFormat::I64 { planar },
        SampleType::F32 => SampleFormat::F32 { planar },
        SampleType::F64 => SampleFormat::F64 { planar },
    }
}

pub(crate) fn parse_sample_type(s: &str) -> Option<SampleType> {
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

pub(crate) fn codec_from_str(s: &str) -> Option<&'static str> {
    match s.to_ascii_lowercase().as_str() {
        "wav" | "pcm" => Some("wav"),
        "mp3" => Some("mp3"),
        "aac" => Some("aac"),
        "opus" => Some("opus"),
        "flac" => Some("flac"),
        _ => None,
    }
}

pub(crate) fn warnings_warn(py: Python<'_>, msg: &str) -> PyResult<()> {
    let warnings = py.import_bound("warnings")?;
    warnings.call_method1("warn", (msg,))?;
    Ok(())
}

pub(crate) fn packet_time_base_from_den(den: i32) -> Rational {
    Rational::new(1, den)
}

pub(crate) fn sample_type_to_str(st: SampleType) -> &'static str {
    match st {
        SampleType::U8 => "u8",
        SampleType::I16 => "i16",
        SampleType::I32 => "i32",
        SampleType::I64 => "i64",
        SampleType::F32 => "f32",
        SampleType::F64 => "f64",
    }
}

pub(crate) fn audio_format_from_rs(fmt: RsAudioFormat) -> AudioFormat {
    AudioFormat {
        sample_rate: fmt.sample_rate,
        channels: fmt.channels(),
        sample_type: sample_type_to_str(fmt.sample_format.sample_type()).to_string(),
        planar: fmt.is_planar(),
        channel_layout_mask: fmt.channel_layout.mask,
    }
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
    pub(crate) fn to_rs(&self) -> PyResult<RsAudioFormat> {
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

    pub(crate) fn sample_type_rs(&self) -> PyResult<SampleType> {
        parse_sample_type(&self.sample_type).ok_or_else(|| PyValueError::new_err("invalid sample_type"))
    }
}

pub(crate) fn ascontig_cast_2d<'py>(
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

pub(crate) fn ndarray_to_frame_planar<'py, T: Copy + Element>(
    arr_any: &Bound<'py, PyAny>,
    fmt: RsAudioFormat,
) -> PyResult<AudioFrame> {
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

    AudioFrame::from_planes(fmt, ns, Rational::new(1, fmt.sample_rate as i32), None, planes).map_err(map_audio_err)
}

pub(crate) fn ndarray_to_frame_interleaved<'py, T: Copy + Element>(
    arr_any: &Bound<'py, PyAny>,
    fmt: RsAudioFormat,
) -> PyResult<AudioFrame> {
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

    AudioFrame::from_planes(
        fmt,
        ns,
        Rational::new(1, fmt.sample_rate as i32),
        None,
        vec![out],
    )
    .map_err(map_audio_err)
}

pub(crate) fn ensure_planar_frame(frame: AudioFrame) -> Result<AudioFrame, AudioError> {
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
    let src = frame
        .planes_ref()
        .get(0)
        .ok_or(AudioError::InvalidPlaneCount {
            expected: 1,
            actual: 0,
        })?;
    let mut planes: Vec<Vec<u8>> = vec![vec![0u8; ns * bps]; ch];
    for s in 0..ns {
        for c in 0..ch {
            let src_off = (s * ch + c) * bps;
            let dst_off = s * bps;
            planes[c][dst_off..dst_off + bps].copy_from_slice(&src[src_off..src_off + bps]);
        }
    }
    AudioFrame::from_planes(planar_fmt, ns, frame.time_base(), frame.pts(), planes)
}

pub(crate) fn frame_to_numpy_planar<'py>(py: Python<'py>, frame: &AudioFrame) -> PyResult<PyObject> {
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
                    let src_b = frame
                        .plane(c)
                        .ok_or_else(|| PyRuntimeError::new_err("missing plane"))?;
                    let src = unsafe { std::slice::from_raw_parts(src_b.as_ptr() as *const $t, ns) };
                    for s in 0..ns {
                        rw[[c, s]] = src[s];
                    }
                }
            } else {
                let src_b = frame
                    .plane(0)
                    .ok_or_else(|| PyRuntimeError::new_err("missing plane 0"))?;
                let src = unsafe { std::slice::from_raw_parts(src_b.as_ptr() as *const $t, ns * ch) };
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

pub(crate) fn frame_to_numpy_interleaved<'py>(py: Python<'py>, frame: &AudioFrame) -> PyResult<PyObject> {
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
                        let src_b = frame
                            .plane(c)
                            .ok_or_else(|| PyRuntimeError::new_err("missing plane"))?;
                        let src = unsafe { std::slice::from_raw_parts(src_b.as_ptr() as *const $t, ns) };
                        rw[[s, c]] = src[s];
                    }
                }
            } else {
                let src_b = frame
                    .plane(0)
                    .ok_or_else(|| PyRuntimeError::new_err("missing plane 0"))?;
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

pub(crate) fn frame_to_numpy<'py>(py: Python<'py>, frame: &AudioFrame, planar: bool) -> PyResult<PyObject> {
    if planar {
        frame_to_numpy_planar(py, frame)
    } else {
        frame_to_numpy_interleaved(py, frame)
    }
}


