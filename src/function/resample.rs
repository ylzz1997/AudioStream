use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResampleError {
    InvalidSampleRate,
    UnsupportedFormat(&'static str),
    Backend(String),
}

impl fmt::Display for ResampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResampleError::InvalidSampleRate => write!(f, "invalid sample_rate"),
            ResampleError::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
            ResampleError::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for ResampleError {}

pub trait Sample: Copy + Send + Sync + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl Sample for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl Sample for i16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        let v = v.round();
        if v.is_nan() {
            return 0;
        }
        if v > i16::MAX as f32 {
            i16::MAX
        } else if v < i16::MIN as f32 {
            i16::MIN
        } else {
            v as i16
        }
    }
}

impl Sample for i32 {
    #[inline]
    fn to_f32(self) -> f32 {
        // i32 的有效精度会损失一些，但做线性插值通常够用；
        // 真要高精度可改为 f64。
        self as f32
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        let v = v.round();
        if v.is_nan() {
            return 0;
        }
        if v > i32::MAX as f32 {
            i32::MAX
        } else if v < i32::MIN as f32 {
            i32::MIN
        } else {
            v as i32
        }
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn out_len_round(in_len: usize, in_rate: u32, out_rate: u32) -> usize {
    // 四舍五入：让“时长”尽量对齐
    (((in_len as u64 * out_rate as u64) + (in_rate as u64 / 2)) / (in_rate as u64)) as usize
}

/// 线性插值重采样 MONO
pub fn resample_mono_linear<T: Sample>(
    input: &[T],
    in_rate: u32,
    out_rate: u32,
) -> Result<Vec<T>, ResampleError> {
    if in_rate == 0 || out_rate == 0 {
        return Err(ResampleError::InvalidSampleRate);
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if in_rate == out_rate {
        return Ok(input.to_vec());
    }
    if input.len() == 1 {
        // 时长不确定：保守返回 1 个 sample
        return Ok(vec![input[0]]);
    }

    let out_len = out_len_round(input.len(), in_rate, out_rate).max(1);
    let step = in_rate as f64 / out_rate as f64; // src_pos per out sample

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) * step;
        let idx0 = src_pos.floor() as usize;
        let frac = (src_pos - (idx0 as f64)) as f32;

        if idx0 + 1 >= input.len() {
            out.push(input[input.len() - 1]);
            continue;
        }
        let a = input[idx0].to_f32();
        let b = input[idx0 + 1].to_f32();
        out.push(T::from_f32(lerp(a, b, frac)));
    }
    Ok(out)
}

/// 多声道线性重采样
pub fn resample_planar_linear<T: Sample>(
    planes: &[&[T]],
    in_rate: u32,
    out_rate: u32,
) -> Result<Vec<Vec<T>>, ResampleError> {
    if planes.is_empty() {
        return Ok(Vec::new());
    }
    let in_len = planes[0].len();
    for p in planes.iter().skip(1) {
        if p.len() != in_len {
            // 先简单处理：要求每声道长度一致
            return Err(ResampleError::InvalidSampleRate);
        }
    }
    let mut out = Vec::with_capacity(planes.len());
    for &p in planes {
        out.push(resample_mono_linear(p, in_rate, out_rate)?);
    }
    Ok(out)
}

/// 流式线性重采样器
pub struct LinearResampler {
    in_rate: u32,
    out_rate: u32,
    step: f64,          // src samples per output sample
    next_src_pos: f64,  // 绝对“源采样位置”（单位：samples），从 0 开始
    consumed: u64,      // 已累计输入的 samples 数（用于将 chunk 映射到绝对 index）
    last: Option<f32>,  // 上一个 chunk 的最后一个 sample（f32 域）
}

impl LinearResampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Result<Self, ResampleError> {
        if in_rate == 0 || out_rate == 0 {
            return Err(ResampleError::InvalidSampleRate);
        }
        Ok(Self {
            in_rate,
            out_rate,
            step: in_rate as f64 / out_rate as f64,
            next_src_pos: 0.0,
            consumed: 0,
            last: None,
        })
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    pub fn reset(&mut self) {
        self.next_src_pos = 0.0;
        self.consumed = 0;
        self.last = None;
    }

    /// 输入一个 chunk，产出尽可能多的输出。
    pub fn process<T: Sample>(&mut self, input: &[T]) -> Vec<T> {
        if input.is_empty() {
            return Vec::new();
        }

        // 当前 chunk 在“绝对源采样序列”中的 index 范围：
        // - 如果有 last：我们视作在 chunk 前面额外拼了一个样本，其绝对 index = consumed-1
        // - 否则：chunk 的首样本绝对 index = consumed
        let has_last = self.last.is_some();
        let base_index: i64 = if has_last {
            (self.consumed as i64) - 1
        } else {
            self.consumed as i64
        };
        let last_index: i64 = (self.consumed as i64) + (input.len() as i64) - 1;

        let get_f32 = |abs_idx: i64, last: Option<f32>, base: i64, chunk: &[T]| -> f32 {
            // abs_idx 落在 [base, last_index] 内
            let off = (abs_idx - base) as usize;
            if has_last {
                if off == 0 {
                    last.unwrap()
                } else {
                    chunk[off - 1].to_f32()
                }
            } else {
                chunk[off].to_f32()
            }
        };

        let mut out: Vec<T> = Vec::new();

        // 需要 idx0 和 idx0+1 都可读
        loop {
            let idx0 = self.next_src_pos.floor() as i64;
            let idx1 = idx0 + 1;
            if idx1 > last_index {
                break;
            }
            if idx0 < base_index {
                // 理论上不应发生（意味着之前 chunk 没产完就把源数据丢了）
                // 这里选择直接跳到 base_index，避免死循环。
                self.next_src_pos = base_index as f64;
                continue;
            }
            let frac = (self.next_src_pos - (idx0 as f64)) as f32;
            let a = get_f32(idx0, self.last, base_index, input);
            let b = get_f32(idx1, self.last, base_index, input);
            out.push(T::from_f32(lerp(a, b, frac)));
            self.next_src_pos += self.step;
        }

        // 更新 state：累计 consumed + 缓存 last
        self.consumed += input.len() as u64;
        self.last = Some(input[input.len() - 1].to_f32());
        out
    }

    /// 输入结束后尽量补齐尾部输出：用“最后一个 sample”保持值。
    pub fn flush_hold_last<T: Sample>(&mut self) -> Vec<T> {
        let Some(last) = self.last else {
            return Vec::new();
        };
        // 当输入结束时，理论上我们还可以输出到 idx0 <= last_index（不再需要 idx1）。
        // 这里用 last 值填充 idx1。
        let last_index = (self.consumed as i64) - 1;
        let mut out = Vec::new();
        loop {
            let idx0 = self.next_src_pos.floor() as i64;
            if idx0 > last_index {
                break;
            }
            let frac = (self.next_src_pos - (idx0 as f64)) as f32;
            // 没有更多数据了：a/b 都用 last
            let v = lerp(last, last, frac);
            out.push(T::from_f32(v));
            self.next_src_pos += self.step;
        }
        out
    }
}

// 以上实现都不抗混叠，如果要高质量（尤其 downsample），后续建议接 FIR/polyphase 或 FFmpeg swresample

// -----------------------------
// FFmpeg backend (libswresample)
// -----------------------------
//
// 说明：
// - 该后端在 `--features ffmpeg` 下可用
// - 依赖系统已安装并可链接 FFmpeg（含 swresample）
#[cfg(feature = "ffmpeg")]
pub mod ffmpeg_backend {
    use super::ResampleError;
    use crate::common::audio::audio::{
        AudioFormat, AudioFrame, AudioFrameView, ChannelLayout, Rational, SampleFormat,
    };
    use core::ptr;
    use std::ffi::CStr;

    extern crate ffmpeg_sys_next as ff;
    use libc;

    pub struct FfmpegResampler {
        ctx: *mut ff::SwrContext,
        in_fmt: AudioFormat,
        out_fmt: AudioFormat,
    }

    unsafe impl Send for FfmpegResampler {}

    fn ff_err_to_string(err: i32) -> String {
        let mut buf = [0u8; 256];
        unsafe {
            ff::av_strerror(err, buf.as_mut_ptr() as *mut i8, buf.len());
        }
        let cstr = match CStr::from_bytes_until_nul(&buf) {
            Ok(s) => s,
            Err(_) => return format!("ffmpeg error {err}"),
        };
        cstr.to_string_lossy().into_owned()
    }

    fn map_ff_err(err: i32) -> ResampleError {
        ResampleError::Backend(ff_err_to_string(err))
    }

    fn map_sample_format(sf: SampleFormat) -> Result<ff::AVSampleFormat, ResampleError> {
        use ff::AVSampleFormat::*;
        let av = match sf {
            SampleFormat::I16 { planar: false } => AV_SAMPLE_FMT_S16,
            SampleFormat::I16 { planar: true } => AV_SAMPLE_FMT_S16P,
            SampleFormat::I32 { planar: false } => AV_SAMPLE_FMT_S32,
            SampleFormat::I32 { planar: true } => AV_SAMPLE_FMT_S32P,
            SampleFormat::F32 { planar: false } => AV_SAMPLE_FMT_FLT,
            SampleFormat::F32 { planar: true } => AV_SAMPLE_FMT_FLTP,
            SampleFormat::U8 { planar: false } => AV_SAMPLE_FMT_U8,
            SampleFormat::U8 { planar: true } => AV_SAMPLE_FMT_U8P,
            _ => return Err(ResampleError::UnsupportedFormat("only u8/i16/i32/f32 are supported in ffmpeg resampler")),
        };
        Ok(av)
    }

    fn fill_av_channel_layout(dst: &mut ff::AVChannelLayout, layout: ChannelLayout) -> Result<(), ResampleError> {
        unsafe {
            *dst = core::mem::zeroed();
            let channels = layout.channels as i32;
            if channels <= 0 {
                return Err(ResampleError::UnsupportedFormat("invalid channels"));
            }
            if layout.mask != 0 {
                let ret = ff::av_channel_layout_from_mask(dst, layout.mask);
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
            } else {
                ff::av_channel_layout_default(dst, channels);
            }
            Ok(())
        }
    }

    fn rescale_pts(pts: i64, tb_in: Rational, tb_out: Rational) -> Option<i64> {
        if tb_in.den == 0 || tb_out.den == 0 || tb_out.num == 0 {
            return None;
        }
        let num = (pts as i128)
            .checked_mul(tb_in.num as i128)?
            .checked_mul(tb_out.den as i128)?;
        let den = (tb_in.den as i128).checked_mul(tb_out.num as i128)?;
        Some((num / den) as i64)
    }

    impl FfmpegResampler {
        pub fn new(in_fmt: AudioFormat, out_fmt: AudioFormat) -> Result<Self, ResampleError> {
            if in_fmt.sample_rate == 0 || out_fmt.sample_rate == 0 {
                return Err(ResampleError::InvalidSampleRate);
            }

            unsafe {
                let mut in_ch: ff::AVChannelLayout = core::mem::zeroed();
                let mut out_ch: ff::AVChannelLayout = core::mem::zeroed();
                fill_av_channel_layout(&mut in_ch, in_fmt.channel_layout)?;
                fill_av_channel_layout(&mut out_ch, out_fmt.channel_layout)?;

                let in_sf = map_sample_format(in_fmt.sample_format)?;
                let out_sf = map_sample_format(out_fmt.sample_format)?;

                let mut ctx: *mut ff::SwrContext = ptr::null_mut();
                let ret = ff::swr_alloc_set_opts2(
                    &mut ctx as *mut *mut ff::SwrContext,
                    &out_ch as *const ff::AVChannelLayout,
                    out_sf,
                    out_fmt.sample_rate as i32,
                    &in_ch as *const ff::AVChannelLayout,
                    in_sf,
                    in_fmt.sample_rate as i32,
                    0,
                    ptr::null_mut::<libc::c_void>(),
                );
                // 释放临时 AVChannelLayout（FFmpeg 6+ 需要 uninit）
                ff::av_channel_layout_uninit(&mut in_ch);
                ff::av_channel_layout_uninit(&mut out_ch);

                if ret < 0 || ctx.is_null() {
                    if !ctx.is_null() {
                        ff::swr_free(&mut ctx as *mut *mut ff::SwrContext);
                    }
                    return Err(map_ff_err(if ret < 0 { ret } else { -1 }));
                }

                let ret = ff::swr_init(ctx);
                if ret < 0 {
                    ff::swr_free(&mut ctx as *mut *mut ff::SwrContext);
                    return Err(map_ff_err(ret));
                }

                Ok(Self { ctx, in_fmt, out_fmt })
            }
        }

        pub fn in_format(&self) -> AudioFormat {
            self.in_fmt
        }

        pub fn out_format(&self) -> AudioFormat {
            self.out_fmt
        }

        pub fn reset(&mut self) -> Result<(), ResampleError> {
            unsafe {
                ff::swr_close(self.ctx);
                let ret = ff::swr_init(self.ctx);
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
            }
            Ok(())
        }

        /// 输入一个 `AudioFrameView`，输出一个新的 `AudioFrame`（采样率/格式按 out_format）。
        pub fn process_frame(&mut self, input: &dyn AudioFrameView) -> Result<AudioFrame, ResampleError> {
            let fmt = input.format();
            // 目前先要求输入格式完全匹配，避免隐式转换导致上层“不知情”
            if fmt != self.in_fmt {
                return Err(ResampleError::UnsupportedFormat("input AudioFormat mismatch for this resampler"));
            }
            let in_nb = input.nb_samples();
            if in_nb == 0 {
                return Ok(AudioFrame::from_planes(
                    self.out_fmt,
                    0,
                    Rational::new(1, self.out_fmt.sample_rate as i32),
                    None,
                    {
                        let plane_count = AudioFrame::expected_plane_count(&self.out_fmt);
                        let mut planes = Vec::with_capacity(plane_count);
                        for _ in 0..plane_count {
                            planes.push(Vec::new());
                        }
                        planes
                    },
                ).map_err(|_| ResampleError::Backend("failed to build empty AudioFrame".into()))?);
            }

            unsafe {
                let out_tb = Rational::new(1, self.out_fmt.sample_rate as i32);
                let pts = input.pts().and_then(|p| rescale_pts(p, input.time_base(), out_tb));

                // 估算最多输出多少 samples（FFmpeg 会考虑内部 delay）
                let out_max = ff::swr_get_out_samples(self.ctx, in_nb as i32);
                if out_max < 0 {
                    return Err(map_ff_err(out_max));
                }
                let out_max = (out_max as usize).max(1);

                // 准备输入指针
                let in_planes = input.plane_count();
                if in_planes == 0 {
                    return Err(ResampleError::UnsupportedFormat("input plane_count=0"));
                }
                let mut in_ptrs: Vec<*const u8> = Vec::with_capacity(in_planes);
                for i in 0..in_planes {
                    let p = input.plane(i).ok_or(ResampleError::UnsupportedFormat("missing input plane"))?;
                    in_ptrs.push(p.as_ptr());
                }

                // 准备输出 buffer（按 out_max 分配，之后会按实际 out_samples 截断）
                let plane_count = AudioFrame::expected_plane_count(&self.out_fmt);
                let plane_bytes_max = AudioFrame::expected_bytes_per_plane(&self.out_fmt, out_max);
                let mut out_planes: Vec<Vec<u8>> = Vec::with_capacity(plane_count);
                for _ in 0..plane_count {
                    out_planes.push(vec![0u8; plane_bytes_max]);
                }
                let mut out_ptrs: Vec<*mut u8> = out_planes.iter_mut().map(|p| p.as_mut_ptr()).collect();

                // 执行转换
                let ret = ff::swr_convert(
                    self.ctx,
                    out_ptrs.as_mut_ptr(),
                    out_max as i32,
                    in_ptrs.as_ptr(),
                    in_nb as i32,
                );
                if ret < 0 {
                    return Err(map_ff_err(ret));
                }
                let out_nb = ret as usize;

                let plane_bytes = AudioFrame::expected_bytes_per_plane(&self.out_fmt, out_nb);
                for p in out_planes.iter_mut() {
                    p.truncate(plane_bytes);
                }

                AudioFrame::from_planes(self.out_fmt, out_nb, out_tb, pts, out_planes)
                    .map_err(|_| ResampleError::Backend("failed to build AudioFrame".into()))
            }
        }
    }

    impl Drop for FfmpegResampler {
        fn drop(&mut self) {
            unsafe {
                if !self.ctx.is_null() {
                    ff::swr_free(&mut self.ctx as *mut *mut ff::SwrContext);
                }
            }
        }
    }
}

