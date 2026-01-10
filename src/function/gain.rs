use core::fmt;

use crate::common::audio::audio::SampleType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GainError {
    InvalidBuffer(&'static str),
    UnsupportedFormat(&'static str),
}

impl fmt::Display for GainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GainError::InvalidBuffer(msg) => write!(f, "invalid buffer: {msg}"),
            GainError::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
        }
    }
}

impl std::error::Error for GainError {}

#[inline]
fn bytes_per_sample(ty: SampleType) -> usize {
    match ty {
        SampleType::U8 => 1,
        SampleType::I16 => 2,
        SampleType::I32 | SampleType::F32 => 4,
        SampleType::I64 | SampleType::F64 => 8,
    }
}

#[inline]
fn clamp_i64(v: f64) -> i64 {
    if v.is_nan() {
        return 0;
    }
    if v >= i64::MAX as f64 {
        i64::MAX
    } else if v <= i64::MIN as f64 {
        i64::MIN
    } else {
        v.round() as i64
    }
}

#[inline]
fn clamp_i32(v: f64) -> i32 {
    if v.is_nan() {
        return 0;
    }
    if v >= i32::MAX as f64 {
        i32::MAX
    } else if v <= i32::MIN as f64 {
        i32::MIN
    } else {
        v.round() as i32
    }
}

#[inline]
fn clamp_i16(v: f64) -> i16 {
    if v.is_nan() {
        return 0;
    }
    if v >= i16::MAX as f64 {
        i16::MAX
    } else if v <= i16::MIN as f64 {
        i16::MIN
    } else {
        v.round() as i16
    }
}

#[inline]
fn clamp_u8_pcm(v: f64) -> u8 {
    // PCM U8 通常以 128 为“零点”（unsigned offset binary）
    if v.is_nan() {
        return 128;
    }
    if v >= 255.0 {
        255
    } else if v <= 0.0 {
        0
    } else {
        v.round() as u8
    }
}

/// 对原始 PCM bytes 就地应用增益（乘法）。
///
/// - `bytes` 必须按 `sample_type` 对齐（长度为 bytes_per_sample 的整数倍）
/// - `gain` 为线性增益系数（例如 0.5 降 6dB，2.0 升 6dB）
/// - 对整数类型会做**饱和裁剪**
/// - 对 `U8` 采用 offset-binary 语义：`(x-128)*gain + 128`
pub fn apply_gain_bytes_inplace(
    bytes: &mut [u8],
    sample_type: SampleType,
    gain: f64,
) -> Result<(), GainError> {
    let bps = bytes_per_sample(sample_type);
    if bps == 0 {
        return Err(GainError::UnsupportedFormat("bytes_per_sample=0"));
    }
    if bytes.len() % bps != 0 {
        return Err(GainError::InvalidBuffer("buffer size is not aligned to sample size"));
    }

    match sample_type {
        SampleType::U8 => {
            for b in bytes.iter_mut() {
                let x = (*b as i32) - 128;
                let y = (x as f64) * gain;
                let out = clamp_u8_pcm(y + 128.0);
                *b = out;
            }
            Ok(())
        }
        SampleType::I16 => {
            for ch in bytes.chunks_exact_mut(2) {
                let v = i16::from_ne_bytes([ch[0], ch[1]]) as f64;
                let out = clamp_i16(v * gain).to_ne_bytes();
                ch.copy_from_slice(&out);
            }
            Ok(())
        }
        SampleType::I32 => {
            for ch in bytes.chunks_exact_mut(4) {
                let v = i32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]) as f64;
                let out = clamp_i32(v * gain).to_ne_bytes();
                ch.copy_from_slice(&out);
            }
            Ok(())
        }
        SampleType::I64 => {
            for ch in bytes.chunks_exact_mut(8) {
                let v = i64::from_ne_bytes([ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7]]) as f64;
                let out = clamp_i64(v * gain).to_ne_bytes();
                ch.copy_from_slice(&out);
            }
            Ok(())
        }
        SampleType::F32 => {
            for ch in bytes.chunks_exact_mut(4) {
                let v = f32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]) as f64;
                let mut y = v * gain;
                if y.is_nan() {
                    y = 0.0;
                }
                let out = (y as f32).to_ne_bytes();
                ch.copy_from_slice(&out);
            }
            Ok(())
        }
        SampleType::F64 => {
            for ch in bytes.chunks_exact_mut(8) {
                let v = f64::from_ne_bytes([ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7]]);
                let mut y = v * gain;
                if y.is_nan() {
                    y = 0.0;
                }
                let out = y.to_ne_bytes();
                ch.copy_from_slice(&out);
            }
            Ok(())
        }
    }
}


