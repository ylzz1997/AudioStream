use core::fmt;

use crate::common::audio::audio::SampleType;

/// Dynamics压缩器参数。
///
/// 约定：threshold/expansion_threshold 等单位均为 dBFS（0 dBFS == full-scale）。
#[derive(Debug, Clone, Copy)]
pub struct DynamicsParams {
    // --- 用户参数 (User Parameters) ---
    pub sample_rate: f32,
    pub threshold_db: f32,           // 压缩阈值
    pub knee_width_db: f32,          // soft-knee 宽度（dB）
    pub ratio: f32,                  // 压缩比 (比如 4.0)
    pub expansion_ratio: f32,        // 扩展比 (比如 2.0)
    pub expansion_threshold_db: f32, // 扩展/噪声门阈值
    pub attack_time: f32,            // 秒
    pub release_time: f32,           // 秒
    pub master_gain_db: f32,         // 输出增益
}

impl Default for DynamicsParams {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            threshold_db: -18.0,
            knee_width_db: 6.0,
            ratio: 4.0,
            expansion_ratio: 1.0,
            expansion_threshold_db: -80.0,
            attack_time: 0.01,
            release_time: 0.1,
            master_gain_db: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicsError {
    InvalidParameter(&'static str),
    InvalidBuffer(&'static str),
    UnsupportedFormat(&'static str),
}

impl fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynamicsError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            DynamicsError::InvalidBuffer(msg) => write!(f, "invalid buffer: {msg}"),
            DynamicsError::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
        }
    }
}

impl std::error::Error for DynamicsError {}

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
fn db_to_lin(db: f32) -> f32 {
    // 10^(db/20)
    10.0_f32.powf(db * 0.05)
}

#[inline]
fn lin_to_db(lin: f32) -> f32 {
    // 20*log10(lin)
    20.0 * lin.log10()
}

#[inline]
fn attack_release_coeff(time_s: f32, sample_rate: f32) -> f32 {
    // exp(-1/(time*fs)) in (0,1); time<=0 => 0 (instant)
    if !time_s.is_finite() || time_s <= 0.0 || !sample_rate.is_finite() || sample_rate <= 0.0 {
        0.0
    } else {
        (-1.0 / (time_s * sample_rate)).exp()
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

/// 动态压缩/扩展器（带状态：每声道 envelope）。
///
/// 说明：
/// - detector: peak(|x|)
/// - envelope: 一阶 attack/release
/// - curve: expansion/门（低于 expansion_threshold） + compression（高于 threshold，支持 soft-knee）
pub struct DynamicsCompressor {
    params: DynamicsParams,
    attack_coeff: f32,
    release_coeff: f32,
    env: Vec<f32>, // 每声道 envelope（线性幅度，0..）
}

impl DynamicsCompressor {
    pub fn new(params: DynamicsParams, channels: usize) -> Result<Self, DynamicsError> {
        if channels == 0 {
            return Err(DynamicsError::InvalidParameter("channels=0"));
        }
        Self::validate_params(&params)?;
        let attack_coeff = attack_release_coeff(params.attack_time, params.sample_rate);
        let release_coeff = attack_release_coeff(params.release_time, params.sample_rate);
        Ok(Self {
            params,
            attack_coeff,
            release_coeff,
            env: vec![0.0; channels],
        })
    }

    pub fn params(&self) -> DynamicsParams {
        self.params
    }

    pub fn set_params(&mut self, params: DynamicsParams) -> Result<(), DynamicsError> {
        Self::validate_params(&params)?;
        self.params = params;
        self.attack_coeff = attack_release_coeff(params.attack_time, params.sample_rate);
        self.release_coeff = attack_release_coeff(params.release_time, params.sample_rate);
        Ok(())
    }

    pub fn reset(&mut self) {
        for e in self.env.iter_mut() {
            *e = 0.0;
        }
    }

    pub fn channels(&self) -> usize {
        self.env.len()
    }

    fn validate_params(p: &DynamicsParams) -> Result<(), DynamicsError> {
        if !p.sample_rate.is_finite() || p.sample_rate <= 0.0 {
            return Err(DynamicsError::InvalidParameter("sample_rate must be > 0"));
        }
        if !p.threshold_db.is_finite() {
            return Err(DynamicsError::InvalidParameter("threshold_db must be finite"));
        }
        if !p.knee_width_db.is_finite() || p.knee_width_db < 0.0 {
            return Err(DynamicsError::InvalidParameter("knee_width_db must be >= 0"));
        }
        if !p.ratio.is_finite() || p.ratio < 1.0 {
            return Err(DynamicsError::InvalidParameter("ratio must be >= 1"));
        }
        if !p.expansion_ratio.is_finite() || p.expansion_ratio < 1.0 {
            return Err(DynamicsError::InvalidParameter("expansion_ratio must be >= 1"));
        }
        if !p.expansion_threshold_db.is_finite() {
            return Err(DynamicsError::InvalidParameter("expansion_threshold_db must be finite"));
        }
        if !p.attack_time.is_finite() || p.attack_time < 0.0 {
            return Err(DynamicsError::InvalidParameter("attack_time must be >= 0"));
        }
        if !p.release_time.is_finite() || p.release_time < 0.0 {
            return Err(DynamicsError::InvalidParameter("release_time must be >= 0"));
        }
        if !p.master_gain_db.is_finite() {
            return Err(DynamicsError::InvalidParameter("master_gain_db must be finite"));
        }
        Ok(())
    }

    #[inline]
    fn update_env(&self, prev: f32, x: f32) -> f32 {
        // x: detector sample (>=0)
        if x >= prev {
            self.attack_coeff * prev + (1.0 - self.attack_coeff) * x
        } else {
            self.release_coeff * prev + (1.0 - self.release_coeff) * x
        }
    }

    #[inline]
    fn compute_gain_db(&self, level_db: f32) -> f32 {
        let p = self.params;

        // 1) expansion/门（低于 expansion_threshold）
        if level_db < p.expansion_threshold_db && p.expansion_ratio > 1.0 {
            // out = th + (in-th)*ratio
            return (p.expansion_ratio - 1.0) * (level_db - p.expansion_threshold_db);
        }

        // 2) compression（高于 threshold，支持 soft-knee）
        let w = p.knee_width_db;
        if w <= 0.0 {
            if level_db <= p.threshold_db {
                0.0
            } else {
                (1.0 / p.ratio - 1.0) * (level_db - p.threshold_db)
            }
        } else {
            let lower = p.threshold_db - w * 0.5;
            let upper = p.threshold_db + w * 0.5;
            if level_db < lower {
                0.0
            } else if level_db > upper {
                (1.0 / p.ratio - 1.0) * (level_db - p.threshold_db)
            } else {
                // knee region: quadratic
                let x = level_db - p.threshold_db + w * 0.5; // 0..w
                (1.0 / p.ratio - 1.0) * (x * x) / (2.0 * w)
            }
        }
    }

    /// 处理一个 planar 声道（channel_index 对应的 plane），就地修改 bytes。
    pub fn process_planar_channel_bytes_inplace(
        &mut self,
        bytes: &mut [u8],
        sample_type: SampleType,
        channel_index: usize,
    ) -> Result<(), DynamicsError> {
        if channel_index >= self.env.len() {
            return Err(DynamicsError::InvalidParameter("channel_index out of range"));
        }
        let bps = bytes_per_sample(sample_type);
        if bps == 0 {
            return Err(DynamicsError::UnsupportedFormat("bytes_per_sample=0"));
        }
        if bytes.len() % bps != 0 {
            return Err(DynamicsError::InvalidBuffer("buffer size is not aligned to sample size"));
        }
        let nb = bytes.len() / bps;
        self.process_samples_inplace(bytes, sample_type, nb, /*channels=*/ 1, channel_index, true)
    }

    /// 处理 interleaved PCM（单 plane，包含 channels 交错），就地修改 bytes。
    pub fn process_interleaved_bytes_inplace(
        &mut self,
        bytes: &mut [u8],
        sample_type: SampleType,
        channels: usize,
    ) -> Result<(), DynamicsError> {
        if channels == 0 {
            return Err(DynamicsError::InvalidParameter("channels=0"));
        }
        if channels != self.env.len() {
            return Err(DynamicsError::InvalidParameter("channels mismatch"));
        }
        let bps = bytes_per_sample(sample_type);
        if bps == 0 {
            return Err(DynamicsError::UnsupportedFormat("bytes_per_sample=0"));
        }
        if bytes.len() % (bps * channels) != 0 {
            return Err(DynamicsError::InvalidBuffer("interleaved buffer size mismatch"));
        }
        let nb = bytes.len() / (bps * channels);
        self.process_samples_inplace(bytes, sample_type, nb, channels, 0, false)
    }

    fn process_samples_inplace(
        &mut self,
        bytes: &mut [u8],
        sample_type: SampleType,
        nb_samples: usize,
        channels: usize,
        channel_index_for_planar: usize,
        planar_single_channel: bool,
    ) -> Result<(), DynamicsError> {
        let eps = 1e-12_f32;
        let master = self.params.master_gain_db;

        match sample_type {
            SampleType::U8 => {
                if planar_single_channel {
                    let mut env = self.env[channel_index_for_planar];
                    for b in bytes.iter_mut().take(nb_samples) {
                        let v = ((*b as i32) - 128) as f32 / 128.0;
                        let x = v.abs();
                        env = self.update_env(env, x);
                        let level_db = lin_to_db(env.max(eps));
                        let gain_db = self.compute_gain_db(level_db) + master;
                        let g = db_to_lin(gain_db);
                        let y = (v * g * 128.0) as f64 + 128.0;
                        *b = clamp_u8_pcm(y);
                    }
                    self.env[channel_index_for_planar] = env;
                } else {
                    let mut idx = 0usize;
                    for _ in 0..nb_samples {
                        for c in 0..channels {
                            let b = bytes[idx];
                            let v = ((b as i32) - 128) as f32 / 128.0;
                            let x = v.abs();
                            let env_new = self.update_env(self.env[c], x);
                            self.env[c] = env_new;
                            let level_db = lin_to_db(env_new.max(eps));
                            let gain_db = self.compute_gain_db(level_db) + master;
                            let g = db_to_lin(gain_db);
                            let y = (v * g * 128.0) as f64 + 128.0;
                            bytes[idx] = clamp_u8_pcm(y);
                            idx += 1;
                        }
                    }
                }
                Ok(())
            }
            SampleType::I16 => {
                let bps = 2;
                if planar_single_channel {
                    let mut env = self.env[channel_index_for_planar];
                    for ch in bytes.chunks_exact_mut(bps) {
                        let raw = i16::from_ne_bytes([ch[0], ch[1]]);
                        let v = (raw as f32) / 32768.0;
                        let x = v.abs();
                        env = self.update_env(env, x);
                        let level_db = lin_to_db(env.max(eps));
                        let gain_db = self.compute_gain_db(level_db) + master;
                        let g = db_to_lin(gain_db);
                        let y = (v * g * 32768.0) as f64;
                        let out = clamp_i16(y).to_ne_bytes();
                        ch.copy_from_slice(&out);
                    }
                    self.env[channel_index_for_planar] = env;
                } else {
                    let mut i = 0usize;
                    for _ in 0..nb_samples {
                        for c in 0..channels {
                            let base = i * bps;
                            let raw = i16::from_ne_bytes([bytes[base], bytes[base + 1]]);
                            let v = (raw as f32) / 32768.0;
                            let x = v.abs();
                            let env_new = self.update_env(self.env[c], x);
                            self.env[c] = env_new;
                            let level_db = lin_to_db(env_new.max(eps));
                            let gain_db = self.compute_gain_db(level_db) + master;
                            let g = db_to_lin(gain_db);
                            let y = (v * g * 32768.0) as f64;
                            let out = clamp_i16(y).to_ne_bytes();
                            bytes[base..base + 2].copy_from_slice(&out);
                            i += 1;
                        }
                    }
                }
                Ok(())
            }
            SampleType::I32 => {
                let bps = 4;
                if planar_single_channel {
                    let mut env = self.env[channel_index_for_planar];
                    for ch in bytes.chunks_exact_mut(bps) {
                        let raw = i32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]);
                        let v = (raw as f32) / 2147483648.0;
                        let x = v.abs();
                        env = self.update_env(env, x);
                        let level_db = lin_to_db(env.max(eps));
                        let gain_db = self.compute_gain_db(level_db) + master;
                        let g = db_to_lin(gain_db);
                        let y = (v * g * 2147483648.0) as f64;
                        let out = clamp_i32(y).to_ne_bytes();
                        ch.copy_from_slice(&out);
                    }
                    self.env[channel_index_for_planar] = env;
                } else {
                    let mut i = 0usize;
                    for _ in 0..nb_samples {
                        for c in 0..channels {
                            let base = i * bps;
                            let raw = i32::from_ne_bytes([bytes[base], bytes[base + 1], bytes[base + 2], bytes[base + 3]]);
                            let v = (raw as f32) / 2147483648.0;
                            let x = v.abs();
                            let env_new = self.update_env(self.env[c], x);
                            self.env[c] = env_new;
                            let level_db = lin_to_db(env_new.max(eps));
                            let gain_db = self.compute_gain_db(level_db) + master;
                            let g = db_to_lin(gain_db);
                            let y = (v * g * 2147483648.0) as f64;
                            let out = clamp_i32(y).to_ne_bytes();
                            bytes[base..base + 4].copy_from_slice(&out);
                            i += 1;
                        }
                    }
                }
                Ok(())
            }
            SampleType::I64 => {
                let bps = 8;
                let scale = 9223372036854775808.0_f64;
                if planar_single_channel {
                    let mut env = self.env[channel_index_for_planar];
                    for ch in bytes.chunks_exact_mut(bps) {
                        let raw = i64::from_ne_bytes([ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7]]);
                        let v = raw as f64 / scale;
                        let x = v.abs() as f32;
                        env = self.update_env(env, x);
                        let level_db = lin_to_db(env.max(eps));
                        let gain_db = f64::from(self.compute_gain_db(level_db) + master);
                        let g = 10.0_f64.powf(gain_db * 0.05);
                        let y = v * g * scale;
                        let out = clamp_i64(y).to_ne_bytes();
                        ch.copy_from_slice(&out);
                    }
                    self.env[channel_index_for_planar] = env;
                } else {
                    let mut i = 0usize;
                    for _ in 0..nb_samples {
                        for c in 0..channels {
                            let base = i * bps;
                            let raw = i64::from_ne_bytes([
                                bytes[base],
                                bytes[base + 1],
                                bytes[base + 2],
                                bytes[base + 3],
                                bytes[base + 4],
                                bytes[base + 5],
                                bytes[base + 6],
                                bytes[base + 7],
                            ]);
                            let v = raw as f64 / scale;
                            let x = v.abs() as f32;
                            let env_new = self.update_env(self.env[c], x);
                            self.env[c] = env_new;
                            let level_db = lin_to_db(env_new.max(eps));
                            let gain_db = f64::from(self.compute_gain_db(level_db) + master);
                            let g = 10.0_f64.powf(gain_db * 0.05);
                            let y = v * g * scale;
                            let out = clamp_i64(y).to_ne_bytes();
                            bytes[base..base + 8].copy_from_slice(&out);
                            i += 1;
                        }
                    }
                }
                Ok(())
            }
            SampleType::F32 => {
                let bps = 4;
                if planar_single_channel {
                    let mut env = self.env[channel_index_for_planar];
                    for ch in bytes.chunks_exact_mut(bps) {
                        let mut v = f32::from_ne_bytes([ch[0], ch[1], ch[2], ch[3]]);
                        if !v.is_finite() {
                            v = 0.0;
                        }
                        let x = v.abs();
                        env = self.update_env(env, x);
                        let level_db = lin_to_db(env.max(eps));
                        let gain_db = self.compute_gain_db(level_db) + master;
                        let g = db_to_lin(gain_db);
                        let y = v * g;
                        ch.copy_from_slice(&y.to_ne_bytes());
                    }
                    self.env[channel_index_for_planar] = env;
                } else {
                    let mut i = 0usize;
                    for _ in 0..nb_samples {
                        for c in 0..channels {
                            let base = i * bps;
                            let mut v = f32::from_ne_bytes([
                                bytes[base],
                                bytes[base + 1],
                                bytes[base + 2],
                                bytes[base + 3],
                            ]);
                            if !v.is_finite() {
                                v = 0.0;
                            }
                            let x = v.abs();
                            let env_new = self.update_env(self.env[c], x);
                            self.env[c] = env_new;
                            let level_db = lin_to_db(env_new.max(eps));
                            let gain_db = self.compute_gain_db(level_db) + master;
                            let g = db_to_lin(gain_db);
                            let y = v * g;
                            bytes[base..base + 4].copy_from_slice(&y.to_ne_bytes());
                            i += 1;
                        }
                    }
                }
                Ok(())
            }
            SampleType::F64 => {
                let bps = 8;
                if planar_single_channel {
                    let mut env = self.env[channel_index_for_planar];
                    for ch in bytes.chunks_exact_mut(bps) {
                        let mut v = f64::from_ne_bytes([ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7]]);
                        if !v.is_finite() {
                            v = 0.0;
                        }
                        let x = v.abs() as f32;
                        env = self.update_env(env, x);
                        let level_db = lin_to_db(env.max(eps));
                        let gain_db = f64::from(self.compute_gain_db(level_db) + master);
                        let g = 10.0_f64.powf(gain_db * 0.05);
                        let y = v * g;
                        ch.copy_from_slice(&y.to_ne_bytes());
                    }
                    self.env[channel_index_for_planar] = env;
                } else {
                    let mut i = 0usize;
                    for _ in 0..nb_samples {
                        for c in 0..channels {
                            let base = i * bps;
                            let mut v = f64::from_ne_bytes([
                                bytes[base],
                                bytes[base + 1],
                                bytes[base + 2],
                                bytes[base + 3],
                                bytes[base + 4],
                                bytes[base + 5],
                                bytes[base + 6],
                                bytes[base + 7],
                            ]);
                            if !v.is_finite() {
                                v = 0.0;
                            }
                            let x = v.abs() as f32;
                            let env_new = self.update_env(self.env[c], x);
                            self.env[c] = env_new;
                            let level_db = lin_to_db(env_new.max(eps));
                            let gain_db = f64::from(self.compute_gain_db(level_db) + master);
                            let g = 10.0_f64.powf(gain_db * 0.05);
                            let y = v * g;
                            bytes[base..base + 8].copy_from_slice(&y.to_ne_bytes());
                            i += 1;
                        }
                    }
                }
                Ok(())
            }
        }
    }
}


