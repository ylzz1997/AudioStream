// Audio Common Types

use core::fmt;

/// 有理数（常用于 time_base，例如 1/48000）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    pub num: i32,
    pub den: i32,
}

impl Rational {
    pub const fn new(num: i32, den: i32) -> Self {
        Self { num, den }
    }

    pub const fn is_valid(&self) -> bool {
        self.den != 0
    }
}

/// 采样数据类型（不含 planar/interleaved 信息）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleType {
    U8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

/// 采样格式（包含 planar/interleaved 信息，类似 FFmpeg 的 AVSampleFormat）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    U8 { planar: bool },
    I16 { planar: bool },
    I32 { planar: bool },
    I64 { planar: bool },
    F32 { planar: bool },
    F64 { planar: bool },
}

impl SampleFormat {
    pub const fn is_planar(&self) -> bool {
        match self {
            SampleFormat::U8 { planar }
            | SampleFormat::I16 { planar }
            | SampleFormat::I32 { planar }
            | SampleFormat::I64 { planar }
            | SampleFormat::F32 { planar }
            | SampleFormat::F64 { planar } => *planar,
        }
    }

    pub const fn sample_type(&self) -> SampleType {
        match self {
            SampleFormat::U8 { .. } => SampleType::U8,
            SampleFormat::I16 { .. } => SampleType::I16,
            SampleFormat::I32 { .. } => SampleType::I32,
            SampleFormat::I64 { .. } => SampleType::I64,
            SampleFormat::F32 { .. } => SampleType::F32,
            SampleFormat::F64 { .. } => SampleType::F64,
        }
    }

    pub const fn bytes_per_sample(&self) -> usize {
        match self.sample_type() {
            SampleType::U8 => 1,
            SampleType::I16 => 2,
            SampleType::I32 | SampleType::F32 => 4,
            SampleType::I64 | SampleType::F64 => 8,
        }
    }
}

/// 声道布局掩码（简化版，后续可对齐 FFmpeg 的完整布局集合）。
///
/// - `channels`: 声道数（用于快速校验/计算）
/// - `mask`: 位掩码（用于描述具体空间位置）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelLayout {
    pub channels: u16,
    pub mask: u64,
}

impl ChannelLayout {
    pub const FRONT_LEFT: u64 = 1 << 0;
    pub const FRONT_RIGHT: u64 = 1 << 1;
    pub const FRONT_CENTER: u64 = 1 << 2;
    pub const LOW_FREQUENCY: u64 = 1 << 3;
    pub const BACK_LEFT: u64 = 1 << 4;
    pub const BACK_RIGHT: u64 = 1 << 5;

    pub const fn mono() -> Self {
        Self {
            channels: 1,
            mask: Self::FRONT_CENTER,
        }
    }

    pub const fn stereo() -> Self {
        Self {
            channels: 2,
            mask: Self::FRONT_LEFT | Self::FRONT_RIGHT,
        }
    }

    /// 掩码未知但声道数已知的情况（比如某些容器/码流只给 channels）。
    pub const fn unspecified(channels: u16) -> Self {
        Self { channels, mask: 0 }
    }
}

/// 音频格式描述（对标 FFmpeg 的 AVCodecParameters / SwrContext 输入输出参数组合）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub sample_format: SampleFormat,
    pub channel_layout: ChannelLayout,
}

impl AudioFormat {
    pub const fn channels(&self) -> u16 {
        self.channel_layout.channels
    }

    pub const fn is_planar(&self) -> bool {
        self.sample_format.is_planar()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    InvalidTimeBase(Rational),
    InvalidPlaneCount { expected: usize, actual: usize },
    InvalidPlaneSize { plane: usize, expected: usize, actual: usize },
    InvalidFormat(&'static str),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::InvalidTimeBase(tb) => write!(f, "invalid time_base: {}/{}", tb.num, tb.den),
            AudioError::InvalidPlaneCount { expected, actual } => {
                write!(f, "invalid plane count: expected {expected}, got {actual}")
            }
            AudioError::InvalidPlaneSize {
                plane,
                expected,
                actual,
            } => write!(
                f,
                "invalid plane size for plane {plane}: expected {expected} bytes, got {actual}"
            ),
            AudioError::InvalidFormat(msg) => write!(f, "invalid audio format: {msg}"),
        }
    }
}

impl std::error::Error for AudioError {}

/// 只读音频帧视图 trait。
///
/// 你后续做 encoder/decoder/filter 时，参数尽量接收 `&dyn AudioFrameView`，
/// 这样 `AudioFrame`、零拷贝引用帧、甚至 wasm 映射内存的帧都能统一接入。
pub trait AudioFrameView: Send + Sync {
    /// 格式描述（采样率/声道/采样格式）。
    fn format(&self) -> AudioFormat;

    /// 每个声道的采样数（而不是总采样点数）。
    fn nb_samples(&self) -> usize;

    /// 时间基（pts 的单位）。
    fn time_base(&self) -> Rational;

    /// 展示时间戳（可选；对齐 FFmpeg 的 `AVFrame.pts`）。
    fn pts(&self) -> Option<i64>;

    /// plane 数：planar=channels，interleaved=1。
    fn plane_count(&self) -> usize;

    /// 取某个 plane 的原始字节视图。
    fn plane(&self, index: usize) -> Option<&[u8]>;

    /// 常用派生信息：是否 planar。
    fn is_planar(&self) -> bool {
        self.format().is_planar()
    }

    /// 常用派生信息：声道数。
    fn channels(&self) -> u16 {
        self.format().channels()
    }

    /// 常用派生信息：采样率。
    fn sample_rate(&self) -> u32 {
        self.format().sample_rate
    }

    /// 常用派生信息：每采样字节数。
    fn bytes_per_sample(&self) -> usize {
        self.format().sample_format.bytes_per_sample()
    }
}

/// 可写音频帧视图 trait（用于滤镜/重采样/解码输出写入等）。
pub trait AudioFrameViewMut: AudioFrameView {
    fn set_pts(&mut self, pts: Option<i64>);
    fn set_time_base(&mut self, tb: Rational) -> Result<(), AudioError>;
    fn plane_mut(&mut self, index: usize) -> Option<&mut [u8]>;
}

/// 拥有型音频帧实现（类似一个更 Rust 的 AVFrame 子集）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioFrame {
    format: AudioFormat,
    nb_samples: usize,
    time_base: Rational,
    pts: Option<i64>,
    planes: Vec<Vec<u8>>,
}

impl AudioFrame {
    /// 创建并分配内存（planes 大小按 format/nb_samples 计算）。
    pub fn new_alloc(format: AudioFormat, nb_samples: usize) -> Result<Self, AudioError> {
        Self::validate_format(&format)?;
        let planes = Self::alloc_planes(format, nb_samples);
        Ok(Self {
            format,
            nb_samples,
            time_base: Rational::new(1, format.sample_rate as i32),
            pts: None,
            planes,
        })
    }

    /// 使用外部 planes 构造（会校验数量与大小）。
    pub fn from_planes(
        format: AudioFormat,
        nb_samples: usize,
        time_base: Rational,
        pts: Option<i64>,
        planes: Vec<Vec<u8>>,
    ) -> Result<Self, AudioError> {
        Self::validate_format(&format)?;
        if !time_base.is_valid() {
            return Err(AudioError::InvalidTimeBase(time_base));
        }
        Self::validate_planes(&format, nb_samples, &planes)?;
        Ok(Self {
            format,
            nb_samples,
            time_base,
            pts,
            planes,
        })
    }

    pub fn format_ref(&self) -> &AudioFormat {
        &self.format
    }

    pub fn pts_mut(&mut self) -> &mut Option<i64> {
        &mut self.pts
    }

    pub fn planes_ref(&self) -> &[Vec<u8>] {
        &self.planes
    }

    pub fn planes_mut(&mut self) -> &mut [Vec<u8>] {
        &mut self.planes
    }

    pub fn expected_plane_count(format: &AudioFormat) -> usize {
        if format.is_planar() {
            format.channels() as usize
        } else {
            1
        }
    }

    /// 单个 plane 期望的字节数。
    pub fn expected_bytes_per_plane(format: &AudioFormat, nb_samples: usize) -> usize {
        let bps = format.sample_format.bytes_per_sample();
        if format.is_planar() {
            nb_samples * bps
        } else {
            nb_samples * (format.channels() as usize) * bps
        }
    }

    fn alloc_planes(format: AudioFormat, nb_samples: usize) -> Vec<Vec<u8>> {
        let plane_count = Self::expected_plane_count(&format);
        let plane_bytes = Self::expected_bytes_per_plane(&format, nb_samples);
        let mut planes = Vec::with_capacity(plane_count);
        for _ in 0..plane_count {
            planes.push(vec![0u8; plane_bytes]);
        }
        planes
    }

    fn validate_format(format: &AudioFormat) -> Result<(), AudioError> {
        if format.sample_rate == 0 {
            return Err(AudioError::InvalidFormat("sample_rate must be > 0"));
        }
        if format.channel_layout.channels == 0 {
            return Err(AudioError::InvalidFormat("channels must be > 0"));
        }
        Ok(())
    }

    fn validate_planes(
        format: &AudioFormat,
        nb_samples: usize,
        planes: &[Vec<u8>],
    ) -> Result<(), AudioError> {
        let expected_count = Self::expected_plane_count(format);
        if planes.len() != expected_count {
            return Err(AudioError::InvalidPlaneCount {
                expected: expected_count,
                actual: planes.len(),
            });
        }
        let expected_bytes = Self::expected_bytes_per_plane(format, nb_samples);
        for (i, p) in planes.iter().enumerate() {
            if p.len() != expected_bytes {
                return Err(AudioError::InvalidPlaneSize {
                    plane: i,
                    expected: expected_bytes,
                    actual: p.len(),
                });
            }
        }
        Ok(())
    }
}

impl AudioFrameView for AudioFrame {
    fn format(&self) -> AudioFormat {
        self.format
    }

    fn nb_samples(&self) -> usize {
        self.nb_samples
    }

    fn time_base(&self) -> Rational {
        self.time_base
    }

    fn pts(&self) -> Option<i64> {
        self.pts
    }

    fn plane_count(&self) -> usize {
        self.planes.len()
    }

    fn plane(&self, index: usize) -> Option<&[u8]> {
        self.planes.get(index).map(|p| p.as_slice())
    }
}

impl AudioFrameViewMut for AudioFrame {
    fn set_pts(&mut self, pts: Option<i64>) {
        self.pts = pts;
    }

    fn set_time_base(&mut self, tb: Rational) -> Result<(), AudioError> {
        if !tb.is_valid() {
            return Err(AudioError::InvalidTimeBase(tb));
        }
        self.time_base = tb;
        Ok(())
    }

    fn plane_mut(&mut self, index: usize) -> Option<&mut [u8]> {
        self.planes.get_mut(index).map(|p| p.as_mut_slice())
    }
}

