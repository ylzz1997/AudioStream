// Codec Packet Types
use crate::common::audio::audio::Rational;

/// - `pts/dts` 的单位由 `time_base` 决定
/// - `data` 的内容由具体 codec 定义（例如 AAC raw / ADTS / LATM 等）
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecPacket {
    pub data: Vec<u8>,

    /// 时间基（pts/dts/duration 的单位）。
    pub time_base: Rational,

    /// 显示时间戳（可选）。
    pub pts: Option<i64>,

    /// 解码时间戳（可选，某些有 B-frames 的音频编码也可能用到）。
    pub dts: Option<i64>,

    /// 包持续时间（可选）。
    pub duration: Option<i64>,

    /// 额外标记（例如 keyframe/discard 等；音频里通常较少用）。
    pub flags: PacketFlags,
}

impl CodecPacket {
    pub fn new(data: Vec<u8>, time_base: Rational) -> Self {
        Self {
            data,
            time_base,
            pts: None,
            dts: None,
            duration: None,
            flags: PacketFlags::empty(),
        }
    }
}

/// 包标记位（先做最小集合；需要时再扩充）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PacketFlags(u32);

impl PacketFlags {
    pub const fn empty() -> Self {
        Self(0)
    }

    /// 从原始 bitmask 构造（用于跨语言/跨层透传）。
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// 取出原始 bitmask（用于跨语言/跨层透传）。
    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}


