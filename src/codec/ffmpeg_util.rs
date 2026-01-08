//! 兼容层：旧路径 `crate::codec::ffmpeg_util`。
//!
//! 实际实现已迁移到 `crate::common::ffmpeg_util`。
//! 为了避免 IDE/诊断缓存与历史引用造成困扰，这里保留一个薄封装做 re-export。

#[cfg(feature = "ffmpeg")]
pub use crate::common::ffmpeg_util::{ff_err_to_string, map_ff_err};


