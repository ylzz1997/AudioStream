//! WASM exports (wasm-bindgen).
//!
//! 主要用途：在浏览器里接收 Python server 发送的“编码帧 frame(bytes)”并解码成 PCM，
//! 然后把 PCM 作为 `Float32Array` 交给 WebAudio 做流式播放。
//!
//! 当前仅导出 **WAV/PCM**（也就是本项目里的 "wav"/"pcm" encoder 输出的 raw PCM bytes）。
//! MP3/AAC 需要 ffmpeg backend，不适合 wasm32-unknown-unknown（可后续改成 wasm 侧引入纯 Rust 解码器再支持）。

use wasm_bindgen::prelude::*;

use js_sys::Float32Array;
use std::collections::VecDeque;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleType {
    U8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

fn bps(st: SampleType) -> usize {
    match st {
        SampleType::U8 => 1,
        SampleType::I16 => 2,
        SampleType::I32 | SampleType::F32 => 4,
        SampleType::I64 | SampleType::F64 => 8,
    }
}

fn norm_i(st: SampleType) -> f32 {
    match st {
        SampleType::U8 => 128.0, // 0..255 -> (-1..~1)
        SampleType::I16 => 32768.0,
        SampleType::I32 => 2147483648.0,
        SampleType::I64 => 9223372036854775808.0_f64 as f32, // 精度够用（只是归一化）
        SampleType::F32 | SampleType::F64 => 1.0,
    }
}

fn state_str(ready: bool, need_more: bool) -> &'static str {
    if ready {
        "ready"
    } else if need_more {
        "need_more"
    } else {
        "empty"
    }
}

/// 面向浏览器的 WAV/PCM 解码器（其实是“把 bytes 解释成 PCM 并转 float32”）。
///
/// - 输入：Python server 发送来的 frame(bytes)，内容是 **interleaved PCM bytes**
/// - 输出：`Float32Array`，内容是 **interleaved float32 samples**（长度 = channels * samples）
///
/// chunk 语义：
/// - `get_pcm(force=false)`：只有当累计样本 >= chunk_samples 才返回一个 chunk（否则 None）
/// - `get_pcm(force=true)`：若剩余不满一个 chunk 且不为 0，则强制返回最后残留
#[wasm_bindgen]
pub struct WavPcmDecoder {
    sample_rate: u32,
    channels: u32,
    sample_type: SampleType,
    chunk_samples: u32,
    // interleaved float32 FIFO：len = channels * samples
    q: VecDeque<f32>,
}

#[wasm_bindgen]
impl WavPcmDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: u32, channels: u32, sample_type: SampleType, chunk_samples: u32) -> Result<WavPcmDecoder, JsValue> {
        if sample_rate == 0 {
            return Err(JsValue::from_str("sample_rate must be > 0"));
        }
        if channels == 0 {
            return Err(JsValue::from_str("channels must be > 0"));
        }
        if chunk_samples == 0 {
            return Err(JsValue::from_str("chunk_samples must be > 0"));
        }
        Ok(Self {
            sample_rate,
            channels,
            sample_type,
            chunk_samples,
            q: VecDeque::new(),
        })
    }

    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u32 {
        self.channels
    }

    #[wasm_bindgen(getter)]
    pub fn chunk_samples(&self) -> u32 {
        self.chunk_samples
    }

    /// 当前 FIFO 内累计的 PCM 样本数（每声道）。
    #[wasm_bindgen]
    pub fn pending_samples(&self) -> u32 {
        let ch = self.channels as usize;
        if ch == 0 {
            return 0;
        }
        (self.q.len() / ch) as u32
    }

    /// 当前状态： "ready" | "need_more" | "empty"
    #[wasm_bindgen]
    pub fn state(&self) -> String {
        let left = self.pending_samples();
        let ready = left >= self.chunk_samples;
        let need_more = left > 0 && left < self.chunk_samples;
        state_str(ready, need_more).to_string()
    }

    /// 输入一帧 interleaved PCM bytes（来自 python server）。
    ///
    /// 说明：这里不做解码（WAV/PCM 本质就是 bytes），只做类型转换到 float32 并入队。
    #[wasm_bindgen]
    pub fn put_frame(&mut self, frame: &[u8]) -> Result<(), JsValue> {
        let ch = self.channels as usize;
        let bps = bps(self.sample_type);
        if frame.len() % (ch * bps) != 0 {
            return Err(JsValue::from_str("frame size not aligned to channels * bytes_per_sample"));
        }
        let ns = frame.len() / (ch * bps);
        let denom = norm_i(self.sample_type);

        match self.sample_type {
            SampleType::U8 => {
                for i in 0..(ns * ch) {
                    let v = frame[i] as f32;
                    self.q.push_back((v - 128.0) / denom);
                }
            }
            SampleType::I16 => {
                for i in 0..(ns * ch) {
                    let off = i * 2;
                    let v = i16::from_le_bytes([frame[off], frame[off + 1]]) as f32;
                    self.q.push_back(v / denom);
                }
            }
            SampleType::I32 => {
                for i in 0..(ns * ch) {
                    let off = i * 4;
                    let v = i32::from_le_bytes([frame[off], frame[off + 1], frame[off + 2], frame[off + 3]]) as f32;
                    self.q.push_back(v / denom);
                }
            }
            SampleType::I64 => {
                for i in 0..(ns * ch) {
                    let off = i * 8;
                    let v = i64::from_le_bytes([
                        frame[off],
                        frame[off + 1],
                        frame[off + 2],
                        frame[off + 3],
                        frame[off + 4],
                        frame[off + 5],
                        frame[off + 6],
                        frame[off + 7],
                    ]) as f64;
                    self.q.push_back((v as f32) / denom);
                }
            }
            SampleType::F32 => {
                for i in 0..(ns * ch) {
                    let off = i * 4;
                    let v = f32::from_le_bytes([frame[off], frame[off + 1], frame[off + 2], frame[off + 3]]);
                    self.q.push_back(v);
                }
            }
            SampleType::F64 => {
                for i in 0..(ns * ch) {
                    let off = i * 8;
                    let v = f64::from_le_bytes([
                        frame[off],
                        frame[off + 1],
                        frame[off + 2],
                        frame[off + 3],
                        frame[off + 4],
                        frame[off + 5],
                        frame[off + 6],
                        frame[off + 7],
                    ]);
                    self.q.push_back(v as f32);
                }
            }
        }

        Ok(())
    }

    /// 获取一段 PCM（interleaved float32）。
    ///
    /// - force=false：不够一个 chunk 则返回 None
    /// - force=true：若有残留则返回残留（最后不足一个 chunk）
    #[wasm_bindgen]
    pub fn get_pcm(&mut self, force: bool) -> Option<Float32Array> {
        let ch = self.channels as usize;
        if ch == 0 {
            return None;
        }

        let avail = self.q.len() / ch;
        let want = self.chunk_samples as usize;

        let take_samples = if avail >= want {
            want
        } else if force && avail > 0 {
            avail
        } else {
            return None;
        };

        let take_len = take_samples * ch;
        let mut out = Vec::with_capacity(take_len);
        for _ in 0..take_len {
            if let Some(v) = self.q.pop_front() {
                out.push(v);
            } else {
                break;
            }
        }
        Some(Float32Array::from(out.as_slice()))
    }
}


