use crate::common::audio::audio::{AudioError, AudioFormat, AudioFrame, AudioFrameView, Rational};
use std::collections::VecDeque;

/// PCM 音频 FIFO（用于重分帧 / chunk 聚合 / 流式边读边写）。
/// 
/// 约束：
/// - FIFO 绑定一个固定 `AudioFormat`（不做重采样/重排声道/采样格式转换）
/// - 支持 planar / interleaved 两种内存布局
/// - pts 处理：如果输入帧提供了连续且一致的 pts+time_base，则 pop 出来的帧也会带 pts；
///   否则输出 pts=None（上层可自行决定如何处理时间线）
pub struct AudioFifo {
    format: AudioFormat,
    time_base: Rational,
    pts_base: Option<i64>, // FIFO 第一个 sample 的 pts（单位：time_base）
    planes: Vec<VecDeque<u8>>, // planar=channels 个；interleaved=1 个
}

impl AudioFifo {
    pub fn new(format: AudioFormat, time_base: Rational) -> Result<Self, AudioError> {
        if !time_base.is_valid() {
            return Err(AudioError::InvalidTimeBase(time_base));
        }
        let plane_count = if format.is_planar() {
            format.channels() as usize
        } else {
            1
        };
        Ok(Self {
            format,
            time_base,
            pts_base: None,
            planes: (0..plane_count).map(|_| VecDeque::new()).collect(),
        })
    }

    pub fn format(&self) -> AudioFormat {
        self.format
    }

    pub fn time_base(&self) -> Rational {
        self.time_base
    }

    /// 当前 FIFO 内可用的 samples 数（每声道）。
    pub fn available_samples(&self) -> usize {
        let bps = self.format.sample_format.bytes_per_sample();
        if self.format.is_planar() {
            // 以最短 plane 为准（理论上应该一致）
            self.planes
                .iter()
                .map(|p| p.len() / bps)
                .min()
                .unwrap_or(0)
        } else {
            let ch = self.format.channels() as usize;
            let bytes_per_sample_all = ch * bps;
            self.planes
                .get(0)
                .map(|p| p.len() / bytes_per_sample_all)
                .unwrap_or(0)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.available_samples() == 0
    }

    pub fn clear(&mut self) {
        for p in &mut self.planes {
            p.clear();
        }
        self.pts_base = None;
    }

    /// 向 FIFO 追加一帧 PCM。
    pub fn push_frame(&mut self, frame: &dyn AudioFrameView) -> Result<(), AudioError> {
        let fmt = frame.format();
        if fmt != self.format {
            return Err(AudioError::InvalidFormat("AudioFifo format mismatch"));
        }
        if frame.time_base() != self.time_base {
            // 时间基不一致时，我们直接放弃 pts 追踪，避免产生错误时间线。
            self.pts_base = None;
        }

        // pts 连续性处理（尽量保持；不连续则放弃）
        if let Some(pts) = frame.pts() {
            let expected = self.pts_base.map(|base| base + self.available_samples() as i64);
            if self.pts_base.is_none() && self.available_samples() == 0 {
                self.pts_base = Some(pts);
            } else if let Some(exp) = expected {
                if exp != pts {
                    self.pts_base = None;
                }
            } else {
                self.pts_base = None;
            }
        } else {
            self.pts_base = None;
        }

        let bps = self.format.sample_format.bytes_per_sample();
        let nb = frame.nb_samples();

        if self.format.is_planar() {
            let ch = self.format.channels() as usize;
            for c in 0..ch {
                let src = frame
                    .plane(c)
                    .ok_or(AudioError::InvalidFormat("missing plane"))?;
                let expected = nb * bps;
                if src.len() != expected {
                    return Err(AudioError::InvalidPlaneSize {
                        plane: c,
                        expected,
                        actual: src.len(),
                    });
                }
                self.planes[c].extend(src);
            }
        } else {
            let src = frame
                .plane(0)
                .ok_or(AudioError::InvalidFormat("missing plane 0"))?;
            let ch = self.format.channels() as usize;
            let expected = nb * ch * bps;
            if src.len() != expected {
                return Err(AudioError::InvalidPlaneSize {
                    plane: 0,
                    expected,
                    actual: src.len(),
                });
            }
            self.planes[0].extend(src);
        }

        Ok(())
    }

    /// 如果可用 samples >= `nb_samples`，弹出一帧；否则返回 None。
    pub fn pop_frame(&mut self, nb_samples: usize) -> Result<Option<AudioFrame>, AudioError> {
        if self.available_samples() < nb_samples {
            return Ok(None);
        }
        let bps = self.format.sample_format.bytes_per_sample();

        let pts = self.pts_base;
        if let Some(base) = self.pts_base {
            self.pts_base = Some(base + nb_samples as i64);
        }

        let planes: Vec<Vec<u8>> = if self.format.is_planar() {
            let ch = self.format.channels() as usize;
            let mut out = Vec::with_capacity(ch);
            let bytes = nb_samples * bps;
            for c in 0..ch {
                out.push(drain_bytes(&mut self.planes[c], bytes));
            }
            out
        } else {
            let ch = self.format.channels() as usize;
            let bytes = nb_samples * ch * bps;
            vec![drain_bytes(&mut self.planes[0], bytes)]
        };

        Ok(Some(AudioFrame::from_planes(
            self.format,
            nb_samples,
            self.time_base,
            pts,
            planes,
        )?))
    }
}

fn drain_bytes(q: &mut VecDeque<u8>, n: usize) -> Vec<u8> {
    q.drain(..n).collect()
}


