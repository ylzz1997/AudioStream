// Audio Encoder Interface
use crate::codec::error::CodecResult;
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrameView};

/// ## 调用约束
/// - 正常编码：
///   - 循环：`send_frame(Some(frame))` 成功后，反复调用 `receive_packet()` 直到返回 `Again`
/// - 如果 `send_frame` 返回 `Again`：
///   - 说明内部输出队列未取空；先 `receive_packet()` 取到 `Again` 再继续 `send_frame`
/// - flush：
///   - 调用一次 `send_frame(None)` 表示输入结束
///   - 然后反复 `receive_packet()`，直到返回 `Eof`
pub trait AudioEncoder: Send {
    /// 编码器名（用于日志/调试）。
    fn name(&self) -> &'static str;

    /// 输入格式约束（部分编码器固定，只接受一种；也可能接受多种）。
    ///
    /// - 返回 `Some`: 表示必须匹配该格式（或至少匹配其中关键字段）
    /// - 返回 `None`: 表示尚未配置/或由实现自行检查
    fn input_format(&self) -> Option<AudioFormat>;

    /// 编码器推荐的输入帧长（单位：samples/每声道）。
    ///
    /// - 返回 `None`：表示编码器可接受可变帧长（上层按自己的 chunk 送也行）
    fn preferred_frame_samples(&self) -> Option<usize> {
        None
    }

    /// 编码器 lookahead/算法前瞻（单位：samples/每声道）。
    ///
    /// 对有重叠窗/MDCT 等编码器，内部往往需要缓存一部分“前一帧尾部”的样本, 0表示没有
    fn lookahead_samples(&self) -> usize {
        0
    }

    /// 送入一个原始音频帧。
    ///
    /// - `Some(frame)`: 正常输入
    /// - `None`: flush（输入结束）
    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()>;

    /// 取出一个压缩包。
    ///
    /// 返回值语义：
    /// - `Ok(pkt)`: 成功得到一个包
    /// - `Err(Again)`: 需要更多输入帧（或需要继续 send_frame 推进）
    /// - `Err(Eof)`: flush 后已无更多输出
    fn receive_packet(&mut self) -> CodecResult<CodecPacket>;

    /// 重置内部状态（丢弃缓存、回到初始态）。
    fn reset(&mut self) -> CodecResult<()>;
}

