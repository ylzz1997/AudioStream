// Audio Decoder Interface
use crate::codec::error::CodecResult;
use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::{AudioFormat, AudioFrame};

/// ## 调用约束
/// - 正常解码：
///   - 循环：`send_packet(Some(pkt))` 成功后，反复调用 `receive_frame()` 直到返回 `Again`
/// - 如果 `send_packet` 返回 `Again`：
///   - 说明内部输出队列未取空；先 `receive_frame()` 取到 `Again` 再继续 `send_packet`
/// - flush：
///   - 调用一次 `send_packet(None)` 表示输入结束
///   - 然后反复 `receive_frame()`，直到返回 `Eof`
pub trait AudioDecoder: Send {
    /// 解码器名（用于日志/调试）。
    fn name(&self) -> &'static str;

    /// 当前输出音频格式（部分解码器可能在吃到首包后才确定）。
    fn output_format(&self) -> Option<AudioFormat>;

    /// 解码器引入的算法延迟（单位：samples/每声道）。
    ///
    /// - 对很多音频解码器来说可能是 0
    /// - 对有 filterbank/overlap-add 的解码器，可以用它帮助下游做 A/V sync 或首帧对齐
    fn delay_samples(&self) -> usize {
        0
    }

    /// 送入一个压缩包。
    ///
    /// - `Some(pkt)`: 正常输入
    /// - `None`: flush（输入结束）
    fn send_packet(&mut self, packet: Option<CodecPacket>) -> CodecResult<()>;

    /// 取出一个解码后的音频帧。
    ///
    /// 返回值语义：
    /// - `Ok(frame)`: 成功得到一帧
    /// - `Err(Again)`: 需要更多输入包（或需要继续 send_packet 推进）
    /// - `Err(Eof)`: flush 后已无更多输出
    fn receive_frame(&mut self) -> CodecResult<AudioFrame>;

    /// 重置内部状态（丢弃缓存、回到初始态）。
    fn reset(&mut self) -> CodecResult<()>;
}

