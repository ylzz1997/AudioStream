// Audio Processor Interface (PCM -> PCM)
//
// 设计目标：
// - 把“PCM->PCM”作为一等抽象（resample / gain / mix / format convert ...）
// - 调用/背压语义对齐 encoder/decoder：send_frame/receive_frame + Again/Eof + flush(None)

use crate::codec::error::CodecResult;
use crate::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView};

/// ## 调用约束
/// - 正常处理：
///   - 循环：`send_frame(Some(frame))` 成功后，反复调用 `receive_frame()` 直到返回 `Again`
/// - 如果 `send_frame` 返回 `Again`：
///   - 说明内部输出队列未取空；先 `receive_frame()` 取到 `Again` 再继续 `send_frame`
/// - flush：
///   - 调用一次 `send_frame(None)` 表示输入结束
///   - 然后反复 `receive_frame()`，直到返回 `Eof`
pub trait AudioProcessor: Send {
    /// 处理器名（用于日志/调试）。
    fn name(&self) -> &'static str;

    /// 输入格式约束：
    /// - `Some`: 要求输入必须匹配（至少关键字段匹配）
    /// - `None`: 由实现自行检查
    fn input_format(&self) -> Option<AudioFormat>;

    /// 输出格式（有些 processor 在吃到首帧后才确定；也可能固定）。
    fn output_format(&self) -> Option<AudioFormat>;

    /// 处理器引入的算法延迟（单位：samples/每声道）。
    fn delay_samples(&self) -> usize {
        0
    }

    /// 送入一个原始音频帧。
    ///
    /// - `Some(frame)`: 正常输入
    /// - `None`: flush（输入结束）
    fn send_frame(&mut self, frame: Option<&dyn AudioFrameView>) -> CodecResult<()>;

    /// 取出一个处理后的音频帧。
    ///
    /// 返回值语义：
    /// - `Ok(frame)`: 成功得到一帧
    /// - `Err(Again)`: 需要更多输入帧（或需要继续 send_frame 推进）
    /// - `Err(Eof)`: flush 后已无更多输出
    fn receive_frame(&mut self) -> CodecResult<AudioFrame>;

    /// 重置内部状态（丢弃缓存、回到初始态）。
    fn reset(&mut self) -> CodecResult<()>;
}


