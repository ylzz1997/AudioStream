use crate::codec::error::CodecError;
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFrame, AudioFrameView};
use crate::common::io::io::{AudioIOError, AudioIOResult, AudioWriter};

use super::AudioWriterChain;

/// 一条“线性链路”写端：`processors (PCM->PCM)* -> writer`。
pub struct LineAudioWriter {
    processors: Vec<Box<dyn AudioProcessor>>,
    writer: Box<dyn AudioWriter + Send>,
    finalized: bool,
}

impl LineAudioWriter {
    pub fn new(writer: Box<dyn AudioWriter + Send>) -> Self {
        Self {
            processors: vec![],
            writer,
            finalized: false,
        }
    }

    pub fn with_processors(
        processors: Vec<Box<dyn AudioProcessor>>,
        writer: Box<dyn AudioWriter + Send>,
    ) -> Self {
        Self {
            processors,
            writer,
            finalized: false,
        }
    }

    /// 追加一个 processor（追加到链路末端 writer 之前）。
    ///
    /// 注意：如果已经 `finalize()`，再追加 processor 没有意义；这里直接忽略并保留 API 简洁性。
    pub fn push_processor(&mut self, p: Box<dyn AudioProcessor>) {
        <Self as AudioWriterChain>::push_processor(self, p)
    }

    pub fn processors(&self) -> &[Box<dyn AudioProcessor>] {
        <Self as AudioWriterChain>::processors(self)
    }

    pub fn processors_mut(&mut self) -> &mut [Box<dyn AudioProcessor>] {
        <Self as AudioWriterChain>::processors_mut(self)
    }

    pub fn writer(&self) -> &dyn AudioWriter {
        <Self as AudioWriterChain>::writer(self)
    }

    pub fn writer_mut(&mut self) -> &mut dyn AudioWriter {
        <Self as AudioWriterChain>::writer_mut(self)
    }

    pub fn into_inner(self) -> (Vec<Box<dyn AudioProcessor>>, Box<dyn AudioWriter + Send>) {
        (self.processors, self.writer)
    }

    fn send_and_drain(
        &mut self,
        idx: usize,
        input: Option<&dyn AudioFrameView>,
    ) -> AudioIOResult<()> {
        loop {
            match self.processors[idx].send_frame(input) {
                Ok(()) => break,
                Err(CodecError::Again) => {
                    // 背压：先把当前 processor 的输出取空（推进到末端）再重试 send。
                    self.drain_outputs(idx)?;
                }
                Err(e) => return Err(AudioIOError::Codec(e)),
            }
        }
        // send 成功后，尽可能把该 processor 产生的输出全部推进到末端
        self.drain_outputs(idx)
    }

    fn forward_frame(&mut self, idx: usize, frame: AudioFrame) -> AudioIOResult<()> {
        if idx + 1 < self.processors.len() {
            let v = &frame as &dyn AudioFrameView;
            self.send_and_drain(idx + 1, Some(v))
        } else {
            self.writer.write_frame(&frame as &dyn AudioFrameView)
        }
    }

    fn drain_outputs(&mut self, idx: usize) -> AudioIOResult<()> {
        loop {
            match self.processors[idx].receive_frame() {
                Ok(f) => self.forward_frame(idx, f)?,
                Err(CodecError::Again) => break,
                Err(CodecError::Eof) => {
                    // upstream 已结束（flush 后）。把 flush 传递到下游（仅一次）。
                    if idx + 1 < self.processors.len() {
                        self.send_and_drain(idx + 1, None)?;
                    }
                    break;
                }
                Err(e) => return Err(AudioIOError::Codec(e)),
            }
        }
        Ok(())
    }
}

impl AudioWriterChain for LineAudioWriter {
    fn push_processor(&mut self, p: Box<dyn AudioProcessor>) {
        if self.finalized {
            return;
        }
        self.processors.push(p);
    }

    fn processors(&self) -> &[Box<dyn AudioProcessor>] {
        &self.processors
    }

    fn processors_mut(&mut self) -> &mut [Box<dyn AudioProcessor>] {
        &mut self.processors
    }

    fn writer(&self) -> &dyn AudioWriter {
        self.writer.as_ref()
    }

    fn writer_mut(&mut self) -> &mut dyn AudioWriter {
        self.writer.as_mut()
    }
}

impl AudioWriter for LineAudioWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        if self.finalized {
            return Err(AudioIOError::Format("LineAudioWriter already finalized"));
        }

        if self.processors.is_empty() {
            return self.writer.write_frame(frame);
        }

        self.send_and_drain(0, Some(frame))
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        if self.finalized {
            // finalize 设计为幂等（避免调用方二次 finalize 触发错误）。
            return Ok(());
        }
        self.finalized = true;

        if !self.processors.is_empty() {
            // flush processors，并把残留尽可能推进到末端 writer
            self.send_and_drain(0, None)?;
        }

        self.writer.finalize()
    }
}

