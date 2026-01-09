//! Node 抽象：为 pipeline 提供统一的“输入/输出/flush/背压(Again)”语义。
//!
//! 本模块只放通用类型（Buffer/Kind），具体的动态/静态 node 实现在：
//! - `dynamic_node_interface.rs`
//! - `static_node_interface.rs`

use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::AudioFrame;
use crate::codec::error::CodecResult;
use async_trait::async_trait;

/// node 之间传递的数据类型（运行时版 pipeline 用）。
#[derive(Debug)]
pub enum NodeBuffer {
    Pcm(AudioFrame),
    Packet(CodecPacket),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeBufferKind {
    Pcm,
    Packet,
}

impl NodeBuffer {
    pub fn kind(&self) -> NodeBufferKind {
        match self {
            NodeBuffer::Pcm(_) => NodeBufferKind::Pcm,
            NodeBuffer::Packet(_) => NodeBufferKind::Packet,
        }
    }
}

#[async_trait]
pub trait AsyncPipeline {
    type In: Send + 'static;
    type Out: Send + 'static;

    fn push_frame(&self, frame: Self::In) -> CodecResult<()>;
    fn flush(&self) -> CodecResult<()>;
    fn try_get_frame(&mut self) -> CodecResult<Self::Out>;
    async fn get_frame(&mut self) -> CodecResult<Self::Out>;
}