//! Node 抽象：为 pipeline 提供统一的“输入/输出/flush/背压(Again)”语义。
//!
//! 本模块只放通用类型（Buffer/Kind），具体的动态/静态 node 实现在：
//! - `dynamic_node_interface.rs`
//! - `static_node_interface.rs`

use crate::codec::packet::CodecPacket;
use crate::common::audio::audio::AudioFrame;
use crate::codec::error::CodecResult;
use async_trait::async_trait;
use std::collections::VecDeque;

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

/// 将 tokio pipeline 拆分为“输入端”和“输出端”，用于 runner 侧并行驱动：
/// - 输入端负责 send/flush
/// - 输出端负责 recv/try_recv
pub trait AsyncPipelineEndpoint: Sized + Send + 'static {
    type In: Send + 'static;
    type Out: Send + 'static;
    type Producer: AsyncPipelineProducer<In = Self::In> + Send + 'static;
    type Consumer: AsyncPipelineConsumer<Out = Self::Out> + Send + 'static;

    fn endpoints(self) -> (Self::Producer, Self::Consumer);
}

/// pipeline 输入端：push/flush
pub trait AsyncPipelineProducer: Send {
    type In: Send + 'static;
    fn push_frame(&self, frame: Self::In) -> CodecResult<()>;
    fn flush(&self) -> CodecResult<()>;
}

/// pipeline 输出端：try_get/get
#[async_trait]
pub trait AsyncPipelineConsumer: Send {
    type Out: Send + 'static;
    fn try_get_frame(&mut self) -> CodecResult<Self::Out>;
    async fn get_frame(&mut self) -> CodecResult<Self::Out>;
}


/// Identity 节点：把输入 buffer 原样移动到输出（零拷贝，不做任何处理）。
///
/// - 作为占位节点，便于后续插入 processor/filter。
pub struct IdentityNode {
    pub kind: NodeBufferKind,
    pub q: VecDeque<NodeBuffer>,
    pub flushed: bool,
}

impl IdentityNode {
    pub fn new(kind: NodeBufferKind) -> Self {
        Self {
            kind,
            q: VecDeque::new(),
            flushed: false,
        }
    }
}
