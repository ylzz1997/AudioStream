//! Fork/Join 节点的一些共享 trait（用于“静态版本”复用 pipeline 约束）。

use crate::codec::error::CodecResult;
use async_trait::async_trait;
use crate::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;

/// 同步（非 tokio）静态 pipeline 抽象：具备 `push_and_drain/drain_all/reset` 语义。
///
/// 说明：
/// - 这是为了让 `ForkJoinStaticNode` 能接收不止 `Pipeline3` 一种实现
/// - 现有 `Pipeline3` 已在 `static_fork_join_node.rs` 里提供了实现
pub trait SyncStaticPipeline {
    type In: Clone;
    type Out;

    fn push_and_drain(&mut self, input: Option<Self::In>) -> CodecResult<Vec<Self::Out>>;
    fn drain_all(&mut self) -> CodecResult<Vec<Self::Out>>;
    fn reset(&mut self, force: bool) -> CodecResult<()>;
}

/// tokio 异步 pipeline 的 reset 抽象（让 `StaticNode::reset()` 能在 blocking 线程里 block_on）。
///
/// 注意：这里用 `async_trait(?Send)`，因为 reset 通常在 `spawn_blocking` 线程里被 `Handle::block_on(...)`
/// 直接同步等待，不需要把 Future 跨线程移动。
#[async_trait(?Send)]
pub trait AsyncResettablePipeline: Send + 'static {
    async fn reset(&self, force: bool) -> CodecResult<()>;
}

#[async_trait(?Send)]
impl AsyncResettablePipeline for AsyncDynPipeline {
    async fn reset(&self, force: bool) -> CodecResult<()> {
        AsyncDynPipeline::reset(self, force).await
    }
}

