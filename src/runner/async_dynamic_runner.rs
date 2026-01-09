use crate::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::NodeBuffer;
use crate::runner::audio_sink::AudioSink;
use crate::runner::audio_source::AudioSource;
use crate::runner::async_runner_interface::AsyncRunner;
use crate::runner::async_auto_runner::AsyncAutoRunner;
use crate::runner::error::RunnerResult;
use async_trait::async_trait;

/// 异步动态 Runner（薄封装）：Source(NodeBuffer) -> AsyncDynPipeline -> Sink(NodeBuffer)
pub struct AsyncDynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer> + Send + 'static,
    K: AudioSink<In = NodeBuffer> + Send + 'static,
{
    inner: AsyncAutoRunner<AsyncDynPipeline, S, K>,
}

impl<S, K> AsyncDynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer> + Send + 'static,
    K: AudioSink<In = NodeBuffer> + Send + 'static,
{
    pub fn new(source: S, pipeline: AsyncDynPipeline, sink: K) -> Self {
        Self {
            inner: AsyncAutoRunner::new(source, pipeline, sink),
        }
    }

    /// 直接传入节点列表（可包含 1 个节点），Runner 内部会构建 `AsyncDynPipeline` 并校验 kind。
    pub fn from_nodes(source: S, nodes: Vec<Box<dyn DynNode>>, sink: K) -> RunnerResult<Self> {
        let pipeline = AsyncDynPipeline::new(nodes)?;
        Ok(Self::new(source, pipeline, sink))
    }
}

#[async_trait(?Send)]
impl<S, K> AsyncRunner for AsyncDynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer> + Send + 'static,
    K: AudioSink<In = NodeBuffer> + Send + 'static,
{
    async fn execute(&mut self) -> RunnerResult<()> {
        self.inner.execute().await
    }
}


