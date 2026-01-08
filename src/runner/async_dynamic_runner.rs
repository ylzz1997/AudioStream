use crate::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::NodeBuffer;
use crate::runner::audio_sink::AudioSink;
use crate::runner::audio_source::AudioSource;
use crate::runner::async_runner_interface::AsyncRunner;
use crate::runner::auto_runner::AutoRunner;
use crate::runner::error::RunnerResult;

/// 异步动态 Runner（薄封装）：Source(NodeBuffer) -> AsyncDynPipeline -> Sink(NodeBuffer)
pub struct AsyncDynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer>,
    K: AudioSink<In = NodeBuffer>,
{
    inner: AutoRunner<AsyncDynPipeline, S, K>,
}

impl<S, K> AsyncDynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer>,
    K: AudioSink<In = NodeBuffer>,
{
    pub fn new(source: S, pipeline: AsyncDynPipeline, sink: K) -> Self {
        Self {
            inner: AutoRunner::new(source, pipeline, sink),
        }
    }

    /// 直接传入节点列表（可包含 1 个节点），Runner 内部会构建 `AsyncDynPipeline` 并校验 kind。
    pub fn from_nodes(source: S, nodes: Vec<Box<dyn DynNode>>, sink: K) -> RunnerResult<Self> {
        let pipeline = AsyncDynPipeline::new(nodes)?;
        Ok(Self::new(source, pipeline, sink))
    }

    pub fn into_inner(self) -> AutoRunner<AsyncDynPipeline, S, K> {
        self.inner
    }
}

impl<S, K> AsyncRunner for AsyncDynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer>,
    K: AudioSink<In = NodeBuffer>,
{
    fn execute<'a>(
        &'a mut self,
    ) -> core::pin::Pin<Box<dyn core::future::Future<Output = crate::runner::error::RunnerResult<()>> + 'a>> {
        self.inner.execute()
    }
}


