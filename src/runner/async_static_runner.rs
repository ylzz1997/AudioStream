use crate::pipeline::node::async_static_node_interface::AsyncPipeline3;
use crate::pipeline::node::static_node_interface::StaticNode;
use crate::runner::audio_sink::AudioSink;
use crate::runner::audio_source::AudioSource;
use crate::runner::async_runner_interface::AsyncRunner;
use crate::runner::async_auto_runner::AsyncAutoRunner;
use crate::runner::error::RunnerResult;
use async_trait::async_trait;

/// 异步静态 Runner（固定 3 段）：Source(N1::In) -> AsyncPipeline3 -> Sink(N3::Out)。
pub struct AsyncStaticRunner3<S, K, N1, N2, N3>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    N3::Out: Send + 'static,
    S: AudioSource<Out = N1::In> + Send + 'static,
    K: AudioSink<In = N3::Out> + Send + 'static,
{
    inner: AsyncAutoRunner<AsyncPipeline3<N1, N2, N3>, S, K>,
}

impl<S, K, N1, N2, N3> AsyncStaticRunner3<S, K, N1, N2, N3>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    N3::Out: Send + 'static,
    S: AudioSource<Out = N1::In> + Send + 'static,
    K: AudioSink<In = N3::Out> + Send + 'static,
{
    pub fn new(source: S, pipeline: AsyncPipeline3<N1, N2, N3>, sink: K) -> Self {
        Self {
            inner: AsyncAutoRunner::new(source, pipeline, sink),
        }
    }
}

#[async_trait(?Send)]
impl<S, K, N1, N2, N3> AsyncRunner for AsyncStaticRunner3<S, K, N1, N2, N3>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    N3::Out: Send + 'static,
    S: AudioSource<Out = N1::In> + Send + 'static,
    K: AudioSink<In = N3::Out> + Send + 'static,
{
    async fn execute(&mut self) -> RunnerResult<()> {
        self.inner.execute().await
    }
}


