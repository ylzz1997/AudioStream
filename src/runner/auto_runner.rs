use crate::codec::error::CodecError;
use crate::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::AsyncPipeline;
use crate::pipeline::node::node_interface::NodeBuffer;
use crate::runner::audio_sink::AudioSink;
use crate::runner::audio_source::AudioSource;
use crate::runner::async_runner_interface::AsyncRunner;
use crate::runner::error::RunnerResult;
use core::pin::Pin;

/// 自动 Runner：给定一个 `AudioSource`、一个 `AsyncPipeline`、一个 `AudioSink`，
/// 自动完成 push / drain / flush / finalize 的完整驱动。
pub struct AutoRunner<P, S, K>
where
    P: AsyncPipeline,
    S: AudioSource<Out = P::In>,
    K: AudioSink<In = P::Out>,
{
    pub source: S,
    pub pipeline: P,
    pub sink: K,
}

impl<P, S, K> AutoRunner<P, S, K>
where
    P: AsyncPipeline,
    S: AudioSource<Out = P::In>,
    K: AudioSink<In = P::Out>,
{
    pub fn new(source: S, pipeline: P, sink: K) -> Self {
        Self {
            source,
            pipeline,
            sink,
        }
    }
}

impl<S, K> AutoRunner<AsyncDynPipeline, S, K>
where
    S: AudioSource<Out = NodeBuffer>,
    K: AudioSink<In = NodeBuffer>,
{
    pub fn from_nodes(source: S, nodes: Vec<Box<dyn DynNode>>, sink: K) -> RunnerResult<Self> {
        let pipeline = AsyncDynPipeline::new(nodes)?;
        Ok(Self::new(source, pipeline, sink))
    }
}

impl<P, S, K> AsyncRunner for AutoRunner<P, S, K>
where
    P: AsyncPipeline,
    S: AudioSource<Out = P::In>,
    K: AudioSink<In = P::Out>,
{
    fn execute<'a>(&'a mut self) -> Pin<Box<dyn core::future::Future<Output = RunnerResult<()>> + 'a>> {
        Box::pin(async move {
            // ---- feed ----
            'feed: while let Some(v) = self.source.pull()? {
                self.pipeline.push_frame(v)?;
                // 尽可能把末端输出写入 sink，避免内存堆积
                loop {
                    match self.pipeline.try_get_frame() {
                        Ok(out) => self.sink.push(out)?,
                        Err(CodecError::Again) => break,
                        Err(CodecError::Eof) => break 'feed,
                        Err(e) => return Err(e.into()),
                    }
                }
            }

            // ---- flush ----
            self.pipeline.flush()?;
            loop {
                match self.pipeline.get_frame().await {
                    Ok(out) => self.sink.push(out)?,
                    Err(CodecError::Eof) => break,
                    Err(e) => return Err(e.into()),
                }
            }

            self.sink.finalize()?;
            Ok(())
        })
    }
}


