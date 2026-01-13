use crate::pipeline::node::node_interface::{AsyncPipelineConsumer, AsyncPipelineEndpoint, AsyncPipelineProducer};
use crate::runner::async_runner_interface::AsyncRunner;
use crate::pipeline::sink::audio_sink::AsyncAudioSink;
use crate::pipeline::source::audio_source::AsyncAudioSource;
use crate::runner::error::{RunnerError, RunnerResult};
use async_trait::async_trait;

/// 新版异步自动 Runner：
/// - `AsyncAudioSource` / `AsyncAudioSink` 都是 async 的
/// - 与 tokio pipeline 并行：输入 pull、pipeline stage、输出 push 同时进行
///
/// 说明：
/// - 为了并行，这里会 `tokio::spawn` 两个任务，因此要求 source/sink/pipeline 的相关类型都是 `Send + 'static`。
pub struct AsyncAutoRunner<P, S, K>
where
    P: AsyncPipelineEndpoint,
    S: AsyncAudioSource<Out = P::In>,
    K: AsyncAudioSink<In = P::Out>,
{
    pub source: Option<S>,
    pub pipeline: Option<P>,
    pub sink: Option<K>,
}

impl<P, S, K> AsyncAutoRunner<P, S, K>
where
    P: AsyncPipelineEndpoint,
    S: AsyncAudioSource<Out = P::In>,
    K: AsyncAudioSink<In = P::Out>,
{
    pub fn new(source: S, pipeline: P, sink: K) -> Self {
        Self {
            source: Some(source),
            pipeline: Some(pipeline),
            sink: Some(sink),
        }
    }
}

#[async_trait(?Send)]
impl<P, S, K> AsyncRunner for AsyncAutoRunner<P, S, K>
where
    P: AsyncPipelineEndpoint,
    S: AsyncAudioSource<Out = P::In> + Send + 'static,
    K: AsyncAudioSink<In = P::Out> + Send + 'static,
{
    async fn execute(&mut self) -> RunnerResult<()> {
        let source = self
            .source
            .take()
            .ok_or(RunnerError::InvalidState("runner already executed (source taken)"))?;
        let pipeline = self
            .pipeline
            .take()
            .ok_or(RunnerError::InvalidState("runner already executed (pipeline taken)"))?;
        let sink = self
            .sink
            .take()
            .ok_or(RunnerError::InvalidState("runner already executed (sink taken)"))?;

        let (tx, mut rx) = pipeline.endpoints();

        // ---- output task: drain pipeline -> sink ----
        let out_task = tokio::spawn(async move {
            let mut sink = sink;
            loop {
                match rx.get_frame().await {
                    Ok(v) => sink.push(v).await?,
                    Err(crate::codec::error::CodecError::Again) => continue,
                    Err(crate::codec::error::CodecError::Eof) => break,
                    Err(e) => return Err::<(), RunnerError>(e.into()),
                }
            }
            sink.finalize().await?;
            Ok::<(), RunnerError>(())
        });

        // ---- input task: source -> pipeline ----
        let in_task = tokio::spawn(async move {
            let mut source = source;
            loop {
                match source.pull().await? {
                    Some(v) => tx.push_frame(v)?,
                    None => break,
                }
            }
            tx.flush()?;
            Ok::<(), RunnerError>(())
        });

        // 等待两端结束；任一端出错就尽快返回
        let in_res = in_task.await.map_err(|_| RunnerError::InvalidState("input task join failed"))?;
        if let Err(e) = in_res {
            out_task.abort();
            return Err(e);
        }
        let out_res = out_task.await.map_err(|_| RunnerError::InvalidState("output task join failed"))?;
        out_res?;
        Ok(())
    }
}


