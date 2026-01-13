use crate::pipeline::node::dynamic_node_interface::DynPipeline;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::NodeBuffer;
use crate::pipeline::sink::audio_sink::AudioSink;
use crate::pipeline::source::audio_source::AudioSource;
use crate::runner::error::RunnerResult;
use crate::runner::runner_interface::Runner;

/// 同步动态 Runner：Source -> DynPipeline -> Sink。
pub struct DynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer>,
    K: AudioSink<In = NodeBuffer>,
{
    pub source: S,
    pub pipeline: DynPipeline,
    pub sink: K,
}

impl<S, K> DynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer>,
    K: AudioSink<In = NodeBuffer>,
{
    pub fn new(source: S, pipeline: DynPipeline, sink: K) -> Self {
        Self {
            source,
            pipeline,
            sink,
        }
    }

    /// 直接传入节点列表（可包含 1 个节点），Runner 内部会构建 `DynPipeline` 并校验 kind。
    pub fn from_nodes(source: S, nodes: Vec<Box<dyn DynNode>>, sink: K) -> RunnerResult<Self> {
        let pipeline = DynPipeline::new(nodes)?;
        Ok(Self::new(source, pipeline, sink))
    }

    fn drain_to_sink(&mut self, outs: Vec<NodeBuffer>) -> RunnerResult<()> {
        for out in outs {
            self.sink.push(out)?;
        }
        Ok(())
    }
}

impl<S, K> Runner for DynRunner<S, K>
where
    S: AudioSource<Out = NodeBuffer>,
    K: AudioSink<In = NodeBuffer>,
{
    fn execute(&mut self) -> RunnerResult<()> {
        // 逐帧推进
        while let Some(buf) = self.source.pull()? {
            let outs = self.pipeline.push_and_drain(Some(buf))?;
            self.drain_to_sink(outs)?;
        }

        // flush 并把剩余输出写完
        let outs = self.pipeline.push_and_drain(None)?;
        self.drain_to_sink(outs)?;
        loop {
            let outs = self.pipeline.drain_all()?;
            if outs.is_empty() {
                break;
            }
            self.drain_to_sink(outs)?;
        }

        self.sink.finalize()?;
        Ok(())
    }
}


