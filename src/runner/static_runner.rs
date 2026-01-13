use crate::pipeline::node::static_node_interface::{Pipeline3, StaticNode};
use crate::pipeline::sink::audio_sink::AudioSink;
use crate::pipeline::source::audio_source::AudioSource;
use crate::runner::error::RunnerResult;
use crate::runner::runner_interface::Runner;

/// 同步静态 Runner（固定 3 段）：Source -> Pipeline3 -> Sink。
pub struct StaticRunner3<S, K, N1, N2, N3>
where
    N1: StaticNode,
    N2: StaticNode<In = N1::Out>,
    N3: StaticNode<In = N2::Out>,
    S: AudioSource<Out = N1::In>,
    K: AudioSink<In = N3::Out>,
{
    pub source: S,
    pub pipeline: Pipeline3<N1, N2, N3>,
    pub sink: K,
}

impl<S, K, N1, N2, N3> StaticRunner3<S, K, N1, N2, N3>
where
    N1: StaticNode,
    N2: StaticNode<In = N1::Out>,
    N3: StaticNode<In = N2::Out>,
    S: AudioSource<Out = N1::In>,
    K: AudioSink<In = N3::Out>,
{
    pub fn new(source: S, pipeline: Pipeline3<N1, N2, N3>, sink: K) -> Self {
        Self {
            source,
            pipeline,
            sink,
        }
    }
}

impl<S, K, N1, N2, N3> Runner for StaticRunner3<S, K, N1, N2, N3>
where
    N1: StaticNode,
    N2: StaticNode<In = N1::Out>,
    N3: StaticNode<In = N2::Out>,
    S: AudioSource<Out = N1::In>,
    K: AudioSink<In = N3::Out>,
{
    fn execute(&mut self) -> RunnerResult<()> {
        while let Some(v) = self.source.pull()? {
            let outs = self.pipeline.push_and_drain(Some(v))?;
            for out in outs {
                self.sink.push(out)?;
            }
        }

        let outs = self.pipeline.push_and_drain(None)?;
        for out in outs {
            self.sink.push(out)?;
        }
        loop {
            let outs = self.pipeline.drain_all()?;
            if outs.is_empty() {
                break;
            }
            for out in outs {
                self.sink.push(out)?;
            }
        }

        self.sink.finalize()?;
        Ok(())
    }
}


