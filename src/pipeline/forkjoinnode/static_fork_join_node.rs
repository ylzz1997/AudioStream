use crate::codec::error::{CodecError, CodecResult};
use crate::pipeline::forkjoinnode::fork_join_node_interface::SyncStaticPipeline;
use crate::pipeline::forkjoinnode::reduce::Reduce;
use crate::pipeline::node::static_node_interface::{Pipeline3, StaticNode};
use std::collections::VecDeque;

/// 同步静态 Fork/Join 节点：
/// - In/Out 在类型层面固定（编译期校验）
/// - 内部包含多条同步静态 pipeline（实现 `SyncStaticPipeline`）
/// - 对齐 join：每条分支各取 1 个输出，reduce 成 1 个输出
pub struct ForkJoinStaticNode<P, R>
where
    P: SyncStaticPipeline,
    R: Reduce<P::Out>,
{
    pipelines: Vec<P>,
    reducer: R,

    branch_q: Vec<VecDeque<P::Out>>,
    out_q: VecDeque<P::Out>,

    flushed: bool,
    done: bool,
}

impl<P, R> ForkJoinStaticNode<P, R>
where
    P: SyncStaticPipeline,
    R: Reduce<P::Out>,
{
    pub fn new(pipelines: Vec<P>, reducer: R) -> CodecResult<Self> {
        if pipelines.is_empty() {
            return Err(CodecError::InvalidData("fork-join static requires at least 1 pipeline"));
        }

        let mut branch_q = Vec::with_capacity(pipelines.len());
        for _ in 0..pipelines.len() {
            branch_q.push(VecDeque::new());
        }

        Ok(Self {
            branch_q,
            pipelines,
            reducer,
            out_q: VecDeque::new(),
            flushed: false,
            done: false,
        })
    }

    fn pump_all(&mut self) -> CodecResult<()> {
        for (i, p) in self.pipelines.iter_mut().enumerate() {
            let outs = p.drain_all()?;
            self.branch_q[i].extend(outs);
        }
        Ok(())
    }

    fn try_join_reduce(&mut self) -> CodecResult<()> {
        loop {
            if self.branch_q.iter().any(|q| q.is_empty()) {
                return Ok(());
            }
            let mut items = Vec::with_capacity(self.branch_q.len());
            for q in self.branch_q.iter_mut() {
                items.push(q.pop_front().expect("checked non-empty"));
            }
            let reduced = self.reducer.reduce(&items)?;
            self.out_q.push_back(reduced);
        }
    }
}

impl<P, R> StaticNode for ForkJoinStaticNode<P, R>
where
    P: SyncStaticPipeline,
    R: Reduce<P::Out>,
{
    type In = P::In;
    type Out = P::Out;

    fn name(&self) -> &'static str {
        "fork-join-static"
    }

    fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
        if self.done {
            return Err(CodecError::InvalidState("fork-join static already eof"));
        }
        if self.flushed && input.is_some() {
            return Err(CodecError::InvalidState("fork-join static received data after flush"));
        }

        match input {
            Some(v) => {
                for (i, p) in self.pipelines.iter_mut().enumerate() {
                    let outs = p.push_and_drain(Some(v.clone()))?;
                    self.branch_q[i].extend(outs);
                }
            }
            None => {
                self.flushed = true;
                for (i, p) in self.pipelines.iter_mut().enumerate() {
                    let outs = p.push_and_drain(None)?;
                    self.branch_q[i].extend(outs);
                }
            }
        }

        self.try_join_reduce()?;
        Ok(())
    }

    fn pull(&mut self) -> CodecResult<Self::Out> {
        if let Some(v) = self.out_q.pop_front() {
            return Ok(v);
        }
        if self.done {
            return Err(CodecError::Eof);
        }

        self.pump_all()?;
        self.try_join_reduce()?;

        if let Some(v) = self.out_q.pop_front() {
            return Ok(v);
        }

        if self.flushed {
            if self.branch_q.iter().all(|q| q.is_empty()) {
                self.done = true;
                return Err(CodecError::Eof);
            }
        }

        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        for p in self.pipelines.iter_mut() {
            p.reset(true)?;
        }
        for q in self.branch_q.iter_mut() {
            q.clear();
        }
        self.out_q.clear();
        self.flushed = false;
        self.done = false;
        Ok(())
    }
}

/// 为现有 `Pipeline3` 提供 `SyncStaticPipeline` 适配。
impl<N1, N2, N3> SyncStaticPipeline for Pipeline3<N1, N2, N3>
where
    N1: StaticNode,
    N2: StaticNode<In = N1::Out>,
    N3: StaticNode<In = N2::Out>,
    N1::In: Clone,
{
    type In = N1::In;
    type Out = N3::Out;

    fn push_and_drain(&mut self, input: Option<Self::In>) -> CodecResult<Vec<Self::Out>> {
        Pipeline3::push_and_drain(self, input)
    }

    fn drain_all(&mut self) -> CodecResult<Vec<Self::Out>> {
        Pipeline3::drain_all(self)
    }

    fn reset(&mut self, force: bool) -> CodecResult<()> {
        Pipeline3::reset(self, force)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::node::static_node_interface::IdentityStaticNode;
    use crate::pipeline::forkjoinnode::reduce::Sum;

    #[test]
    fn fork_join_static_sum_default_weight() {
        let p1 = Pipeline3::new(
            IdentityStaticNode::<i32>::new(),
            IdentityStaticNode::<i32>::new(),
            IdentityStaticNode::<i32>::new(),
        );
        let p2 = Pipeline3::new(
            IdentityStaticNode::<i32>::new(),
            IdentityStaticNode::<i32>::new(),
            IdentityStaticNode::<i32>::new(),
        );

        let mut n = ForkJoinStaticNode::new(vec![p1, p2], Sum::<i32>::default()).unwrap();
        n.push(Some(7)).unwrap();
        assert_eq!(n.pull().unwrap(), 14);
        n.push(None).unwrap();
        assert!(matches!(n.pull(), Err(CodecError::Eof) | Err(CodecError::Again)));
    }
}

