use crate::codec::error::{CodecError, CodecResult};
use crate::pipeline::forkjoinnode::fork_join_node_interface::AsyncResettablePipeline;
use crate::pipeline::forkjoinnode::reduce::Reduce;
use crate::pipeline::node::async_static_node_interface::AsyncPipeline3;
use crate::pipeline::node::node_interface::AsyncPipeline;
use crate::pipeline::node::static_node_interface::StaticNode;
use std::collections::VecDeque;

/// 异步静态 Fork/Join 节点：
/// - 分支使用 tokio pipeline（实现 `AsyncPipeline`）
/// - 节点本身实现 `StaticNode`，可插入 `AsyncPipeline3` 的某个 stage
/// - `pull()` 中通过 `try_get_frame()` 轮询收集各分支输出，再对齐 reduce
pub struct AsyncForkJoinStaticNode<P, R, TIn, TOut>
where
    P: AsyncPipeline<In = TIn, Out = TOut> + AsyncResettablePipeline + Send + 'static,
    TIn: Clone + Send + 'static,
    TOut: Send + 'static,
    R: Reduce<TOut> + Send + 'static,
{
    pipelines: Vec<P>,
    reducer: R,

    branch_q: Vec<VecDeque<TOut>>,
    branch_eof: Vec<bool>,
    out_q: VecDeque<TOut>,

    flushed: bool,
    done: bool,

    _phantom: core::marker::PhantomData<(TIn, TOut)>,
}

impl<P, R, TIn, TOut> AsyncForkJoinStaticNode<P, R, TIn, TOut>
where
    P: AsyncPipeline<In = TIn, Out = TOut> + AsyncResettablePipeline + Send + 'static,
    TIn: Clone + Send + 'static,
    TOut: Send + 'static,
    R: Reduce<TOut> + Send + 'static,
{
    pub fn new(pipelines: Vec<P>, reducer: R) -> CodecResult<Self> {
        if pipelines.is_empty() {
            return Err(CodecError::InvalidData("async fork-join static requires at least 1 pipeline"));
        }

        let mut branch_q = Vec::with_capacity(pipelines.len());
        for _ in 0..pipelines.len() {
            branch_q.push(VecDeque::new());
        }

        Ok(Self {
            branch_q,
            branch_eof: vec![false; pipelines.len()],
            pipelines,
            reducer,
            out_q: VecDeque::new(),
            flushed: false,
            done: false,
            _phantom: core::marker::PhantomData,
        })
    }

    fn pump_branch(&mut self, idx: usize) -> CodecResult<()> {
        if self.branch_eof[idx] {
            return Ok(());
        }
        loop {
            match self.pipelines[idx].try_get_frame() {
                Ok(v) => self.branch_q[idx].push_back(v),
                Err(CodecError::Again) => return Ok(()),
                Err(CodecError::Eof) => {
                    self.branch_eof[idx] = true;
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn pump_all(&mut self) -> CodecResult<()> {
        for i in 0..self.pipelines.len() {
            self.pump_branch(i)?;
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

impl<P, R, TIn, TOut> StaticNode for AsyncForkJoinStaticNode<P, R, TIn, TOut>
where
    P: AsyncPipeline<In = TIn, Out = TOut> + AsyncResettablePipeline + Send + 'static,
    TIn: Clone + Send + 'static,
    TOut: Send + 'static,
    R: Reduce<TOut> + Send + 'static,
{
    type In = TIn;
    type Out = TOut;

    fn name(&self) -> &'static str {
        "async-fork-join-static"
    }

    fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
        if self.done {
            return Err(CodecError::InvalidState("async fork-join static already eof"));
        }
        if self.flushed && input.is_some() {
            return Err(CodecError::InvalidState("async fork-join static received data after flush"));
        }

        match input {
            Some(v) => {
                for p in self.pipelines.iter() {
                    p.push_frame(v.clone())?;
                }
            }
            None => {
                self.flushed = true;
                for p in self.pipelines.iter() {
                    p.flush()?;
                }
            }
        }

        self.pump_all()?;
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

        let any_hard_eof = self
            .branch_eof
            .iter()
            .zip(self.branch_q.iter())
            .any(|(&eof, q)| eof && q.is_empty());

        if self.flushed && (any_hard_eof || self.branch_q.iter().all(|q| q.is_empty())) {
            self.done = true;
            return Err(CodecError::Eof);
        }

        Err(CodecError::Again)
    }

    fn reset(&mut self) -> CodecResult<()> {
        // reset 需要 tokio runtime：当前节点通常运行在 `AsyncPipeline3` 的 spawn_blocking 线程里。
        let handle = tokio::runtime::Handle::try_current()
            .map_err(|_| CodecError::InvalidState("async fork-join static reset requires tokio runtime"))?;

        for p in self.pipelines.iter() {
            handle.block_on(p.reset(true))?;
        }

        for q in self.branch_q.iter_mut() {
            q.clear();
        }
        self.out_q.clear();
        self.branch_eof.fill(false);
        self.flushed = false;
        self.done = false;
        Ok(())
    }
}

/// 为现有 `AsyncPipeline3` 提供 `AsyncResettablePipeline` 适配（让 fork-join 节点能 reset 分支）。
#[async_trait::async_trait(?Send)]
impl<N1, N2, N3> AsyncResettablePipeline for AsyncPipeline3<N1, N2, N3>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    N3::Out: Send + 'static,
{
    async fn reset(&self, force: bool) -> CodecResult<()> {
        AsyncPipeline3::reset(self, force).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::forkjoinnode::reduce::Sum;
    use std::collections::VecDeque;

    struct AddNode {
        q: VecDeque<i32>,
        flushed: bool,
        add: i32,
    }

    impl AddNode {
        fn new(add: i32) -> Self {
            Self {
                q: VecDeque::new(),
                flushed: false,
                add,
            }
        }
    }

    impl StaticNode for AddNode {
        type In = i32;
        type Out = i32;

        fn name(&self) -> &'static str {
            "add"
        }

        fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
            match input {
                None => {
                    self.flushed = true;
                    Ok(())
                }
                Some(v) => {
                    self.q.push_back(v + self.add);
                    Ok(())
                }
            }
        }

        fn pull(&mut self) -> CodecResult<Self::Out> {
            if let Some(v) = self.q.pop_front() {
                return Ok(v);
            }
            if self.flushed {
                return Err(CodecError::Eof);
            }
            Err(CodecError::Again)
        }

        fn reset(&mut self) -> CodecResult<()> {
            self.q.clear();
            self.flushed = false;
            Ok(())
        }
    }

    struct IdNode {
        q: VecDeque<i32>,
        flushed: bool,
    }

    impl IdNode {
        fn new() -> Self {
            Self {
                q: VecDeque::new(),
                flushed: false,
            }
        }
    }

    impl StaticNode for IdNode {
        type In = i32;
        type Out = i32;

        fn name(&self) -> &'static str {
            "id"
        }

        fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
            match input {
                None => {
                    self.flushed = true;
                    Ok(())
                }
                Some(v) => {
                    self.q.push_back(v);
                    Ok(())
                }
            }
        }

        fn pull(&mut self) -> CodecResult<Self::Out> {
            if let Some(v) = self.q.pop_front() {
                return Ok(v);
            }
            if self.flushed {
                return Err(CodecError::Eof);
            }
            Err(CodecError::Again)
        }

        fn reset(&mut self) -> CodecResult<()> {
            self.q.clear();
            self.flushed = false;
            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn async_fork_join_static_sum() {
        let p1 = AsyncPipeline3::new(AddNode::new(1), IdNode::new(), IdNode::new());
        let p2 = AsyncPipeline3::new(AddNode::new(2), IdNode::new(), IdNode::new());

        let mut n = AsyncForkJoinStaticNode::new(vec![p1, p2], Sum::<i32>::default()).unwrap();

        n.push(Some(10)).unwrap();
        let out = loop {
            match n.pull() {
                Ok(v) => break v,
                Err(CodecError::Again) => {
                    tokio::task::yield_now().await;
                    continue;
                }
                Err(e) => panic!("unexpected err: {e:?}"),
            }
        };
        // (10+1) + (10+2) = 23
        assert_eq!(out, 23);

        n.push(None).unwrap();
        loop {
            match n.pull() {
                Ok(_) => continue,
                Err(CodecError::Again) => {
                    tokio::task::yield_now().await;
                    continue;
                }
                Err(CodecError::Eof) => break,
                Err(e) => panic!("unexpected err: {e:?}"),
            }
        }
    }
}

