use crate::codec::error::{CodecError, CodecResult};
use crate::pipeline::forkjoinnode::reduce::Reduce;
use crate::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::AsyncPipeline;
use crate::pipeline::node::node_interface::{NodeBuffer, NodeBufferKind};
use std::collections::VecDeque;

/// 异步动态 Fork/Join 节点：
/// - 分支使用 `AsyncDynPipeline`
/// - 节点本身仍实现 `DynNode`（用于插入 `AsyncDynPipeline` 作为 stage）
/// - `pull()` 中使用 `try_get_frame()` 轮询收集各分支输出，再对齐 reduce
pub struct AsyncForkJoinNode<R>
where
    R: Reduce<NodeBuffer>,
{
    pipelines: Vec<AsyncDynPipeline>,
    reducer: R,

    in_kind: NodeBufferKind,
    out_kind: NodeBufferKind,

    branch_q: Vec<VecDeque<NodeBuffer>>,
    branch_eof: Vec<bool>,
    out_q: VecDeque<NodeBuffer>,

    flushed: bool,
    done: bool,
}

impl<R> AsyncForkJoinNode<R>
where
    R: Reduce<NodeBuffer>,
{
    pub fn new(pipelines: Vec<AsyncDynPipeline>, reducer: R) -> CodecResult<Self> {
        if pipelines.is_empty() {
            return Err(CodecError::InvalidData("async fork-join requires at least 1 pipeline"));
        }

        let in_kind = pipelines[0].input_kind();
        let out_kind = pipelines[0].output_kind();
        for p in pipelines.iter().skip(1) {
            if p.input_kind() != in_kind {
                return Err(CodecError::InvalidData("async fork-join pipeline input kind mismatch"));
            }
            if p.output_kind() != out_kind {
                return Err(CodecError::InvalidData("async fork-join pipeline output kind mismatch"));
            }
        }

        Ok(Self {
            branch_q: vec![VecDeque::new(); pipelines.len()],
            branch_eof: vec![false; pipelines.len()],
            pipelines,
            reducer,
            in_kind,
            out_kind,
            out_q: VecDeque::new(),
            flushed: false,
            done: false,
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

impl<R> DynNode for AsyncForkJoinNode<R>
where
    R: Reduce<NodeBuffer>,
{
    fn name(&self) -> &'static str {
        "async-fork-join"
    }

    fn input_kind(&self) -> NodeBufferKind {
        self.in_kind
    }

    fn output_kind(&self) -> NodeBufferKind {
        self.out_kind
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()> {
        if self.done {
            return Err(CodecError::InvalidState("async fork-join already eof"));
        }
        if self.flushed && input.is_some() {
            return Err(CodecError::InvalidState("async fork-join received data after flush"));
        }

        match input {
            Some(buf) => {
                if buf.kind() != self.in_kind {
                    return Err(CodecError::InvalidData("async fork-join input kind mismatch"));
                }
                for p in self.pipelines.iter() {
                    p.push_frame(buf.clone())?;
                }
            }
            None => {
                self.flushed = true;
                for p in self.pipelines.iter() {
                    p.flush()?;
                }
            }
        }

        // push 后先尝试捞一次输出（避免延迟）
        self.pump_all()?;
        self.try_join_reduce()?;
        Ok(())
    }

    fn pull(&mut self) -> CodecResult<NodeBuffer> {
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

        // 若任一分支已经 eof 且无缓存输出，则无法再对齐 join（其余分支残余输出会被丢弃）
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
        // AsyncDynPipeline 的 reset 是 async 的，这里运行在 spawn_blocking 线程里，允许 block_on。
        let handle = tokio::runtime::Handle::try_current()
            .map_err(|_| CodecError::InvalidState("async fork-join reset requires tokio runtime"))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::packet::CodecPacket;
    use crate::common::audio::audio::Rational;
    use std::collections::VecDeque;

    struct PacketAddNode {
        q: VecDeque<NodeBuffer>,
        flushed: bool,
        add: u8,
    }

    impl PacketAddNode {
        fn new(add: u8) -> Self {
            Self {
                q: VecDeque::new(),
                flushed: false,
                add,
            }
        }
    }

    impl DynNode for PacketAddNode {
        fn name(&self) -> &'static str {
            "packet-add"
        }
        fn input_kind(&self) -> NodeBufferKind {
            NodeBufferKind::Packet
        }
        fn output_kind(&self) -> NodeBufferKind {
            NodeBufferKind::Packet
        }
        fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()> {
            match input {
                None => {
                    self.flushed = true;
                    Ok(())
                }
                Some(NodeBuffer::Packet(mut p)) => {
                    let base = p.data.get(0).copied().unwrap_or(0);
                    p.data = vec![base.wrapping_add(self.add)];
                    self.q.push_back(NodeBuffer::Packet(p));
                    Ok(())
                }
                Some(_) => Err(CodecError::InvalidData("expects packet")),
            }
        }
        fn pull(&mut self) -> CodecResult<NodeBuffer> {
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
    async fn async_fork_join_dynamic_packet_reduce() {
        let p1 = AsyncDynPipeline::new(vec![Box::new(PacketAddNode::new(1))]).unwrap();
        let p2 = AsyncDynPipeline::new(vec![Box::new(PacketAddNode::new(2))]).unwrap();

        let reducer = |items: &[NodeBuffer]| -> CodecResult<NodeBuffer> {
            let NodeBuffer::Packet(a) = &items[0] else {
                return Err(CodecError::InvalidData("expected packet"));
            };
            let NodeBuffer::Packet(b) = &items[1] else {
                return Err(CodecError::InvalidData("expected packet"));
            };
            let sum = a.data[0].wrapping_add(b.data[0]);
            Ok(NodeBuffer::Packet(CodecPacket::new(vec![sum], Rational::new(1, 1))))
        };

        let mut n = AsyncForkJoinNode::new(vec![p1, p2], reducer).unwrap();
        let input = NodeBuffer::Packet(CodecPacket::new(vec![10], Rational::new(1, 1)));
        n.push(Some(input)).unwrap();

        // pull until output appears
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

        let NodeBuffer::Packet(p) = out else {
            panic!("expected packet out");
        };
        assert_eq!(p.data, vec![23]);

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

