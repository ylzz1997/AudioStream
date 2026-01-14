use crate::codec::error::{CodecError, CodecResult};
use crate::pipeline::forkjoinnode::reduce::Reduce;
use crate::pipeline::node::dynamic_node_interface::DynPipeline;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::{NodeBuffer, NodeBufferKind};
use std::collections::VecDeque;

/// 同步动态 Fork/Join 节点：
/// - 输入/输出为 `NodeBuffer`
/// - 内部包含多条 `DynPipeline`
/// - 采用“对齐 join”：每条分支各取 1 个输出，凑齐后执行 reduce，产出 1 个输出
pub struct ForkJoinNode<R>
where
    R: Reduce<NodeBuffer>,
{
    pipelines: Vec<DynPipeline>,
    reducer: R,

    in_kind: NodeBufferKind,
    out_kind: NodeBufferKind,

    branch_q: Vec<VecDeque<NodeBuffer>>,
    out_q: VecDeque<NodeBuffer>,

    flushed: bool,
    done: bool,
}

impl<R> ForkJoinNode<R>
where
    R: Reduce<NodeBuffer>,
{
    pub fn new(pipelines: Vec<DynPipeline>, reducer: R) -> CodecResult<Self> {
        if pipelines.is_empty() {
            return Err(CodecError::InvalidData("fork-join requires at least 1 pipeline"));
        }

        let in_kind = pipelines[0].input_kind();
        let out_kind = pipelines[0].output_kind();
        for p in pipelines.iter().skip(1) {
            if p.input_kind() != in_kind {
                return Err(CodecError::InvalidData("fork-join pipeline input kind mismatch"));
            }
            if p.output_kind() != out_kind {
                return Err(CodecError::InvalidData("fork-join pipeline output kind mismatch"));
            }
        }

        Ok(Self {
            branch_q: vec![VecDeque::new(); pipelines.len()],
            pipelines,
            reducer,
            in_kind,
            out_kind,
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

impl<R> DynNode for ForkJoinNode<R>
where
    R: Reduce<NodeBuffer>,
{
    fn name(&self) -> &'static str {
        "fork-join"
    }

    fn input_kind(&self) -> NodeBufferKind {
        self.in_kind
    }

    fn output_kind(&self) -> NodeBufferKind {
        self.out_kind
    }

    fn push(&mut self, input: Option<NodeBuffer>) -> CodecResult<()> {
        if self.done {
            return Err(CodecError::InvalidState("fork-join already eof"));
        }
        if self.flushed && input.is_some() {
            return Err(CodecError::InvalidState("fork-join received data after flush"));
        }

        match input {
            Some(buf) => {
                if buf.kind() != self.in_kind {
                    return Err(CodecError::InvalidData("fork-join input kind mismatch"));
                }
                for (i, p) in self.pipelines.iter_mut().enumerate() {
                    let outs = p.push_and_drain(Some(buf.clone()))?;
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
        // 尽可能把当前队列对齐产出
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

        // 尽量推进各分支
        self.pump_all()?;
        self.try_join_reduce()?;

        if let Some(v) = self.out_q.pop_front() {
            return Ok(v);
        }

        if self.flushed {
            // flush 后若无法再产出任何对齐输出，则视为结束（残余的“非对齐”输出会被丢弃）
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::packet::CodecPacket;
    use crate::common::audio::audio::Rational;
    use crate::pipeline::node::node_interface::NodeBuffer;
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

    #[test]
    fn fork_join_dynamic_packet_reduce() {
        let p1 = DynPipeline::new(vec![Box::new(PacketAddNode::new(1))]).unwrap();
        let p2 = DynPipeline::new(vec![Box::new(PacketAddNode::new(2))]).unwrap();

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

        let mut n = ForkJoinNode::new(vec![p1, p2], reducer).unwrap();

        let input = NodeBuffer::Packet(CodecPacket::new(vec![10], Rational::new(1, 1)));
        n.push(Some(input)).unwrap();
        let out = n.pull().unwrap();

        let NodeBuffer::Packet(p) = out else {
            panic!("expected packet out");
        };
        // (10+1) + (10+2) = 23
        assert_eq!(p.data, vec![23]);

        n.push(None).unwrap();
        // drain to eof
        loop {
            match n.pull() {
                Ok(_) => continue,
                Err(CodecError::Again) => continue,
                Err(CodecError::Eof) => break,
                Err(e) => panic!("unexpected err: {e:?}"),
            }
        }
    }
}

