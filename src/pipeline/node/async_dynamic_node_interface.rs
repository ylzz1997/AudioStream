//! tokio 版本的异步（后台并行）运行时 pipeline。

use crate::codec::error::{CodecError, CodecResult};
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::{AsyncPipeline, AsyncPipelineProducer, AsyncPipelineConsumer, AsyncPipelineEndpoint, NodeBuffer};
use tokio::sync::mpsc;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// pipeline 内部消息：
/// - `Ok(Some(buf))`：一条数据
/// - `Ok(None)`：flush（输入结束）
/// - `Err(e)`：错误（会沿链路传递到末端）
type Msg = Result<Option<NodeBuffer>, CodecError>;

fn store_err(store: &Arc<Mutex<Option<CodecError>>>, e: &CodecError) {
    // 只记录“第一个”错误，避免后续噪音覆盖根因
    let mut g = store.lock().expect("err_store poisoned");
    if g.is_none() {
        *g = Some(e.clone());
    }
}

fn err_or_closed(store: &Arc<Mutex<Option<CodecError>>>) -> CodecError {
    store
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .unwrap_or(CodecError::InvalidState("tokio pipeline input channel closed"))
}

fn drain_to_next(node: &mut Box<dyn DynNode>, tx_next: &mpsc::UnboundedSender<Msg>) -> Result<(), CodecError> {
    loop {
        match node.pull() {
            Ok(v) => {
                let _ = tx_next.send(Ok(Some(v)));
            }
            Err(CodecError::Again) => return Ok(()),
            Err(CodecError::Eof) => return Ok(()),
            Err(e) => {
                let _ = tx_next.send(Err(e.clone()));
                return Err(e);
            }
        }
    }
}

fn spawn_stage(
    mut node: Box<dyn DynNode>,
    mut rx: mpsc::UnboundedReceiver<Msg>,
    tx_next: mpsc::UnboundedSender<Msg>,
    err_store: Arc<Mutex<Option<CodecError>>>,
) {
    tokio::task::spawn_blocking(move || {
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                Err(e) => {
                    store_err(&err_store, &e);
                    let _ = tx_next.send(Err(e));
                    break;
                }
                Ok(None) => {
                    // flush
                    let _ = node.push(None);
                    let _ = drain_to_next(&mut node, &tx_next);
                    let _ = tx_next.send(Ok(None));
                    break;
                }
                Ok(Some(buf)) => match node.push(Some(buf)) {
                    Ok(()) | Err(CodecError::Again) => {
                        let _ = drain_to_next(&mut node, &tx_next);
                    }
                    Err(CodecError::Eof) => {
                        let _ = tx_next.send(Ok(None));
                        break;
                    }
                    Err(e) => {
                        store_err(&err_store, &e);
                        let _ = tx_next.send(Err(e));
                        break;
                    }
                },
            }
        }
        // 上游断开：视为 flush
        let _ = tx_next.send(Ok(None));
    });
}

fn spawn_last_stage(
    mut node: Box<dyn DynNode>,
    mut rx: mpsc::UnboundedReceiver<Msg>,
    out_tx: mpsc::UnboundedSender<Result<NodeBuffer, CodecError>>,
    err_store: Arc<Mutex<Option<CodecError>>>,
) {
    tokio::task::spawn_blocking(move || {
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                Err(e) => {
                    store_err(&err_store, &e);
                    let _ = out_tx.send(Err(e));
                    return;
                }
                Ok(None) => {
                    let _ = node.push(None);
                    // flush 后尽可能 drain 到 Eof
                    loop {
                        match node.pull() {
                            Ok(v) => {
                                let _ = out_tx.send(Ok(v));
                            }
                            Err(CodecError::Again) => continue,
                            Err(CodecError::Eof) => {
                                let _ = out_tx.send(Err(CodecError::Eof));
                                return;
                            }
                            Err(e) => {
                                store_err(&err_store, &e);
                                let _ = out_tx.send(Err(e));
                                return;
                            }
                        }
                    }
                }
                Ok(Some(buf)) => match node.push(Some(buf)) {
                    Ok(()) | Err(CodecError::Again) => {
                        loop {
                            match node.pull() {
                                Ok(v) => {
                                    let _ = out_tx.send(Ok(v));
                                }
                                Err(CodecError::Again) => break,
                                Err(CodecError::Eof) => {
                                    let _ = out_tx.send(Err(CodecError::Eof));
                                    return;
                                }
                                Err(e) => {
                                    store_err(&err_store, &e);
                                    let _ = out_tx.send(Err(e));
                                    return;
                                }
                            }
                        }
                    }
                    Err(CodecError::Eof) => {
                        let _ = out_tx.send(Err(CodecError::Eof));
                        return;
                    }
                    Err(e) => {
                        store_err(&err_store, &e);
                        let _ = out_tx.send(Err(e));
                        return;
                    }
                },
            }
        }
        // 上游断开：视为结束
        let _ = out_tx.send(Err(CodecError::Eof));
    });
}

/// Async 异步运行时 pipeline：push 后后台并行流转，末端通过 get/try_get 输出。
pub struct AsyncDynPipeline {
    in_tx: mpsc::UnboundedSender<Msg>,
    out_rx: mpsc::UnboundedReceiver<Result<NodeBuffer, CodecError>>,
    err_store: Arc<Mutex<Option<CodecError>>>,
}

pub struct AsyncDynPipelineProducer {
    tx: mpsc::UnboundedSender<Msg>,
    err_store: Arc<Mutex<Option<CodecError>>>,
}

pub struct AsyncDynPipelineConsumer {
    rx: mpsc::UnboundedReceiver<Result<NodeBuffer, CodecError>>,
}

impl AsyncDynPipelineProducer {
    pub fn push_frame(&self, buf: NodeBuffer) -> CodecResult<()> {
        self.tx
            .send(Ok(Some(buf)))
            .map_err(|_| err_or_closed(&self.err_store))
    }

    pub fn flush(&self) -> CodecResult<()> {
        self.tx
            .send(Ok(None))
            .map_err(|_| err_or_closed(&self.err_store))
    }
}

impl AsyncPipelineProducer for AsyncDynPipelineProducer {
    type In = NodeBuffer;
    fn push_frame(&self, frame: Self::In) -> CodecResult<()> {
        self.push_frame(frame)
    }
    fn flush(&self) -> CodecResult<()> {
        self.flush()
    }
}

impl AsyncDynPipelineConsumer {
    pub fn try_get_frame(&mut self) -> CodecResult<NodeBuffer> {
        match self.rx.try_recv() {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(e),
            Err(mpsc::error::TryRecvError::Empty) => Err(CodecError::Again),
            Err(mpsc::error::TryRecvError::Disconnected) => Err(CodecError::Eof),
        }
    }

    pub async fn get_frame(&mut self) -> CodecResult<NodeBuffer> {
        match self.rx.recv().await {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e),
            None => Err(CodecError::Eof),
        }
    }
}

#[async_trait]
impl AsyncPipelineConsumer for AsyncDynPipelineConsumer {
    type Out = NodeBuffer;
    fn try_get_frame(&mut self) -> CodecResult<Self::Out> {
        self.try_get_frame()
    }
    async fn get_frame(&mut self) -> CodecResult<Self::Out> {
        self.get_frame().await
    }
}

impl AsyncDynPipeline {
    pub fn new(nodes: Vec<Box<dyn DynNode>>) -> Result<Self, CodecError> {
        if nodes.is_empty() {
            return Err(CodecError::InvalidData("pipeline requires at least 1 node"));
        }
        // 基础校验：相邻节点 output_kind == input_kind
        for i in 0..(nodes.len() - 1) {
            let ok = nodes[i].output_kind() == nodes[i + 1].input_kind();
            if !ok {
                return Err(CodecError::InvalidData("pipeline node kind mismatch"));
            }
        }

        let (in_tx, mut rx) = mpsc::unbounded_channel::<Msg>();
        let err_store: Arc<Mutex<Option<CodecError>>> = Arc::new(Mutex::new(None));

        let mut iter = nodes.into_iter();
        let mut current = iter
            .next()
            .ok_or(CodecError::InvalidState("pipeline requires at least 1 node"))?;

        while let Some(next_node) = iter.next() {
            let (tx_next, rx_next) = mpsc::unbounded_channel::<Msg>();
            spawn_stage(current, rx, tx_next, err_store.clone());
            current = next_node;
            rx = rx_next;
        }

        let (out_tx, out_rx) = mpsc::unbounded_channel::<Result<NodeBuffer, CodecError>>();
        spawn_last_stage(current, rx, out_tx, err_store.clone());

        Ok(Self {
            in_tx,
            out_rx,
            err_store,
        })
    }

    /// 拆分为输入端/输出端，用于 runner 并行驱动。
    pub fn endpoints(self) -> (AsyncDynPipelineProducer, AsyncDynPipelineConsumer) {
        (
            AsyncDynPipelineProducer {
                tx: self.in_tx,
                err_store: self.err_store,
            },
            AsyncDynPipelineConsumer { rx: self.out_rx },
        )
    }
}

#[async_trait]
impl AsyncPipeline for AsyncDynPipeline {
    type In = NodeBuffer;
    type Out = NodeBuffer;

    fn push_frame(&self, buf: NodeBuffer) -> CodecResult<()> {
        self.in_tx
            .send(Ok(Some(buf)))
            .map_err(|_| err_or_closed(&self.err_store))
    }

    fn flush(&self) -> CodecResult<()> {
        self.in_tx
            .send(Ok(None))
            .map_err(|_| err_or_closed(&self.err_store))
    }

    /// 非阻塞取末端输出。
    fn try_get_frame(&mut self) -> CodecResult<NodeBuffer> {
        match self.out_rx.try_recv() {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(e),
            Err(mpsc::error::TryRecvError::Empty) => Err(CodecError::Again),
            Err(mpsc::error::TryRecvError::Disconnected) => Err(CodecError::Eof),
        }
    }

    /// 异步等待一个末端输出。
    async fn get_frame(&mut self) -> CodecResult<NodeBuffer> {
        match self.out_rx.recv().await {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e),
            None => Err(CodecError::Eof),
        }
    }
}

impl AsyncPipelineEndpoint for AsyncDynPipeline {
    type In = NodeBuffer;
    type Out = NodeBuffer;
    type Producer = AsyncDynPipelineProducer;
    type Consumer = AsyncDynPipelineConsumer;

    fn endpoints(self) -> (Self::Producer, Self::Consumer) {
        self.endpoints()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::packet::CodecPacket;
    use crate::common::audio::audio::Rational;
    use crate::pipeline::node::dynamic_node_interface::DynNode;
    use crate::pipeline::node::node_interface::{NodeBuffer, NodeBufferKind};
    use std::collections::VecDeque;

    struct PacketEchoNode {
        q: VecDeque<NodeBuffer>,
        flushed: bool,
    }

    impl PacketEchoNode {
        fn new() -> Self {
            Self {
                q: VecDeque::new(),
                flushed: false,
            }
        }
    }

    impl DynNode for PacketEchoNode {
        fn name(&self) -> &'static str {
            "packet-echo"
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
                Some(NodeBuffer::Packet(p)) => {
                    // echo: prefix 0xAA
                    let mut d = vec![0xAA];
                    d.extend_from_slice(&p.data);
                    let mut out = CodecPacket::new(d, p.time_base);
                    out.pts = p.pts;
                    out.duration = p.duration;
                    self.q.push_back(NodeBuffer::Packet(out));
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
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn async_dyn_pipeline_push_then_get() {
        let nodes: Vec<Box<dyn DynNode>> = vec![Box::new(PacketEchoNode::new()), Box::new(PacketEchoNode::new())];
        let mut p = AsyncDynPipeline::new(nodes).unwrap();

        let tb = Rational::new(1, 1);
        p.push_frame(NodeBuffer::Packet(CodecPacket::new(vec![1, 2, 3], tb)))
            .unwrap();
        p.push_frame(NodeBuffer::Packet(CodecPacket::new(vec![4], tb))).unwrap();
        p.flush().unwrap();

        let mut outs = Vec::new();
        loop {
            match p.get_frame().await {
                Ok(NodeBuffer::Packet(pkt)) => outs.push(pkt.data),
                Err(CodecError::Eof) => break,
                Err(e) => panic!("unexpected err: {e:?}"),
                _ => panic!("unexpected kind"),
            }
        }
        assert_eq!(outs, vec![vec![0xAA, 0xAA, 1, 2, 3], vec![0xAA, 0xAA, 4]]);
    }
}


