//! tokio 版本的异步（后台并行）运行时 pipeline。

use crate::codec::error::{CodecError, CodecResult};
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::{
    AsyncPipeline, AsyncPipelineConsumer, AsyncPipelineEndpoint, AsyncPipelineMsg, AsyncPipelineProducer, NodeBuffer,
};
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

type Msg = AsyncPipelineMsg<NodeBuffer>;

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
                let _ = tx_next.send(Msg::Data(Ok(Some(v))));
            }
            Err(CodecError::Again) => return Ok(()),
            Err(CodecError::Eof) => return Ok(()),
            Err(e) => {
                let _ = tx_next.send(Msg::Data(Err(e.clone())));
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
                Msg::Data(Err(e)) => {
                    store_err(&err_store, &e);
                    let _ = tx_next.send(Msg::Data(Err(e)));
                    break;
                }
                Msg::Data(Ok(None)) => {
                    // flush
                    let _ = node.push(None);
                    let _ = drain_to_next(&mut node, &tx_next);
                    let _ = tx_next.send(Msg::Data(Ok(None)));
                }
                Msg::Data(Ok(Some(buf))) => match node.push(Some(buf)) {
                    Ok(()) | Err(CodecError::Again) => {
                        let _ = drain_to_next(&mut node, &tx_next);
                    }
                    Err(CodecError::Eof) => {
                        // 视作“本段提前结束”，向下游广播 flush（但 stage 继续运行，等待 reset 后再复用）
                        let _ = tx_next.send(Msg::Data(Ok(None)));
                    }
                    Err(e) => {
                        store_err(&err_store, &e);
                        let _ = tx_next.send(Msg::Data(Err(e)));
                        break;
                    }
                },
                Msg::Reset { force, ack } => {
                    if !force {
                        // 不强行打断：先把已产出的输出尽可能推进到下一段
                        let _ = drain_to_next(&mut node, &tx_next);
                    }
                    // reset 本段
                    if let Err(e) = node.reset() {
                        store_err(&err_store, &e);
                        let _ = tx_next.send(Msg::Data(Err(e.clone())));
                        break;
                    }
                    let _ = tx_next.send(Msg::Reset { force, ack });
                }
            }
        }
        // 上游断开：视为 flush
        let _ = tx_next.send(Msg::Data(Ok(None)));
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
                Msg::Data(Err(e)) => {
                    store_err(&err_store, &e);
                    let _ = out_tx.send(Err(e));
                    return;
                }
                Msg::Data(Ok(None)) => {
                    let _ = node.push(None);
                    // flush 后尽可能 drain 到 Eof（但不退出：允许 reset 后复用）
                    loop {
                        match node.pull() {
                            Ok(v) => {
                                let _ = out_tx.send(Ok(v));
                            }
                            Err(CodecError::Again) => continue,
                            Err(CodecError::Eof) => {
                                let _ = out_tx.send(Err(CodecError::Eof));
                                break;
                            }
                            Err(e) => {
                                store_err(&err_store, &e);
                                let _ = out_tx.send(Err(e));
                                break;
                            }
                        }
                    }
                }
                Msg::Data(Ok(Some(buf))) => match node.push(Some(buf)) {
                    Ok(()) | Err(CodecError::Again) => {
                        loop {
                            match node.pull() {
                                Ok(v) => {
                                    let _ = out_tx.send(Ok(v));
                                }
                                Err(CodecError::Again) => break,
                                Err(CodecError::Eof) => {
                                    let _ = out_tx.send(Err(CodecError::Eof));
                                    break;
                                }
                                Err(e) => {
                                    store_err(&err_store, &e);
                                    let _ = out_tx.send(Err(e));
                                    break;
                                }
                            }
                        }
                    }
                    Err(CodecError::Eof) => {
                        let _ = out_tx.send(Err(CodecError::Eof));
                    }
                    Err(e) => {
                        store_err(&err_store, &e);
                        let _ = out_tx.send(Err(e));
                    }
                },
                Msg::Reset { force, mut ack } => {
                    if !force {
                        // 不强行打断：先把已产出的输出尽可能吐给下游（末端就是 out_tx）
                        loop {
                            match node.pull() {
                                Ok(v) => {
                                    let _ = out_tx.send(Ok(v));
                                }
                                Err(CodecError::Again) | Err(CodecError::Eof) => break,
                                Err(e) => {
                                    store_err(&err_store, &e);
                                    let _ = out_tx.send(Err(e.clone()));
                                    if let Some(tx) = ack.take() {
                                        let _ = tx.send(Err(e));
                                    }
                                    return;
                                }
                            }
                        }
                    }

                    let res = node.reset();
                    if let Err(e) = &res {
                        store_err(&err_store, e);
                        let _ = out_tx.send(Err(e.clone()));
                    }
                    if let Some(tx) = ack.take() {
                        let _ = tx.send(res);
                    }
                }
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
            .send(Msg::Data(Ok(Some(buf))))
            .map_err(|_| err_or_closed(&self.err_store))
    }

    pub fn flush(&self) -> CodecResult<()> {
        self.tx
            .send(Msg::Data(Ok(None)))
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

    /// reset：从起点向终点传播，直到最后一段完成。
    ///
    /// - `force=false`：不强行打断节点正在处理的 flow（按顺序等它“处理到当前边界”后再 reset）
    /// - `force=true`：强制 reset（节点可丢弃内部缓存/残留）
    pub async fn reset(&self, force: bool) -> CodecResult<()> {
        let (tx, rx) = oneshot::channel::<Result<(), CodecError>>();
        self.in_tx
            .send(Msg::Reset { force, ack: Some(tx) })
            .map_err(|_| err_or_closed(&self.err_store))?;
        match rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(err_or_closed(&self.err_store)),
        }
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
            .send(Msg::Data(Ok(Some(buf))))
            .map_err(|_| err_or_closed(&self.err_store))
    }

    fn flush(&self) -> CodecResult<()> {
        self.in_tx
            .send(Msg::Data(Ok(None)))
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
