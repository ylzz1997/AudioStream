//! Async "node pipeline + writer" sink:
//! - Each `DynNode` runs in its own blocking stage (pipeline parallel).
//! - The final `AudioWriter` is the last stage (single-threaded write of PCM frames).
//! - All stage channels are bounded, providing backpressure to upstream.
//!
//! This sink accepts `NodeBuffer` as input, allowing internal kind transitions:
//! - PCM -> Packet (encoder)
//! - Packet -> PCM (decoder)
//! - etc.
//!
//! Guarantees / constraints:
//! - Adjacent nodes must satisfy: `node[i].output_kind() == node[i+1].input_kind()`.
//! - The **last** node must output PCM (`NodeBufferKind::Pcm`), because the writer consumes PCM.
//! - The **first** node input_kind determines what this sink accepts in `push(...)`.
//!
//! Notes:
//! - Stages are implemented with `tokio::task::spawn_blocking` because most codec / IO work is blocking.
//! - This module intentionally does NOT implement `reset()` semantics.

use crate::codec::error::CodecError;
use crate::common::audio::audio::AudioFrameView;
use crate::common::io::io::AudioWriter;
use crate::pipeline::node::dynamic_node_interface::DynNode;
use crate::pipeline::node::node_interface::{NodeBuffer, NodeBufferKind};
use super::audio_sink::AsyncAudioSink;
use crate::runner::error::{RunnerError, RunnerResult};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

#[derive(Debug)]
enum Msg {
    Data(NodeBuffer),
    Flush,
}

fn store_err(store: &Arc<Mutex<Option<String>>>, msg: String) {
    let mut g = store.lock().expect("err_store poisoned");
    if g.is_none() {
        *g = Some(msg);
    }
}

fn err_or_closed(store: &Arc<Mutex<Option<String>>>) -> RunnerError {
    let msg = store
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .unwrap_or_else(|| "async pipeline audio sink channel closed".to_string());
    RunnerError::Codec(CodecError::Other(msg))
}

fn drain_to_next(
    node: &mut Box<dyn DynNode>,
    tx_next: &mpsc::Sender<Msg>,
    err_store: &Arc<Mutex<Option<String>>>,
) -> Result<(), ()> {
    loop {
        match node.pull() {
            Ok(buf) => {
                if tx_next.blocking_send(Msg::Data(buf)).is_err() {
                    store_err(err_store, "next stage channel closed".to_string());
                    return Err(());
                }
            }
            Err(CodecError::Again) | Err(CodecError::Eof) => return Ok(()),
            Err(e) => {
                store_err(err_store, format!("node {} pull error: {e}", node.name()));
                return Err(());
            }
        }
    }
}

fn spawn_node_stage(
    mut node: Box<dyn DynNode>,
    mut rx: mpsc::Receiver<Msg>,
    tx_next: mpsc::Sender<Msg>,
    err_store: Arc<Mutex<Option<String>>>,
) {
    tokio::task::spawn_blocking(move || {
        let mut flushed = false;
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                Msg::Data(buf) => {
                    if flushed {
                        store_err(&err_store, format!("node {} received data after flush", node.name()));
                        break;
                    }
                    match node.push(Some(buf)) {
                        Ok(()) | Err(CodecError::Again) => {
                            if drain_to_next(&mut node, &tx_next, &err_store).is_err() {
                                break;
                            }
                        }
                        Err(CodecError::Eof) => {
                            let _ = tx_next.blocking_send(Msg::Flush);
                            break;
                        }
                        Err(e) => {
                            store_err(&err_store, format!("node {} push error: {e}", node.name()));
                            let _ = tx_next.blocking_send(Msg::Flush);
                            break;
                        }
                    }
                }
                Msg::Flush => {
                    flushed = true;
                    let _ = node.push(None);
                    let _ = drain_to_next(&mut node, &tx_next, &err_store);
                    let _ = tx_next.blocking_send(Msg::Flush);
                    break;
                }
            }
        }

        // Upstream closed without explicit flush: treat as flush once.
        if !flushed {
            let _ = node.push(None);
            let _ = drain_to_next(&mut node, &tx_next, &err_store);
            let _ = tx_next.blocking_send(Msg::Flush);
        }
    });
}

fn spawn_last_stage(
    mut node: Box<dyn DynNode>,
    mut w: Box<dyn AudioWriter + Send>,
    mut rx: mpsc::Receiver<Msg>,
    err_store: Arc<Mutex<Option<String>>>,
) -> tokio::task::JoinHandle<RunnerResult<()>> {
    tokio::task::spawn_blocking(move || {
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                Msg::Data(buf) => {
                    match node.push(Some(buf)) {
                        Ok(()) | Err(CodecError::Again) => {}
                        Err(CodecError::Eof) => {
                            break;
                        }
                        Err(e) => {
                            store_err(&err_store, format!("node {} push error: {e}", node.name()));
                            break;
                        }
                    }

                    loop {
                        match node.pull() {
                            Ok(NodeBuffer::Pcm(f)) => {
                                w.write_frame(&f as &dyn AudioFrameView).map_err(RunnerError::from)?;
                            }
                            Ok(NodeBuffer::Packet(_)) => {
                                store_err(&err_store, "AsyncPipelineAudioSink requires last node output PCM".to_string());
                                break;
                            }
                            Err(CodecError::Again) => break,
                            Err(CodecError::Eof) => {
                                break;
                            }
                            Err(e) => {
                                store_err(&err_store, format!("node {} pull error: {e}", node.name()));
                                break;
                            }
                        }
                    }
                }
                Msg::Flush => {
                    break;
                }
            }
        }

        // Flush last node and drain outputs to writer
        let _ = node.push(None);
        loop {
            match node.pull() {
                Ok(NodeBuffer::Pcm(f)) => {
                    w.write_frame(&f as &dyn AudioFrameView).map_err(RunnerError::from)?;
                }
                Ok(NodeBuffer::Packet(_)) => {
                    store_err(&err_store, "AsyncPipelineAudioSink requires last node output PCM".to_string());
                    break;
                }
                Err(CodecError::Again) => continue,
                Err(CodecError::Eof) => break,
                Err(e) => {
                    store_err(&err_store, format!("node {} flush pull error: {e}", node.name()));
                    break;
                }
            }
        }
        w.finalize().map_err(RunnerError::from)?;
        Ok(())
    })
}

/// 动态版：`processors (PCM->PCM)* -> writer`，每个 processor 作为一个 stage 并行跑。
///
/// - 输入：`AudioFrame`
/// - 背压：所有 stage channel 都是 bounded；当末端写入慢时，上游 `push().await` 会被阻塞（理想的 backpressure）。
pub struct AsyncPipelineAudioSink {
    in_tx: mpsc::Sender<Msg>,
    join: Option<tokio::task::JoinHandle<RunnerResult<()>>>,
    err_store: Arc<Mutex<Option<String>>>,
    finalized: bool,
    input_kind: NodeBufferKind,
}

impl AsyncPipelineAudioSink {
    /// 构造一个 node pipeline sink。
    ///
    /// - `nodes`: 运行时节点链（允许 PCM/Packet 在内部切换）
    /// - `writer`: 最终写入端（只写 PCM）
    /// - `queue_capacity`: 每个 stage 之间的有界队列容量（>=1）
    pub fn new(
        mut nodes: Vec<Box<dyn DynNode>>,
        writer: Box<dyn AudioWriter + Send>,
        queue_capacity: usize,
    ) -> RunnerResult<Self> {
        let cap = queue_capacity.max(1);
        let err_store = Arc::new(Mutex::new(None));

        if nodes.is_empty() {
            return Err(RunnerError::InvalidData("AsyncPipelineAudioSink requires at least 1 node"));
        }

        // Validate adjacency kind match
        for i in 0..(nodes.len() - 1) {
            if nodes[i].output_kind() != nodes[i + 1].input_kind() {
                return Err(RunnerError::InvalidData("pipeline node kind mismatch"));
            }
        }
        // Require last output PCM (writer consumes PCM)
        if nodes.last().unwrap().output_kind() != NodeBufferKind::Pcm {
            return Err(RunnerError::InvalidData("AsyncPipelineAudioSink requires last node output PCM"));
        }

        let input_kind = nodes[0].input_kind();

        // Build stage channels
        let (in_tx, mut prev_rx) = mpsc::channel::<Msg>(cap);

        // Spawn node stages 0..n-2
        while nodes.len() > 1 {
            let node = nodes.remove(0);
            let (tx_next, rx_next) = mpsc::channel::<Msg>(cap);
            let rx = prev_rx;
            spawn_node_stage(node, rx, tx_next, err_store.clone());
            prev_rx = rx_next;
        }

        // Last stage: last node + writer
        let last = nodes.remove(0);
        let join = spawn_last_stage(last, writer, prev_rx, err_store.clone());

        Ok(Self {
            in_tx,
            join: Some(join),
            err_store,
            finalized: false,
            input_kind,
        })
    }

    pub fn with_default_capacity(nodes: Vec<Box<dyn DynNode>>, writer: Box<dyn AudioWriter + Send>) -> RunnerResult<Self> {
        Self::new(nodes, writer, 8)
    }
}

#[async_trait]
impl AsyncAudioSink for AsyncPipelineAudioSink {
    type In = NodeBuffer;

    fn name(&self) -> &'static str {
        "async-pipeline-audio-sink"
    }

    async fn push(&mut self, input: Self::In) -> RunnerResult<()> {
        if self.finalized {
            return Err(RunnerError::Codec(CodecError::InvalidState(
                "async pipeline audio sink already finalized",
            )));
        }
        if input.kind() != self.input_kind {
            return Err(RunnerError::InvalidData("async pipeline audio sink input kind mismatch"));
        }
        self.in_tx
            .send(Msg::Data(input))
            .await
            .map_err(|_| err_or_closed(&self.err_store))?;
        Ok(())
    }

    async fn finalize(&mut self) -> RunnerResult<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        self.in_tx
            .send(Msg::Flush)
            .await
            .map_err(|_| err_or_closed(&self.err_store))?;

        let Some(join) = self.join.take() else {
            return Ok(());
        };
        match join.await {
            Ok(r) => r,
            Err(e) => Err(RunnerError::Codec(CodecError::Other(format!(
                "async pipeline audio sink join failed: {e}"
            )))),
        }
    }
}

impl Drop for AsyncPipelineAudioSink {
    fn drop(&mut self) {
        if let Some(j) = &self.join {
            j.abort();
        }
    }
}

