//! tokio 版本的异步（后台并行）静态 pipeline（编译期类型校验）。

use crate::codec::error::{CodecError, CodecResult};
use crate::pipeline::node::static_node_interface::StaticNode;
use crate::pipeline::node::node_interface::AsyncPipeline;
use tokio::sync::mpsc;
use async_trait::async_trait;

type Msg<T> = Result<Option<T>, CodecError>;

fn drain_typed<N, TOut>(node: &mut N, tx_next: &mpsc::UnboundedSender<Msg<TOut>>) -> Result<(), CodecError>
where
    N: StaticNode<Out = TOut>,
    TOut: Send + 'static,
{
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

fn spawn_stage<N, TIn, TOut>(
    mut node: N,
    mut rx: mpsc::UnboundedReceiver<Msg<TIn>>,
    tx_next: mpsc::UnboundedSender<Msg<TOut>>,
) where
    N: StaticNode<In = TIn, Out = TOut> + Send + 'static,
    TIn: Send + 'static,
    TOut: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                Err(e) => {
                    let _ = tx_next.send(Err(e));
                    break;
                }
                Ok(None) => {
                    let _ = node.push(None);
                    let _ = drain_typed(&mut node, &tx_next);
                    let _ = tx_next.send(Ok(None));
                    break;
                }
                Ok(Some(v)) => match node.push(Some(v)) {
                    Ok(()) | Err(CodecError::Again) => {
                        let _ = drain_typed(&mut node, &tx_next);
                    }
                    Err(CodecError::Eof) => {
                        let _ = tx_next.send(Ok(None));
                        break;
                    }
                    Err(e) => {
                        let _ = tx_next.send(Err(e));
                        break;
                    }
                },
            }
        }
        let _ = tx_next.send(Ok(None));
    });
}

fn spawn_last_stage<N, TIn, TOut>(
    mut node: N,
    mut rx: mpsc::UnboundedReceiver<Msg<TIn>>,
    out_tx: mpsc::UnboundedSender<Result<TOut, CodecError>>,
) where
    N: StaticNode<In = TIn, Out = TOut> + Send + 'static,
    TIn: Send + 'static,
    TOut: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                Err(e) => {
                    let _ = out_tx.send(Err(e));
                    return;
                }
                Ok(None) => {
                    let _ = node.push(None);
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
                                let _ = out_tx.send(Err(e));
                                return;
                            }
                        }
                    }
                }
                Ok(Some(v)) => match node.push(Some(v)) {
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

/// 固定 3 段的 Async 异步静态 pipeline（N1 -> N2 -> N3）。
pub struct AsyncPipeline3<N1, N2, N3>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    N3::Out: Send + 'static,
{
    in_tx: mpsc::UnboundedSender<Msg<N1::In>>,
    out_rx: mpsc::UnboundedReceiver<Result<N3::Out, CodecError>>,
    _phantom: core::marker::PhantomData<(N1, N2, N3)>,
}

impl<N1, N2, N3> AsyncPipeline3<N1, N2, N3>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    N3::Out: Send + 'static,
{
    pub fn new(n1: N1, n2: N2, n3: N3) -> Self {
        let (in_tx, rx1) = mpsc::unbounded_channel::<Msg<N1::In>>();
        let (tx2, rx2) = mpsc::unbounded_channel::<Msg<N1::Out>>();
        let (tx3, rx3) = mpsc::unbounded_channel::<Msg<N2::Out>>();
        let (out_tx, out_rx) = mpsc::unbounded_channel::<Result<N3::Out, CodecError>>();

        spawn_stage::<N1, N1::In, N1::Out>(n1, rx1, tx2);
        spawn_stage::<N2, N1::Out, N2::Out>(n2, rx2, tx3);
        spawn_last_stage::<N3, N2::Out, N3::Out>(n3, rx3, out_tx);

        Self {
            in_tx,
            out_rx,
            _phantom: core::marker::PhantomData,
        }
    }

}


#[async_trait]
impl<N1, N2, N3> AsyncPipeline for AsyncPipeline3<N1, N2, N3>
where
    N1: StaticNode + Send + 'static,
    N2: StaticNode<In = N1::Out> + Send + 'static,
    N3: StaticNode<In = N2::Out> + Send + 'static,
    N1::In: Send + 'static,
    N1::Out: Send + 'static,
    N2::Out: Send + 'static,
    N3::Out: Send + 'static,
{
    type In = N1::In;
    type Out = N3::Out;

    fn push_frame(&self, input: Self::In) -> CodecResult<()> {
        self.in_tx
            .send(Ok(Some(input)))
            .map_err(|_| CodecError::InvalidState("tokio pipeline input channel closed"))
    }

    fn flush(&self) -> CodecResult<()> {
        self.in_tx
            .send(Ok(None))
            .map_err(|_| CodecError::InvalidState("tokio pipeline input channel closed"))
    }

    fn try_get_frame(&mut self) -> CodecResult<Self::Out> {
        match self.out_rx.try_recv() {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(e),
            Err(mpsc::error::TryRecvError::Empty) => Err(CodecError::Again),
            Err(mpsc::error::TryRecvError::Disconnected) => Err(CodecError::Eof),
        }
    }

    async fn get_frame(&mut self) -> CodecResult<Self::Out> {
        match self.out_rx.recv().await {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e),
            None => Err(CodecError::Eof),
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::error::CodecError;
    use std::collections::VecDeque;

    struct I32Echo {
        q: VecDeque<i32>,
        flushed: bool,
        add: i32,
    }

    impl I32Echo {
        fn new(add: i32) -> Self {
            Self {
                q: VecDeque::new(),
                flushed: false,
                add,
            }
        }
    }

    impl StaticNode for I32Echo {
        type In = i32;
        type Out = i32;

        fn name(&self) -> &'static str {
            "i32-echo"
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
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn async_static_pipeline3_push_then_get() {
        let mut p = AsyncPipeline3::new(I32Echo::new(1), I32Echo::new(10), I32Echo::new(100));
        p.push_frame(1).unwrap();
        p.push_frame(2).unwrap();
        p.flush().unwrap();

        let mut outs = Vec::new();
        loop {
            match p.get_frame().await {
                Ok(v) => outs.push(v),
                Err(CodecError::Eof) => break,
                Err(e) => panic!("unexpected err: {e:?}"),
            }
        }
        // 1 +1 +10 +100 = 112; 2 -> 113
        assert_eq!(outs, vec![112, 113]);
    }
}


