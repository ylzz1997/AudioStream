use crate::runner::error::RunnerResult;
use core::future::Future;
use core::pin::Pin;

/// 异步 Runner：返回一个 Future，执行完成后表示整个链路跑完并 finalize 了 sink。
pub trait AsyncRunner {
    fn execute<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = RunnerResult<()>> + 'a>>;
}


