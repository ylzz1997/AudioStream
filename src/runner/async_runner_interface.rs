use crate::runner::error::RunnerResult;
use async_trait::async_trait;

/// 异步 Runner：执行完成后表示整个链路跑完并 finalize 了 sink。
///
/// 使用 `async-trait` 把 `async fn` 降级成可编译的 trait 方法。
#[async_trait(?Send)]
pub trait AsyncRunner {
    async fn execute(&mut self) -> RunnerResult<()>;
}


