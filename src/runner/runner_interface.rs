use crate::runner::error::RunnerResult;

/// 同步 Runner：一次 `execute()` 跑完整条链路（source -> pipeline -> sink）。
pub trait Runner {
    fn execute(&mut self) -> RunnerResult<()>;
}


