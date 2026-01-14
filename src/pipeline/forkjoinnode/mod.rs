//! Fork/Join 节点：把同一份输入 fork 到多条 pipeline，并在末端 join + reduce。
//!
//! 目标：
//! - 支持动态/静态、同步/异步四种组合
//! - reduce 函数可由用户注入；预置 `Sum`（加权求和）

pub mod reduce;
pub mod fork_join_node_interface;

pub mod dynamic_fork_join_node;
pub mod async_dynamic_fork_join_node;
pub mod static_fork_join_node;
pub mod async_static_fork_join_node;

pub use reduce::{Concat, Max, Mean, Min, Product, Reduce, Sum, Xor};

pub use dynamic_fork_join_node::ForkJoinNode;
pub use async_dynamic_fork_join_node::AsyncForkJoinNode;
pub use static_fork_join_node::ForkJoinStaticNode;
pub use async_static_fork_join_node::AsyncForkJoinStaticNode;

