use std::env;

fn main() {
    // macOS 上构建 PyO3 扩展模块时，通常需要允许未解析的 Python 符号在运行时由宿主解释器提供：
    // - `-undefined dynamic_lookup`
    //
    // 理论上 PyO3 的 `extension-module` feature 会注入该链接参数；但在某些组合（例如 cargo test + cdylib）
    // 下可能没有生效，导致链接阶段报 `_PyBaseObject_Type` 等未定义符号。
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let python_feature = env::var_os("CARGO_FEATURE_PYTHON").is_some();

    if python_feature && target_os == "macos" {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}


