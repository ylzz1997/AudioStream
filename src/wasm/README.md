## WASM 导出（浏览器端解码 + WebAudio 播放）

### 当前支持范围

- 仅支持 **WAV/PCM frame(bytes)**（也就是本项目 `Encoder("wav", ...)` 输出的 raw PCM bytes）。
- MP3/AAC 在本仓库里依赖 FFmpeg backend，默认不适合 `wasm32-unknown-unknown`。

### Rust 导出

导出类型在 `src/wasm/mod.rs`：

- `WavPcmDecoder(sample_rate, channels, sample_type, chunk_samples)`
  - `put_frame(frame: Uint8Array)`
  - `get_pcm(force: bool) -> Float32Array | null`（返回 interleaved float32）
  - `pending_samples()` / `state()`

### 构建（wasm-pack）

在项目根目录：

```bash
cargo install wasm-pack
wasm-pack build --target web -F wasm
```

生成的 `pkg/` 里包含 `.wasm` 与 JS glue。