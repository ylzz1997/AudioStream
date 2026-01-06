# AudioStream

一个面向**音频流式处理**的 Rust 库，并通过 PyO3 暴露给 Python 使用：你可以把 **PCM（numpy）** 按 chunk 推给 `Encoder`，拿到 **编码后的 frame(bytes)**；或把 **编码帧(bytes)** 推给 `Decoder` 按 chunk 取回 **PCM（numpy）**。

目前 Python 侧最常用的场景是：

- **Python 生成/读取 PCM → 编码成 MP3/AAC 帧 → 通过网络推流**
- **Python/网络接收 MP3/AAC 帧 → 解码成 PCM → 做进一步处理/落盘**

> 说明：MP3/AAC 编码/解码通常依赖 `ffmpeg` feature（见下文安装方式）。

---

## 功能概览

- **Python API（PyO3）**：`Encoder` / `Decoder`
- **输入/输出**：
  - **Encoder**：`put_frame(pcm_numpy)` → `get_frame() -> bytes`
  - **Decoder**：`put_frame(frame_bytes)` → `get_frame() -> pcm_numpy`
- **Chunk 语义**：
  - 内部会把输入缓存到 FIFO，累计到 `chunk_samples` 才会产出一个 output frame
  - `get_frame(force=True)` 会在结束时尽可能把残留数据 flush 出来（包含对底层 codec 的 EOF flush）

---

## 依赖与构建（Python）

**注意⚠️：需要先安装FFMPEG！！！！！**

**否则 MP3/AAC encoder 会报需要 ffmpeg**

在项目根目录：

```bash
python -m pip install -U maturin numpy
maturin develop -F python -F ffmpeg
```

安装后 Python 侧可：

```python
import pyaudiostream as ast
```

---

## Python API 快速示例

### Encoder（PCM → 编码帧 bytes）

```python
import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
cfg = ast.Mp3EncoderConfig(fmt, chunk_samples=1152, bitrate=128_000)
enc = ast.Encoder("mp3", cfg)

pcm = np.zeros((2, 1152), dtype=np.float32)  # (channels, samples)
enc.put_frame(pcm)

pkt = enc.get_frame()
if pkt is not None:
    assert isinstance(pkt, (bytes, bytearray))

# 结束时建议 flush
while True:
    last = enc.get_frame(force=True)
    if last is None:
        break
```

### Decoder（编码帧 bytes → PCM numpy）

```python
import pyaudiostream as ast

cfg = ast.Mp3DecoderConfig(chunk_samples=1024, packet_time_base_den=48000)
dec = ast.Decoder("mp3", cfg)

dec.put_frame(b"...mp3 bytes...") # A Frame
pcm = dec.get_frame()
if pcm is not None:
    # numpy ndarray, shape=(channels, samples)
    print(pcm.shape, pcm.dtype)
```

---

## Example 1：`example/server.py`（Python 编码推流 → 浏览器解码流式播放）

这个示例包含三种模式，覆盖完整链路：

- **server**：提供网页 + WebSocket 广播；同时开一个 TCP ingest 端口接收“按帧”推送
- **sender**：使用 `pyaudiostream` 从 PCM（wav 或正弦）编码成 MP3/AAC 帧，按 framing 推给 ingest
- **demo**：同一进程里同时跑 server + sender（方便一条命令自测）

### 安装依赖

```bash
python -m pip install -U aiohttp
```

（并确保你已 `maturin develop -F python -F ffmpeg`）

### 一条命令自测（推荐）

```bash
python example/server.py --mode demo --codec mp3
```

浏览器打开：`http://127.0.0.1:23456`，点击“开始播放”。

### 分开跑（server / sender）

启动 server（HTTP+WS + TCP ingest）：

```bash
python example/server.py --mode server --codec mp3 --http-port 23456 --ingest-port 23457
```

启动 sender（从 wav 编码并推送）：

```bash
python example/server.py --mode sender --codec mp3 --host 127.0.0.1 --ingest-port 23457 --wav "/abs/path/to/input.wav" --seconds 0
```

#### 关键参数说明

- `--wav`
  - 目前示例要求 **16-bit PCM WAV**（sampwidth=2）
  - 且 wav 的 **采样率/声道数** 要和 `--sample-rate/--channels` 一致，否则会报错（避免“音调变高/变低”）
- `--chunk-samples`
  - PCM chunk 大小；越小延迟越低，但编码器/浏览器端可能更容易 buffer underrun
  - MP3 常见每声道 1152 samples（示例默认 1152）
  - AAC 常见为 1024

### 浏览器端说明

- **MP3**：使用 MSE（MediaSource）`audio/mpeg` 流式 append，兼容性最好
- **AAC(ADTS)**：不同浏览器对 `audio/aac` 的 MSE 支持不一致；建议优先用 MP3 验证链路