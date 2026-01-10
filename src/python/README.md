# Python 接口（PyO3）

## 构建/安装（推荐 maturin）

在项目根目录：

```bash
python -m pip install -U maturin numpy
maturin develop -F python
```

或

```bash
pip install pyaudiostream
```

安装后：

```python
import pyaudiostream as ast
```

## Encoder 用法（wav/mp3/aac/opus/flac）

```python
import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
cfg = ast.WavEncoderConfig(fmt, chunk_samples=1024)
enc = ast.Encoder("wav", cfg)

pcm = np.zeros((2, 1024), dtype=np.float32)  # planar: (channels, samples)
enc.put_frame(pcm)

out = enc.get_frame()         # bytes 或 None
out2 = enc.get_frame(force=True)  # 可强制 flush 不满 chunk 的残留
```

### Opus 注意事项

- Opus encoder **目前要求输入采样率为 48kHz**（库内部暂不做隐式重采样）。如需其它采样率，先在 pipeline 里用 `ResampleProcessor` 转到 48kHz。

## Decoder 用法

```python
import numpy as np
import pyaudiostream as ast

# WAV/PCM 解码需要 output_format.planar=False（输入 packet 是 interleaved bytes）
out_fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=False)
cfg = ast.WavDecoderConfig(out_fmt, chunk_samples=1024)
dec = ast.Decoder("wav", cfg)

dec.put_frame(b"...")     # bytes
pcm = dec.get_frame()     # numpy ndarray shape=(channels, samples) 或 None（planar）
pcm_i = dec.get_frame(layout="interleaved")  # shape=(samples, channels)
pcm_last = dec.get_frame(force=True)
```

### Opus Decoder extradata

对于 **raw Opus packet 流**，解码器可能需要 `extradata`（通常是 `OpusHead`）来获知声道数等信息：

```python
cfg = ast.OpusDecoderConfig(chunk_samples=960, packet_time_base_den=48000, extradata=opus_head_bytes)
dec = ast.Decoder("opus", cfg)
```

---

## Processor（PCM->PCM）

目前暴露了：

- `Processor.identity(format=None|AudioFormat)`
- `Processor.resample(in_format, out_format, out_chunk_samples=None, pad_final=True)`
- `Processor.gain(format, gain)`
- `Processor.compressor(format, sample_rate, threshold_db, knee_width_db, ratio, expansion_ratio, expansion_threshold_db, attack_time, release_time, master_gain_db)`

```python
import numpy as np
import pyaudiostream as ast

in_fmt = ast.AudioFormat(44100, 2, "f32", planar=True)
out_fmt = ast.AudioFormat(48000, 2, "f32", planar=True)
p = ast.Processor.resample(in_fmt, out_fmt, out_chunk_samples=960, pad_final=True)

p.put_frame(np.zeros((2, 4410), np.float32))
p.flush()

while True:
    out = p.get_frame()
    if out is None:
        break
    # out: numpy (channels, samples) by default
```

增益示例：

```python
import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(48000, 2, "f32", planar=True)
p = ast.Processor.gain(fmt, gain=0.5)  # 降低一半音量
p.put_frame(np.ones((2, 960), np.float32))
p.flush()
out = p.get_frame(force=True)
```

压缩器示例：

```python
import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(48000, 2, "f32", planar=True)
p = ast.Processor.compressor(
    fmt,
    sample_rate=48000.0,
    threshold_db=-18.0,
    knee_width_db=6.0,
    ratio=4.0,
    expansion_ratio=2.0,
    expansion_threshold_db=-60.0,
    attack_time=0.01,
    release_time=0.10,
    master_gain_db=0.0,
)

# (channels, samples) 因为 fmt.planar=True
p.put_frame(np.random.randn(2, 960).astype(np.float32) * 0.1)
p.flush()
out = p.get_frame(force=True)
```

### 语义说明

- `put_frame(...)`：只负责把输入写入内部缓冲/驱动 codec 状态机，不保证立刻有输出。
- `get_frame(force=False)`：
  - **够一个 chunk**：返回一帧（Encoder 返回 `bytes`；Decoder 返回 `numpy.ndarray`）。
  - **不够一个 chunk**：返回 `None`，并通过 `warnings.warn` 提示。
  - **force=True**：把最后不足一个 chunk 的残留也作为最后一帧返回（如果残留存在）。

---

## 动态 Pipeline / Runner（AsyncDynPipeline / AsyncDynRunner）

这部分 API 面向“把若干动态节点串成一条链”，并支持：

- 纯 Rust 节点（identity / resample / gain / compressor / encoder / decoder）
- Python 侧实现 `AudioSource` / `AudioSink`（通过回调 `pull/push/finalize`）

### 用节点工厂组装 pipeline

```python
import numpy as np
import pyaudiostream as ast

# 44.1k -> 48k（为 Opus encoder 做准备）
# 注意：libopus encoder 仅支持 packed/interleaved 的 flt/s16（不支持 fltp/s16p），
# 所以这里要用 planar=False，并且 numpy shape 约定为 (samples, channels)。
in_fmt = ast.AudioFormat(sample_rate=44100, channels=2, sample_type="f32", planar=False)
out_fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=False)

nodes = [
    ast.make_processor_node("resample", ast.ResampleNodeConfig(in_fmt, out_fmt, out_chunk_samples=960, pad_final=True)),
    ast.make_processor_node(
        "compressor",
        ast.CompressorNodeConfig(
            out_fmt,
            sample_rate=48000.0,
            threshold_db=-18.0,
            knee_width_db=6.0,
            ratio=4.0,
            expansion_ratio=2.0,
            expansion_threshold_db=-60.0,
            attack_time=0.01,
            release_time=0.10,
            master_gain_db=0.0,
        ),
    ),
    ast.make_processor_node("gain", ast.GainNodeConfig(out_fmt, gain=0.9)),
    ast.make_encoder_node("opus", ast.OpusEncoderConfig(out_fmt, chunk_samples=960, bitrate=96_000)),
]
```

### Python 自定义 Node（DynNode）

除了内置的 `make_identity_node / make_processor_node / make_encoder_node / make_decoder_node`，你也可以用 Python 自己实现一个节点，并放进 `AsyncDynPipeline(nodes=[...])` / `AsyncDynRunner(nodes=[...])` 里。

#### 1) Python 侧需要实现的方法

- `push(buf: Optional[ast.NodeBuffer]) -> None`
  - `buf is None` 表示 flush（输入结束）
- `pull() -> Optional[ast.NodeBuffer]`
  - flush 前：返回 `None` 表示“暂无输出”（类似 Again）
  - flush 后：返回 `None` 表示“结束”（类似 EOF）
  - 你也可以显式 `raise BlockingIOError` / `raise EOFError` 来表达 Again/EOF

#### 2) 构造 DynNode

用 `ast.make_python_node(obj, input_kind, output_kind, name="...")` 把你的对象包装成可用于 pipeline 的 `DynNode`：

```python
import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(sample_rate=48000, channels=1, sample_type="f32", planar=True)

class GainNode:
    def __init__(self, gain: float):
        self.gain = gain
        self.q = []
        self.flushed = False

    def push(self, buf):
        if buf is None:
            self.flushed = True
            return

        pcm = buf.as_pcm()
        if pcm is None:
            raise ValueError("expects pcm")

        # 注意：NodeBuffer 在 Rust 侧会被 move；建议输出新的 NodeBuffer
        fmt2, pts, (tb_num, tb_den) = buf.pcm_info()
        out = (pcm * self.gain).astype(np.float32, copy=False)
        self.q.append(ast.NodeBuffer.pcm(out, fmt2, pts=pts, time_base_num=tb_num, time_base_den=tb_den))

    def pull(self):
        if self.q:
            return self.q.pop(0)
        if self.flushed:
            raise EOFError()
        return None

py_gain = ast.make_python_node(GainNode(0.5), input_kind="pcm", output_kind="pcm", name="gain")

nodes = [
    py_gain,
    ast.make_encoder_node("opus", ast.OpusEncoderConfig(fmt, chunk_samples=960, bitrate=96_000)),
]

p = ast.AsyncDynPipeline(nodes)
p.push(ast.NodeBuffer.pcm(np.zeros((1, 960), np.float32), fmt))
p.flush()
while True:
    out = p.get()
    if out is None:
        break
    # out: packet
```

### 用 AsyncDynRunner 串联 Python Source/Sink

Python 侧只要实现：

- `source.pull() -> Optional[ast.NodeBuffer]`
- `sink.push(buf: ast.NodeBuffer) -> None`
- `sink.finalize() -> None`

示例：把 Python 生成的 PCM 通过 Opus 编码，收集所有 packet：

```python
import numpy as np
import pyaudiostream as ast

in_fmt = ast.AudioFormat(sample_rate=48000, channels=1, sample_type="f32", planar=True)

class MySource:
    def __init__(self):
        self.sent = False
    def pull(self):
        if self.sent:
            return None
        self.sent = True
        pcm = np.zeros((1, 960), dtype=np.float32)
        return ast.NodeBuffer.pcm(pcm, in_fmt)

class MySink:
    def __init__(self):
        self.packets = []
    def push(self, buf: ast.NodeBuffer):
        pkt = buf.as_packet()
        if pkt is not None:
            self.packets.append(pkt.data)
    def finalize(self):
        pass

nodes = [
    ast.make_encoder_node("opus", ast.OpusEncoderConfig(in_fmt, chunk_samples=960, bitrate=96_000)),
]

sink = MySink()
r = ast.AsyncDynRunner(MySource(), nodes, sink)
r.run()
print(len(sink.packets))
```

### File Reader/Writer 作为 Source/Sink

```python
import pyaudiostream as ast

src = ast.AudioFileReader("in.wav", "wav")
out_fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
dst = ast.AudioFileWriter("out.flac", "flac", out_fmt, compression_level=8)

nodes = [
    ast.make_processor_node(
        "resample",
        ast.ResampleNodeConfig(
            ast.AudioFormat(44100, 2, "f32", planar=True),  # 仅示例：真实输入格式可从首帧推导/约定
            out_fmt,
        ),
    ),
]

r = ast.AsyncDynRunner(src, nodes, dst)
r.run()
dst.finalize()
```
