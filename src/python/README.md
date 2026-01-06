## Python 接口（PyO3）

### 构建/安装（推荐 maturin）

在项目根目录：

```bash
python -m pip install -U maturin numpy
maturin develop -F python
```

安装后：

```python
import pyaudiostream as ast
```

### Encoder 用法

```python
import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
cfg = ast.WavEncoderConfig(fmt, chunk_samples=1024)
enc = ast.Encoder("wav", cfg)

pcm = np.zeros((2, 1024), dtype=np.float32)
enc.put_frame(pcm)

out = enc.get_frame()         # bytes 或 None
out2 = enc.get_frame(force=True)  # 可强制 flush 不满 chunk 的残留
```

### Decoder 用法

```python
import numpy as np
import pyaudiostream as ast

# WAV/PCM 解码需要 output_format.planar=False（输入 packet 是 interleaved bytes）
out_fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=False)
cfg = ast.WavDecoderConfig(out_fmt, chunk_samples=1024)
dec = ast.Decoder("wav", cfg)

dec.put_frame(b"...")     # bytes
pcm = dec.get_frame()     # numpy ndarray shape=(channels, samples) 或 None
pcm_last = dec.get_frame(force=True)
```

### 语义说明

- `put_frame(...)`：只负责把输入写入内部缓冲/驱动 codec 状态机，不保证立刻有输出。
- `get_frame(force=False)`：
  - **够一个 chunk**：返回一帧（Encoder 返回 `bytes`；Decoder 返回 `numpy.ndarray`）。
  - **不够一个 chunk**：返回 `None`，并通过 `warnings.warn` 提示。
  - **force=True**：把最后不足一个 chunk 的残留也作为最后一帧返回（如果残留存在）。


