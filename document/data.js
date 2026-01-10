window.AUDIOSTREAM_DOCS = {
  python: {
    label: "Python",
    hero: {
      kicker: "默认语言：Python",
      title: "Python 使用说明",
      desc:
        "这里是 AudioStream 的 Python 接口文档/示例/FAQ。左侧是条目列表，如果想要看Rust的接口文档请从左边找",
      chips: [
        {
          href: "https://github.com/ylzz1997/AudioStream",
          text:
            '<svg class="chip__icon" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z"></path></svg><span>GitHub：ylzz1997/AudioStream</span>',
        },
        ],
    },
    groups: [
      {
        title: "快速开始",
        items: [
          {
            id: "py-quickstart",
            title: "快速开始",
            desc: "介绍如何安装本库，并快速运行一个最简单的示例",
            body: `
              <div class="hero">
                <div class="hero__kicker">快速开始</div>
                <h1 class="hero__title">Python：最小可运行示例</h1>
                <p class="hero__desc">介绍如何安装本库，并快速运行一个最简单的示例。</p>
              </div>

              <section class="section">
                <h2>安装</h2>
                <p>如果安装本仓库最好的办法是使用pip，但是想要最新版最好git clone下来然后安装。</p>
                <pre><code>
git clone https://github.com/ylzz1997/AudioStream.git
cd AudioStream
python -m pip install -U maturin numpy
maturin develop -F python -F ffmpeg
                </code></pre>

              <p>如果pip install</p>
              <pre><code>
pip install pyaudiostream
              </code></pre>

              </section>

              <section class="section" id="py-api-overview">
                <h2>快速启动</h2>
                <p>这里举一个简单的示例，读取一个音频文件，然后将它编码为MP3格式并保存到本地。</p>
                <pre><code>
import pyaudiostream as ast
import librosa

y, sr = librosa.load("test.wav", sr=None, mono=False)

fmt = ast.AudioFormat(sample_rate=sr, channels=y.shape[0], sample_type="f32", planar=True)
cfg = ast.Mp3EncoderConfig(fmt, chunk_samples=1152, bitrate=320_000)
enc = ast.Encoder("mp3", cfg)

for chunk in librosa.util.frame(y, frame_length=1152, hop_length=1152).transpose(2,0,1):
    enc.put_frame(chunk)

mp3_frames = []
while True:
    pkt = enc.get_frame()
    if pkt is not None:
        mp3_frames.append(pkt)
    else:
        mp3_frames.append(enc.get_frame(force=True))
        break

with open("test.mp3", "wb") as f:
    f.write(b''.join(mp3_frames))
                </code></pre>
              </section>
              <section class="section" id="py-api-overview">
                <h2>更进一步</h2>
                <p>举一个更进一步的示例，将一个音频编码为MP3Frame，并通过网络传输到前端播放</p>
                <p>由于代码太长，请去仓库路径example/server.py查看</p>
              </section>
            `,
          },
        ],
      },
      {
        title: "进阶",
        items: [
          {
            id: "py-audioflow",
            title: "AudioFlow模型",
            desc: "认识AudioStream最基本的抽象模型: AudioFlow",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">AudioFlow：用“流”来理解音频处理</h1>
                <p class="hero__desc">把音频看作持续到达的数据流：Source 产出数据、Node 变换数据、Sink 消费数据；Runner 负责驱动整条链路。</p>
              </div>

              <section class="section">
                <h2>两种载荷（Payload）</h2>
                <p>系统内部只传两类数据：</p>
                <ul>
                  <li><b>PCM</b>：原始音频帧（numpy），用 <code>NodeBuffer.pcm(...)</code> 表示</li>
                  <li><b>Packet</b>：编码后的包（<code>Packet</code>），用 <code>NodeBuffer.packet(...)</code> 表示</li>
                </ul>
                <p>它们的“容器”统一为 <code>NodeBuffer</code>，从而让 Pipeline 能以一致方式拼装。</p>
              </section>

              <section class="section">
                <h2>最小链路示例：PCM → 编码成 Opus Packet</h2>
                <p>这里不做文件/网络 IO，用一个 Python Source 生成静音 PCM，交给 Opus encoder node 产出 Packet。</p>
                <pre><code>
import numpy as np
import pyaudiostream as ast

# Opus encoder 目前要求 48kHz（库内部不做隐式重采样）
fmt = ast.AudioFormat(sample_rate=48000, channels=1, sample_type="f32", planar=True)

class MySource:
    def __init__(self):
        self.sent = False
    def pull(self):
        if self.sent:
            return None  # EOF
        self.sent = True
        pcm = np.zeros((1, 960), dtype=np.float32)  # (channels, samples)
        return ast.NodeBuffer.pcm(pcm, fmt)

class MySink:
    def __init__(self):
        self.pkts = []
    def push(self, buf: ast.NodeBuffer):
        pkt = buf.as_packet()
        if pkt is not None:
            self.pkts.append(pkt.data)
    def finalize(self):
        pass

nodes = [
    ast.make_encoder_node("opus", ast.OpusEncoderConfig(fmt, chunk_samples=960, bitrate=96_000)),
]

r = ast.AsyncDynRunner(MySource(), nodes, MySink())
r.run()
                </code></pre>
              </section>
            `,
          },
          {
            id: "py-encoder",
            title: "认识Encoder",
            desc: "认识AudioStream Codec的Encoder抽象，以及如何使用它们",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">Encoder：PCM → 编码帧（bytes / Packet）</h1>
                <p class="hero__desc">Encoder 接收 PCM（numpy），内部做缓存与编码；当累计到 <code>chunk_samples</code> 时产出编码数据。</p>
              </div>

              <section class="section">
                <h2>核心 API</h2>
                <ul>
                  <li><code>enc.put_frame(pcm_numpy)</code></li>
                  <li><code>enc.get_frame(force=False) -&gt; bytes | None</code></li>
                  <li><code>enc.get_packet(force=False) -&gt; Packet | None</code></li>
                </ul>
                <p><b>chunk 语义</b>：不够一个 chunk 时 <code>get_*</code> 会返回 None；结束时建议 <code>force=True</code> flush 残留。</p>
              </section>

              <section class="section">
                <h2>示例：编码成 MP3 并保存</h2>
                <pre><code>import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
enc = ast.Encoder("mp3", ast.Mp3EncoderConfig(fmt, chunk_samples=1152, bitrate=128_000))

# 生成 1 秒静音：48kHz / 1152 ≈ 41.6 个 chunk
for _ in range(42):
    enc.put_frame(np.zeros((2, 1152), dtype=np.float32))

out = []
while True:
    b = enc.get_frame()

    if b is None:
        b = enc.get_frame(force=True)
        if b is not None:
            out.append(b)
        else:
            print("no more data")
        break
    else:
        out.append(b)

with open("out.mp3", "wb") as f:
    f.write(b"".join(out))
                </code></pre>
              </section>

              <section class="section">
                <h2>注意事项</h2>
                <ul>
                  <li><b>Opus</b>：输入采样率建议固定 48kHz；如需 44.1kHz 等，先用 Processor/Pipeline 做重采样。</li>
                  <li><b>MP3/AAC</b>：通常需要带 <code>ffmpeg</code> feature 构建（见快速开始安装命令）。</li>
                </ul>
              </section>
            `,
          },
          {
            id: "py-decoder",
            title: "认识Decoder",
            desc: "认识AudioStream Codec的Decoder抽象，以及如何使用它们",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">Decoder：编码帧（bytes / Packet）→ PCM</h1>
                <p class="hero__desc">Decoder 适合“网络收到按帧分包的数据”这种场景：你持续 <code>put_frame/put_packet</code>，按 chunk 取回 PCM（numpy）。</p>
              </div>

              <section class="section">
                <h2>核心 API</h2>
                <ul>
                  <li><code>dec.put_frame(frame_bytes)</code> 或 <code>dec.put_packet(pkt)</code></li>
                  <li><code>dec.get_frame(force=False, layout="planar|interleaved") -&gt; numpy | None</code></li>
                </ul>
              </section>

              <section class="section">
                <h2>示例：解码 MP3 帧（来自网络/文件分帧）</h2>
                <p>下面演示“收到一帧 MP3 bytes 就喂给解码器”的用法（分帧方式可参考仓库 <code>example/server.py</code>）。</p>
                <pre><code>
import pyaudiostream as ast

dec = ast.Decoder("mp3", ast.Mp3DecoderConfig(chunk_samples=1152, packet_time_base_den=sr))

def on_mp3_frame(mp3_frames: list[bytes]):
    pcm_list = []
    for frame in mp3_frames:
        dec.put_frame(frame)
    while True:
        pcm = dec.get_frame() # pcm: numpy.ndarray, 默认 planar (channels, samples)
        if pcm is None:
            break
        pcm_list.append(pcm)
    return np.concatenate(pcm_list, axis=1)

pcm = on_mp3_frame(mp3_frames) # mp3_frames: list[bytes] mp3 encoder 的输出
print("pcm", pcm.shape, pcm.dtype)

import soundfile as sf
sf.write("test.wav", pcm.T, sr)

                </code></pre>
              </section>

              <section class="section">
                <h2>常见坑</h2>
                <ul>
                  <li><b>WAV/PCM 解码</b>：配置里 <code>output_format.planar=False</code>（输入 bytes 通常是 interleaved）。</li>
                  <li><b>Opus raw packet 流</b>：可能需要 <code>extradata</code>（OpusHead）来获知声道数等信息。</li>
                </ul>
              </section>
            `,
          },
          {
            id: "py-processor",
            title: "认识Processor",
            desc: "认识AudioStream Codec的Processor抽象，以及如何使用它们",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">Processor：PCM → PCM 的“算子”</h1>
                <p class="hero__desc">Processor 用于增益、重采样、压缩等 DSP 处理；语义与 Encoder/Decoder 类似：put 输入、get 输出。</p>
              </div>

              <section class="section">
                <h2>示例：44.1kHz 重采样到 48kHz（为 Opus 做准备）</h2>
                <pre><code>
import numpy as np
import pyaudiostream as ast

in_fmt = ast.AudioFormat(44100, 2, "f32", planar=True)
out_fmt = ast.AudioFormat(48000, 2, "f32", planar=True)

p = ast.Processor.resample(in_fmt, out_fmt, out_chunk_samples=960, pad_final=True)

# 假设你有一段 44.1kHz 的 PCM
p.put_frame(np.random.randn(2, 4410).astype(np.float32))
p.flush()

while True:
    out = p.get_frame()
    if out is None:
        break
    print("out", out.shape)
                </code></pre>
              </section>

              <section class="section">
                <h2>示例：增益（Gain）</h2>
                <pre><code>import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(48000, 1, "f32", planar=True)
p = ast.Processor.gain(fmt, gain=0.5)
p.put_frame(np.ones((1, 960), dtype=np.float32))
out = p.get_frame(force=True)
print(out.shape, out.max())
                </code></pre>
              </section>
            `,
          },{
            id: "py-io",
            title: "认识IO",
            desc: "认识AudioStream的IO抽象(AudioReader/AudioWriter)，以及如何使用它们",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">IO：AudioFileReader / AudioFileWriter</h1>
                <p class="hero__desc">Python 侧提供文件读写封装：<code>AudioFileReader</code> 读文件并产出 PCM；<code>AudioFileWriter</code> 接收 PCM 并写出编码文件。</p>
              </div>

              <section class="section">
                <h2>AudioFileReader：从文件读 PCM（numpy）</h2>
                <p>支持格式：wav / mp3 / aac_adts / flac / opus_ogg。</p>
                <pre><code>
import pyaudiostream as ast

r = ast.AudioFileReader("in.wav", "wav")
while True:
    pcm = r.next_frame(layout="planar")
    if pcm is None:
        break
    print(pcm.shape, pcm.dtype)
                </code></pre>
              </section>

              <section class="section">
                <h2>AudioFileWriter：把 PCM 写成文件</h2>
                <pre><code>import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
w = ast.AudioFileWriter("out.wav", "wav", input_format=fmt)

w.write_pcm(np.zeros((2, 960), dtype=np.float32))
w.finalize()
                </code></pre>
              </section>
            `,
          },
          {
            id: "py-pipeline",
            title: "认识Pipeline",
            desc: "接下来就是难点了，认识AudioStream的Pipeline抽象，以及如何使用它们！",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">Pipeline：把多个节点串成一条可运行的链</h1>
                <p class="hero__desc"><code>AsyncDynPipeline</code> 接收一组动态节点（DynNode），负责在节点之间搬运数据；你可以手动 push/flush/get，也可以交给 Runner 自动驱动。</p>
              </div>

              <section class="section">
                <h2>用节点工厂组装 Pipeline</h2>
                <pre><code>
import numpy as np
import pyaudiostream as ast

# libopus 只支持 packed/interleaved 的 flt/s16（不支持 planar 的 fltp/s16p）
in_fmt = ast.AudioFormat(44100, 2, "f32", planar=True)
out_fmt = ast.AudioFormat(48000, 2, "f32", planar=False)

nodes = [
    ast.make_processor_node("resample", ast.ResampleNodeConfig(in_fmt, out_fmt, out_chunk_samples=960, pad_final=True)),
    ast.make_processor_node("gain", ast.GainNodeConfig(out_fmt, gain=0.9)),
    ast.make_encoder_node("opus", ast.OpusEncoderConfig(out_fmt, chunk_samples=960, bitrate=96_000)),
]

p = ast.AsyncDynPipeline(nodes)

# 推入一段 44.1k PCM（示例用随机数），然后 flush
# interleaved: shape = (samples, channels)
p.push(ast.NodeBuffer.pcm(np.random.randn(2, 441000).astype(np.float32), in_fmt))
p.flush()

# 读取输出（此处输出为 packet）
while True:
    out = p.get()
    if out is None:
        break
    pkt = out.as_packet()
    if pkt is not None:
        print("packet bytes:", len(pkt.data))
                </code></pre>
              </section>

              <section class="section">
                <h2>关键点</h2>
                <ul>
                  <li><b>kind 对齐</b>：节点之间 input/output kind 必须匹配（pcm 或 packet）。</li>
                  <li><b>flush</b>：输入结束要 push None 或调用 <code>flush()</code>，才能把内部残留全部排空。</li>
                </ul>
              </section>
            `,
          },{
            id: "py-source",
            title: "认识AudioSource",
            desc: "认识AudioStream的AudioSource抽象(他与Pipeline平级)，以及如何使用它们",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">AudioSource：数据从哪里来</h1>
                <p class="hero__desc">在 Python 里，一个 Source 只需要实现 <code>pull() -&gt; Optional[NodeBuffer]</code>：返回 None 表示输入结束（EOF）。</p>
              </div>

              <section class="section">
                <h2>示例：生成正弦波 PCM Source</h2>
                <pre><code>
import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(48000, 1, "f32", planar=True)

class SineSource:
    def __init__(self, seconds=1.0, freq=440.0, chunk_samples=960):
        self.sr = 48000
        self.total = int(seconds * self.sr)
        self.freq = freq
        self.chunk = chunk_samples
        self.pos = 0

    def pull(self):
        if self.pos >= self.total:
            return None
        n = min(self.chunk, self.total - self.pos)
        t = (np.arange(n) + self.pos) / self.sr
        pcm = (0.2 * np.sin(2.0 * np.pi * self.freq * t)).astype(np.float32)[None, :]
        self.pos += n
        return ast.NodeBuffer.pcm(pcm, fmt)
                </code></pre>
              </section>
            `,
          },
          {
            id: "py-sink",
            title: "认识AudioSink",
            desc: "认识AudioStream的AudioSink抽象(他与Pipeline平级)，以及如何使用它们",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">AudioSink：数据到哪里去</h1>
                <p class="hero__desc">在 Python 里，一个 Sink 需要实现 <code>push(buf)</code> 与 <code>finalize()</code>：push 逐个消费输出，finalize 做收尾。</p>
              </div>

              <section class="section">
                <h2>示例：收集所有 Packet 到内存</h2>
                <pre><code>
import pyaudiostream as ast

class CollectPackets:
    def __init__(self):
        self.packets = []
    def push(self, buf: ast.NodeBuffer):
        pkt = buf.as_packet()
        if pkt is None:
            raise ValueError("expects packet")
        self.packets.append(pkt.data)
    def finalize(self):
        print("total packets:", len(self.packets))
                </code></pre>
              </section>
            `,
          },
          {
            id: "py-runner",
            title: "认识Runner",
            desc: "认识AudioStream的Runner抽象，以及如何使用它们",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">Runner：驱动 Source → Pipeline → Sink</h1>
                <p class="hero__desc"><code>AsyncDynRunner</code> 是最常用的执行入口：给它一个 Source、一个 nodes 列表、一个 Sink，然后 <code>run()</code> 即可跑完整条流式链路。</p>
              </div>

              <section class="section">
                <h2>示例：文件读取 → 重采样 → 写出文件</h2>
                <pre><code>
import pyaudiostream as ast

src = ast.AudioFileReader("in.wav", "wav")  # 读文件并产出 PCM

out_fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
dst = ast.AudioFileWriter("out.flac", "flac", out_fmt, compression_level=8)

nodes = [
    ast.make_processor_node(
        "resample",
        ast.ResampleNodeConfig(
            None,  # None 表示首帧推断输入格式，更适合接入 NodeBuffer/文件 reader 等“自带格式”的场景，如果需要严格要求输入格式，可以传入 AudioFormat
            out_fmt,
        ),
    ),
]

r = ast.AsyncDynRunner(src, nodes, dst)
r.run()
dst.finalize()
                </code></pre>
              </section>

              <section class="section">
                <h2>调试建议</h2>
                <ul>
                  <li><b>先跑通最短链路</b>：Source → encoder → Sink（收包）</li>
                  <li><b>再逐步加节点</b>：resample/gain/compressor，确认每步的 input/output kind 一致</li>
                </ul>
              </section>
            `,
          },
        ],
      },
      {
        title: "API 文档",
        items: [
          {
            id: "py-api-doc",
            title: "API 文档",
            desc: "AudioStream 的 Python API 文档。",
            body: `
              <div class="hero">
                <div class="hero__kicker">进阶</div>
                <h1 class="hero__title">配置与参数</h1>
                <p class="hero__desc">建议按“场景 → 推荐参数 → 可选参数 → 注意事项”维护，方便扩展。</p>
              </div>

              <section class="section">
                <h2>推荐结构</h2>
                <pre><code>{
  "preset": "voice",
  "target_lufs": -16,
  "true_peak_db": -1.0
}</code></pre>
              </section>
            `,
          },
        ],
      },
    ],
  },

  rust: {
    label: "Rust",
    hero: {
      kicker: "语言：Rust",
      title: "Rust 使用说明",
      desc: "这里放 AudioStream 的 Rust crate API、模块结构、示例代码、性能注意事项等", // chip
    },
    groups: [
      {
        title: "结构与模块（暂时没做）",
        items: [
          {
            id: "rs-overview",
            title: "模块概览（占位）",
            desc: "占位",
            body: `
              <div class="hero">
                <div class="hero__kicker">结构与模块</div>
                <h1 class="hero__title">Rust：模块概览</h1>
                <p class="hero__desc">占位</p>
              </div>
            `,
          },
        ],
      },
      {
        title: "示例",
        items: [
          {
            id: "rs-snippet",
            title: "最小示例",
            desc: "占位",
            body: `
              <div class="hero">
                <div class="hero__kicker">示例</div>
                <h1 class="hero__title">Rust：最小示例</h1>
                <p class="hero__desc">占位</p>
              </div>
            `,
          },
        ],
      },
    ],
  },
};


