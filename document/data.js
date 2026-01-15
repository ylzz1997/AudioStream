window.AUDIOSTREAM_DOCS = {
  python: {
    label: "Python",
    hero: {
      kicker: "默认语言：Python",
      title: "Python 使用说明",
      desc:
        `这里是 AudioStream 的 Python 接口文档/示例/FAQ。左侧是条目列表，如果想要看Rust的接口文档请从左边找 <br/> <br/>
        
        
        AudioStream一个面向<b>音频流式处理</b>的 Rust 库，并通过 PyO3 暴露给 Python 使用：你可以把 <b>PCM（numpy）</b> 按 chunk 推给 Encoder，拿到 <b>编码后的 frame(bytes)</b>；或把 <b>编码帧(bytes)</b> 推给 Decoder 按 chunk 取回 <b>PCM（numpy）</b>。
        
        `,
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
                <pre><code class="language-bash">
git clone https://github.com/ylzz1997/AudioStream.git
cd AudioStream
python -m pip install -U maturin numpy
maturin develop -F python -F ffmpeg
                </code></pre>

              <p>如果pip install</p>
              <pre><code class="language-bash">
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

              <section class="section">
                <h2>Encoder 的 input_format 自动推断</h2>
                <p>
                  <b>Rust 层</b>：所有 Encoder 的 <code>input_format</code> 支持 <code>None</code>（首帧推断并锁定）。
                  这在 “上游提供的 PCM 本身携带格式”（例如 Rust 侧 <code>AudioFrame</code>、或 Pipeline 里 <code>NodeBuffer.pcm(...)</code> 自带 <code>AudioFormat</code>）的场景很有用。
                </p>
                <ul>
                  <li><b>推断触发</b>：第一次收到 <b>非空帧</b>（<code>nb_samples &gt; 0</code>）时锁定格式；空帧不会触发推断。</li>
                  <li><b>一致性</b>：锁定后如果后续帧格式改变，会报 <code>input AudioFormat mismatch</code>（当前库不做隐式重采样/布局转换）。</li>
                  <li><b>reset</b>：对“推断模式”的 Encoder，<code>reset()</code> 会清空已推断的格式并回到未初始化状态；下次再用首帧重新推断。</li>
                  <li><b>flush(None)</b>：如果从未收到过非空帧，flush 只会结束流，不会初始化 encoder。</li>
                </ul>
                <p>
                  <b>Python 的 ast.Encoder</b>：目前构造时仍要求显式 <code>AudioFormat</code>（因为需要将 numpy 的 dtype/shape 与格式严格对齐，再拷贝成内部 PCM 帧）。
                  如果你希望“从首帧 numpy 自动推断 sample_rate/channels/sample_type/planar”，需要额外约定元数据来源（目前未提供）。
                </p>
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

              <section class="section">
                <h2>示例：延迟（Delay）</h2>
                <pre><code>import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(48000, 1, "f32", planar=True)
p = ast.Processor.delay(fmt, delay_ms=200.0)
p.put_frame(np.ones((1, 960), dtype=np.float32))
p.flush()

out = []
while True:
    f = p.get_frame()
    if f is None:
        break
    out.append(f)
print("frames:", len(out))</code></pre>
              </section>

              <section class="section">
                <h2>Processor 的格式自动推断</h2>
                <p>
                  多数 Processor 都支持把输入格式参数设为 <code>None</code>，表示：<b>首帧推断输入 AudioFormat</b>。
                  这是为了方便把 Processor 接到 <code>AudioFileReader</code>/<code>NodeBuffer</code> 这种“帧本身携带格式”的上游。
                </p>
                <ul>
                  <li><b>推断触发</b>：第一次收到 <b>非空帧</b>（<code>nb_samples &gt; 0</code>）时锁定输入格式；空帧不会触发推断。</li>
                  <li><b>一致性</b>：锁定后如果后续帧格式改变，会报格式不匹配（不会隐式做 resample/convert）。</li>
                  <li><b>reset</b>：对“推断模式”的 Processor，<code>reset()</code> 会清空已推断的格式并回到未初始化状态；下次再用首帧重新推断。</li>
                  <li><b>resample</b>：<code>out_format</code> 必须显式给定；<code>in_format=None</code> 只影响输入端推断，不影响输出端。</li>
                </ul>
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
                <p class="hero__desc">
                  Python 侧提供文件读写封装：<code>AudioFileReader</code> 读文件并产出 PCM；<code>AudioFileWriter</code> 接收 PCM 并写出编码文件。
                  同时，它们背后对应 Rust 的统一 IO 抽象：<code>AudioReader</code> / <code>AudioWriter</code>。
                </p>
              </div>

              <section class="section">
                <h2>抽象：AudioReader / AudioWriter</h2>
                <p>AudioStream 把“从哪里读 PCM / 往哪里写 PCM”抽象成两个 trait：</p>
                <ul>
                  <li><b>AudioReader</b>：<code>next_frame() -&gt; Option[AudioFrame]</code>（EOF 返回 None）</li>
                  <li><b>AudioWriter</b>：<code>write_frame(frame)</code> + <code>finalize()</code></li>
                </ul>
                <p>
                  这样做的好处是：Runner 只关心“拉取一帧 → 推进 pipeline → 写出输出”，不关心底层是文件、网络还是内存。
                  在 Rust 里，任何实现了 <code>AudioReader</code> 的类型会自动成为 <code>AudioSource</code>；实现了 <code>AudioWriter</code> 的类型会自动成为 <code>AudioSink</code>（因此可以直接被 Runner 使用）。
                </p>
                <p>
                  在 Python 里，<code>AudioFileReader</code> / <code>AudioFileWriter</code> 也额外实现了 Runner 需要的 <code>pull</code> / <code>push</code> / <code>finalize</code> 方法，
                  因而可以直接作为 <code>AsyncDynRunner</code> 的 Source/Sink（见 “Runner” 小节示例）。
                </p>
              </section>

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
                <p class="hero__desc">
                  Pipeline 用来把多个 Node 串成一条链，并在节点之间搬运数据。
                  Rust 侧同时提供同步/异步两套 Pipeline；Python 侧目前主要暴露异步动态版本 <code>AsyncDynPipeline</code>（最通用、也最适合与 Runner 搭配）。
                </p>
              </div>

              <section class="section">
                <h2>同步 Pipeline vs 异步 Pipeline</h2>
                <ul>
                  <li>
                    <b>同步（Sync）</b>：前台单线程推进，调用 <code>push_and_drain / drain_all</code> 时才会把数据从上游推到下游；
                    Node 通过 <code>Again</code> 表达“暂无输出/背压”，通过 <code>Eof</code> 表达“flush 后结束”。
                  </li>
                  <li>
                    <b>异步（Async）</b>：<code>push</code> 后后台并行流转，各 stage 通过 channel 串联；末端用 <code>get/try_get</code> 取输出。
                    这种模式更适合“每帧处理耗时明显、希望流水线并行”的场景（Runner 小节有示意图）。
                  </li>
                </ul>
                <p>
                  Python 暴露的 <code>AsyncDynPipeline</code> 走 tokio 后台任务；当你把 Python 自定义 Node/Sink/Source 接入时，
                  内部会使用 <b>current-thread runtime</b>，保证 Python 对象不会跨线程访问（避免 GIL/unsendable 问题）。
                </p>
              </section>

              <section class="section">
                <h2>动态 Pipeline vs 静态 Pipeline（Rust 侧概念）</h2>
                <ul>
                  <li><b>动态（Dyn）</b>：节点输入/输出统一用 <code>NodeBuffer(pcm|packet)</code> 做类型擦除，适合“运行时拼装节点列表”。Python 使用的就是这一套。</li>
                  <li><b>静态（Static）</b>：节点类型在编译期确定，连接关系由泛型保证（更强的类型安全/更少运行时检查），通常用于纯 Rust 场景。</li>
                </ul>
              </section>

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

              <section class="section">
                <h2>Python 自定义 Node（DynNode）：与 Rust 原生节点的区别与注意事项</h2>
                <p>除内置的 Rust 节点（resample/gain/compressor/encoder/decoder 等）外，你也可以用 Python 实现一个节点，然后用 <code>make_python_node</code> 包装成 DynNode 放进 pipeline。</p>
                <ul>
                  <li><b>性能差异</b>：Rust 原生节点在 Rust 内部执行，开销更低；Python 节点需要跨语言回调 + GIL，适合轻量逻辑或做 glue，不建议在其中做很重的逐样本 DSP。</li>
                  <li><b>NodeBuffer move 语义</b>：<code>NodeBuffer</code> 在 Rust 侧会被 move；同一个 buffer 不能重复使用/重复读取。需要缓存输出时，请创建新的 <code>NodeBuffer.pcm(...)</code> / <code>NodeBuffer.packet(...)</code>。</li>
                  <li><b>控制流语义</b>：<code>pull()</code> 在 flush 前返回 None 表示 “Again/暂无输出”；flush 后返回 None 表示 “Eof/结束”。也可以显式抛 <code>BlockingIOError</code> / <code>EOFError</code> 表达 Again/Eof。</li>
                  <li><b>kind 必须匹配</b>：你在 <code>make_python_node(obj, input_kind, output_kind)</code> 声明的 kind，必须与实际 push/pull 的 buffer kind 一致。</li>
                </ul>
                <pre><code>
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
        fmt2, pts, (tb_num, tb_den) = buf.pcm_info()
        out = (pcm * self.gain).astype(np.float32, copy=False)
        self.q.append(ast.NodeBuffer.pcm(out, fmt2, pts=pts, time_base_num=tb_num, time_base_den=tb_den))

    def pull(self):
        if self.q:
            return self.q.pop(0)
        if self.flushed:
            raise EOFError()
        return None  # 暂无输出（Again）

py_gain = ast.make_python_node(GainNode(0.5), input_kind="pcm", output_kind="pcm", name="gain")
                </code></pre>
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
                <p class="hero__desc">
                  Source 表示“只出不进”的数据源。在 Python 里，一个 Source 只需要实现 <code>pull() -&gt; Optional[NodeBuffer]</code>：返回 None 表示输入结束（EOF）。
                  它通常与 <code>AsyncDynRunner</code> 搭配使用，由 Runner 持续拉取并推进 pipeline。
                </p>
              </div>

              <section class="section">
                <h2>注意事项</h2>
                <ul>
                  <li><b>返回值约束</b>：必须返回 <code>NodeBuffer</code> 或 None；返回其它类型会报错。</li>
                  <li><b>一次性消费（move）</b>：Runner 会把你返回的 <code>NodeBuffer</code> move 进 Rust；同一个 buffer 不要重复返回/复用。</li>
                  <li><b>kind 对齐</b>：Source 的输出 kind 必须与 pipeline 第一个节点的 input kind 对齐（pcm 或 packet）。</li>
                </ul>
              </section>

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
                <p class="hero__desc">
                  Sink 表示“只进不出”的数据汇。在 Python 里，一个 Sink 需要实现 <code>push(buf)</code> 与 <code>finalize()</code>：push 逐个消费输出，finalize 做收尾。
                </p>
              </div>

              <section class="section">
                <h2>注意事项</h2>
                <ul>
                  <li><b>一次性消费（move）</b>：传入 <code>push</code> 的 <code>NodeBuffer</code> 在 Rust 侧会被 move；建议在 push 内立刻提取所需数据（如 <code>buf.as_packet()</code> / <code>buf.as_pcm()</code>）。</li>
                  <li><b>kind 对齐</b>：Sink 期待的 kind 必须与 pipeline 最后一个节点的 output kind 对齐（pcm 或 packet）。</li>
                  <li><b>finalize</b>：Runner 结束时一定会调用；用于关闭文件、flush 缓冲、提交尾部索引等。</li>
                </ul>
              </section>

              <section class="section">
                <h2>示例：收集所有 Packet 到内存</h2>
                <pre><code class="language-python">
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
                <p class="hero__desc">
                  Runner 是最常用的执行入口：它会不断从 Source 拉取数据，推进 pipeline，并把末端输出写入 Sink，最后调用 Sink.finalize()。
                  Python 侧主要暴露 <code>AsyncDynRunner</code>（内部异步流水线并行），对外提供阻塞式 <code>run()</code>。
                </p>
              </div>

              <section class="section">
                <h2>同步 Runner(Pipeline) vs 异步 Runner(Pipeline)</h2>
                <p>Rust 侧同时存在：</p>
                <ul>
                  <li><b>同步 Runner(Pipeline)</b>：前台串行推进（每处理完一帧再处理下一帧），实现简单、调试直观。</li>
                  <li><b>异步 Runner(Pipeline)</b>：把每个 stage 视为流水线并行处理，不同帧可以在不同 stage 上重叠执行，整体吞吐更高。</li>
                </ul>
                <div class="figure">
                  <img class="docimg" src="../logo/i1.jpg" alt="Runner vs AsyncRunner：串行与流水线并行示意图" />
                  <div class="figcap">图：串行 Runner（上）需要等一帧完整走完再处理下一帧；AsyncRunner（下）可以让不同帧在 Read/Process/Encode/Send 等阶段上并行重叠，提高吞吐。</div>
                </div>
              </section>

              <section class="section">
                <h2>示例：文件读取 → 重采样 → 写出文件</h2>
                <pre><code class="language-python">
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
            title: "API 文档概览",
            desc: "入口与核心概念；详细参考请看左侧分类。",
            body: `
              <div class="hero">
                <div class="hero__kicker">语言：Python · 参考</div>
                <h1 class="hero__title">Python API 文档：概览</h1>
                <p class="hero__desc">
                  先建立整体心智模型，再按分类查看详细 API：格式 / 编码 / 解码 / 处理 / Pipeline·IO。
                </p>
              </div>

              <section class="section">
                <h2>导入与命名空间</h2>
                <pre><code class="language-python">import pyaudiostream as ast</code></pre>
              </section>

              <section class="section">
                <h2>核心概念</h2>
                <ul>
                  <li>
                    <b>PCM 表示</b>：使用 <code>numpy.ndarray</code>（2D）。
                    <code>AudioFormat(planar=True)</code> 时 shape 为 <code>(channels, samples)</code>；<code>planar=False</code> 时为 <code>(samples, channels)</code>。
                  </li>
                  <li>
                    <b>dtype</b>：需与 <code>AudioFormat.sample_type</code> 对齐（<code>"u8"|"i16"|"i32"|"i64"|"f32"|"f64"</code>）。
                  </li>
                  <li>
                    <b>两条路线</b>：
                    <ul>
                      <li><b>面向对象</b>：<code>Encoder</code>/<code>Decoder</code>/<code>Processor</code>（你自己写循环）。</li>
                      <li><b>Pipeline/Runner</b>：<code>DynNode</code> + <code>AsyncDynPipeline</code>/<code>AsyncDynRunner</code>（拼装链路，一次跑完）。</li>
                    </ul>
                  </li>
                  <li>
                    <b>chunk_samples</b>：一次取/出多少“每声道样本数”。不足一整块时，默认不返回残帧；可用 <code>force=True</code> 强制吐尾巴（不同组件略有差异）。
                  </li>
                  <li>
                    <b>重要：move 语义</b>：<code>NodeBuffer</code> / <code>DynNode</code> 被 Pipeline/Runner 消费后不可复用，否则会报 “已被移动（不可再次使用）”。
                  </li>
                </ul>
              </section>

              <section class="section">
                <h2>分类索引</h2>
                <ul>
                  <li><b>格式</b>：<a href="#python/py-api-format">AudioFormat</a></li>
                  <li><b>编码</b>：<a href="#python/py-api-encode">Encoder / *EncoderConfig / make_encoder_node</a></li>
                  <li><b>解码</b>：<a href="#python/py-api-decode">Decoder / *DecoderConfig / make_decoder_node</a></li>
                  <li><b>处理</b>：<a href="#python/py-api-process">Processor / *NodeConfig / make_processor_node</a></li>
                  <li><b>Pipeline / IO</b>：<a href="#python/py-api-pipeline-io">Packet / NodeBuffer / DynNode / AsyncDynPipeline / AsyncDynRunner / AudioFileReader / AudioFileWriter / ParallelAudioWriter / make_identity_node / make_tap_node / make_python_node</a></li>
                </ul>
              </section>

              <section class="section">
                <h2>最小可运行示例（PCM → 编码 bytes）</h2>
                <pre><code class="language-python">import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
enc = ast.Encoder("aac", ast.AacEncoderConfig(fmt, chunk_samples=1024, bitrate=128_000))

pcm = np.zeros((2, 1024), dtype=np.float32)
enc.put_frame(pcm)

out = enc.get_frame()
if out is not None:
    assert isinstance(out, (bytes, bytearray))</code></pre>
              </section>
            `,
          },

          {
            id: "py-api-format",
            title: "格式：AudioFormat",
            desc: "PCM 格式描述与 numpy 约定。",
            body: `
              <div class="hero">
                <div class="hero__kicker">API · 格式</div>
                <h1 class="hero__title">AudioFormat</h1>
                <p class="hero__desc">描述 PCM 的采样率、通道数、sample_type、planar 与（可选）通道布局。</p>
              </div>

              <section class="section">
                <h2>构造</h2>
                <pre><code class="language-python">ast.AudioFormat(sample_rate, channels, sample_type, planar=True, channel_layout_mask=0)</code></pre>
              </section>

              <section class="section">
                <h2>参数</h2>
                <ul>
                  <li><b>sample_rate</b>（int）: 采样率（Hz），必须 &gt; 0。</li>
                  <li><b>channels</b>（int）: 通道数，必须 &gt; 0。</li>
                  <li><b>sample_type</b>（str）: <code>"u8"|"i16"|"i32"|"i64"|"f32"|"f64"</code>。</li>
                  <li><b>planar</b>（bool）: True 时 shape=(channels, samples)；False 时 shape=(samples, channels)。</li>
                  <li><b>channel_layout_mask</b>（int）: 可选通道布局 bitmask。为 0 时按 channels 推断 mono/stereo/unspecified。</li>
                </ul>
              </section>

              <section class="section">
                <h2>属性（只读）</h2>
                <ul>
                  <li><b>sample_rate</b>, <b>channels</b>, <b>sample_type</b>, <b>planar</b>, <b>channel_layout_mask</b></li>
                </ul>
              </section>
            `,
          },

          {
            id: "py-api-encode",
            title: "编码：Encoder",
            desc: "Encoder，*EncoderConfig，make_encoder_node。",
            body: `
              <div class="hero">
                <div class="hero__kicker">API · 编码</div>
                <h1 class="hero__title">Encoder</h1>
                <p class="hero__desc">输入 PCM（numpy），输出编码帧（bytes）或 Packet；也支持创建 pipeline 节点。</p>
              </div>

              <section class="section">
                <h2>Encoder</h2>
                <pre><code class="language-python">ast.Encoder(codec: str, config: Any)</code></pre>
                <ul>
                  <li><b>codec</b>：<code>"wav"|"mp3"|"aac"|"opus"|"flac"</code>（也接受 <code>"pcm"</code> 作为 wav 的别名）。</li>
                  <li><b>config</b>：对应的 *EncoderConfig 实例。</li>
                </ul>
                <h3>方法</h3>
                <ul>
                  <li><b>put_frame(pcm, pts=None, format=None)</b></li>
                  <li><b>get_frame(force=False) -&gt; Optional[bytes]</b></li>
                  <li><b>get_packet(force=False) -&gt; Optional[Packet]</b></li>
                  <li><b>reset()</b></li>
                  <li><b>pending_samples() -&gt; int</b></li>
                  <li><b>state() -&gt; str</b></li>
                </ul>
                <h3>参数说明</h3>
                <ul>
                  <li><b>put_frame.pcm</b>：2D numpy。planar=True 时 shape=(channels, samples)；planar=False 时 shape=(samples, channels)。dtype 需与 format.sample_type 对齐。</li>
                  <li><b>put_frame.pts</b>：可选时间戳（int）。单位由 time_base 决定；Encoder 侧一般按“samples/每声道”推进。</li>
                  <li><b>put_frame.format</b>：可选 <code>AudioFormat</code>。当你用 <code>*EncoderConfig(input_format=None)</code> 构造时，第一次 <code>put_frame</code> 必须传入，用于首帧推断并锁定格式。</li>
                  <li><b>get_frame.force</b>：是否强制 flush 尾巴。False：不足一个 chunk 默认不返回；True：会尝试把最后不足 chunk 的残留也编码输出（若 codec 支持）。</li>
                  <li><b>get_packet.force</b>：同上（但返回 <code>Packet</code>）。</li>
                  <li><b>pending_samples()</b>：当前 FIFO 内累计的样本数（每声道）。</li>
                  <li><b>state()</b>：返回 <code>"ready"</code>/<code>"need_more"</code>/<code>"empty"</code>，用于快速判断是否可取输出。</li>
                </ul>
                <h3>属性（只读）</h3>
                <ul>
                  <li><b>codec</b></li>
                  <li><b>chunk_samples</b></li>
                </ul>
                <h3>Notes</h3>
                <ul>
                  <li><b>尾帧</b>：默认不足一个 chunk 不返回；<code>force=True</code> 会触发 flush/吐尾巴。</li>
                </ul>
              </section>

              <section class="section">
                <h2>make_encoder_node</h2>
                <pre><code class="language-python">ast.make_encoder_node(codec: str, config: Any) -&gt; DynNode</code></pre>
                <ul>
                  <li><b>codec</b>：同 <code>ast.Encoder</code> 的 codec。</li>
                  <li><b>config</b>：对应的 *EncoderConfig。注意：此处 config 内 <code>chunk_samples</code> 会被忽略（由上游分帧决定）。</li>
                </ul>
                <p><b>注意</b>：config 内 <code>chunk_samples</code> 在这里会被忽略（由上游分帧决定）。</p>
                <p>相关类型请见：<a href="#python/py-api-pipeline-io">Pipeline / IO</a></p>
              </section>

              <section class="section">
                <h2>*EncoderConfig</h2>
                <h3>WavEncoderConfig</h3>
                <pre><code class="language-python">ast.WavEncoderConfig(input_format: Optional[AudioFormat] = None, chunk_samples: int = 1024)</code></pre>
                <ul>
                  <li><b>input_format</b>：输入 PCM 格式。None 表示首帧推断（此时第一次 put_frame 必须提供 format）。</li>
                  <li><b>chunk_samples</b>：每次编码的 samples/每声道；仅用于面向对象 Encoder（make_encoder_node 会忽略）。</li>
                </ul>
                <h3>Mp3EncoderConfig</h3>
                <pre><code class="language-python">ast.Mp3EncoderConfig(input_format: Optional[AudioFormat] = None, chunk_samples: int = 1152, bitrate: Optional[int] = 128_000)</code></pre>
                <ul>
                  <li><b>input_format</b>：同上（None=首帧推断）。</li>
                  <li><b>chunk_samples</b>：MP3 常见 1152。</li>
                  <li><b>bitrate</b>：目标码率（bps）。</li>
                </ul>
                <h3>AacEncoderConfig</h3>
                <pre><code class="language-python">ast.AacEncoderConfig(input_format: Optional[AudioFormat] = None, chunk_samples: int = 1024, bitrate: Optional[int] = None)</code></pre>
                <ul>
                  <li><b>input_format</b>：同上（None=首帧推断）。</li>
                  <li><b>chunk_samples</b>：AAC-LC 常见 1024。</li>
                  <li><b>bitrate</b>：目标码率（bps）。None 表示使用默认策略。</li>
                </ul>
                <h3>OpusEncoderConfig</h3>
                <pre><code class="language-python">ast.OpusEncoderConfig(input_format: Optional[AudioFormat] = None, chunk_samples: int = 960, bitrate: Optional[int] = 96_000)</code></pre>
                <ul>
                  <li><b>input_format</b>：同上（None=首帧推断）。</li>
                  <li><b>chunk_samples</b>：常用 20ms@48k =&gt; 960。</li>
                  <li><b>bitrate</b>：目标码率（bps）。</li>
                  <li><b>约束</b>：Opus encoder 目前要求 48kHz 输入；如需 44.1kHz 等，请先重采样。</li>
                </ul>
                <h3>FlacEncoderConfig</h3>
                <pre><code class="language-python">ast.FlacEncoderConfig(input_format: Optional[AudioFormat] = None, chunk_samples: int = 4096, compression_level: Optional[int] = None)</code></pre>
                <ul>
                  <li><b>input_format</b>：同上（None=首帧推断）。</li>
                  <li><b>chunk_samples</b>：用于内部重分帧（FLAC 本身支持可变帧长）。</li>
                  <li><b>compression_level</b>：压缩等级（常见 0..=12；越大越慢/更小）。</li>
                </ul>
              </section>
            `,
          },

          {
            id: "py-api-decode",
            title: "解码：Decoder",
            desc: "Decoder，*DecoderConfig，make_decoder_node。",
            body: `
              <div class="hero">
                <div class="hero__kicker">API · 解码</div>
                <h1 class="hero__title">Decoder</h1>
                <p class="hero__desc">输入 bytes/Packet，输出 PCM numpy；也支持创建 pipeline 节点。</p>
              </div>

              <section class="section">
                <h2>Decoder</h2>
                <pre><code class="language-python">ast.Decoder(codec: str, config: Any)</code></pre>
                <h3>方法</h3>
                <ul>
                  <li><b>put_frame(frame: bytes)</b></li>
                  <li><b>put_packet(pkt: Packet)</b></li>
                  <li><b>get_frame(force=False, layout="planar") -&gt; Optional[numpy.ndarray]</b></li>
                  <li><b>get_frame_info(force=False, layout="planar") -&gt; Optional[(numpy, pts, (tb_num, tb_den))]</b></li>
                  <li><b>reset()</b></li>
                  <li><b>pending_samples()</b></li>
                  <li><b>state()</b></li>
                  <li><b>output_format()</b></li>
                </ul>
                <h3>参数说明</h3>
                <ul>
                  <li><b>Decoder(codec, config).codec</b>：<code>"wav"|"mp3"|"aac"|"opus"|"flac"</code>。</li>
                  <li><b>Decoder(codec, config).config</b>：对应的 *DecoderConfig。</li>
                  <li><b>put_frame.frame</b>：一段编码数据（bytes）。适合你自己做“按帧切分”的场景。</li>
                  <li><b>put_packet.pkt</b>：<code>Packet</code>（带 time_base/pts/duration 等元数据）。</li>
                  <li><b>get_frame.force</b>：是否强制 flush 尾巴。通常用于输入结束时把残留全部吐出。</li>
                  <li><b>get_frame.layout</b>：输出 numpy 布局：<code>"planar"</code>=&gt;(channels,samples)，<code>"interleaved"</code>=&gt;(samples,channels)。</li>
                  <li><b>get_frame_info(...)</b>：比 get_frame 多返回 pts 与 time_base，用于你自己做时间戳同步。</li>
                </ul>
                <h3>Notes</h3>
                <ul>
                  <li><b>layout</b>：支持 <code>planar</code>/<code>interleaved</code> 输出。</li>
                </ul>
              </section>

              <section class="section">
                <h2>make_decoder_node</h2>
                <pre><code class="language-python">ast.make_decoder_node(codec: str, config: Any) -&gt; DynNode</code></pre>
                <ul>
                  <li><b>codec</b>：同 <code>ast.Decoder</code> 的 codec。</li>
                  <li><b>config</b>：对应的 *DecoderConfig。注意：此处 config 内 <code>chunk_samples</code> 会被忽略（由下游取帧决定）。</li>
                </ul>
                <p><b>注意</b>：config 内 <code>chunk_samples</code> 在这里会被忽略（由下游取帧决定）。</p>
                <p>相关类型请见：<a href="#python/py-api-pipeline-io">Pipeline / IO</a></p>
              </section>
              
              <section class="section">
                <h2>*DecoderConfig</h2>
                <h3>WavDecoderConfig</h3>
                <pre><code class="language-python">ast.WavDecoderConfig(output_format: AudioFormat, chunk_samples: int)</code></pre>
                <p><b>约束</b>：WAV/PCM 的 <code>output_format.planar</code> 必须为 False。</p>
                <ul>
                  <li><b>output_format</b>：解码输出 PCM 的格式（采样率/通道数/类型/布局）。</li>
                  <li><b>chunk_samples</b>：每次 get_frame 返回的 samples/每声道。</li>
                </ul>
                <h3>Mp3DecoderConfig</h3>
                <pre><code class="language-python">ast.Mp3DecoderConfig(chunk_samples: int, packet_time_base_den: int = 48000)</code></pre>
                <ul>
                  <li><b>chunk_samples</b>：每次 get_frame 返回的 samples/每声道。</li>
                  <li><b>packet_time_base_den</b>：输入 packet 的 time_base 分母（time_base = 1/den）。如果你的 packet 以 samples 为单位，通常用采样率。</li>
                </ul>
                <h3>AacDecoderConfig</h3>
                <pre><code class="language-python">ast.AacDecoderConfig(chunk_samples: int, packet_time_base_den: int = 48000)</code></pre>
                <ul>
                  <li><b>chunk_samples</b>：每次 get_frame 返回的 samples/每声道。</li>
                  <li><b>packet_time_base_den</b>：同上。</li>
                </ul>
                <h3>OpusDecoderConfig</h3>
                <pre><code class="language-python">ast.OpusDecoderConfig(chunk_samples: int, packet_time_base_den: int = 48000, extradata: Optional[bytes] = None)</code></pre>
                <ul>
                  <li><b>chunk_samples</b>：每次 get_frame 返回的 samples/每声道。</li>
                  <li><b>packet_time_base_den</b>：同上。Opus 常用 48000（samples）。</li>
                  <li><b>extradata</b>：可选 OpusHead 等头信息；某些输入流需要它来获知声道数等。</li>
                </ul>
                <h3>FlacDecoderConfig</h3>
                <pre><code class="language-python">ast.FlacDecoderConfig(chunk_samples: int, packet_time_base_den: int = 48000)</code></pre>
                <ul>
                  <li><b>chunk_samples</b>：每次 get_frame 返回的 samples/每声道。</li>
                  <li><b>packet_time_base_den</b>：同上。</li>
                </ul>
              </section>

            `,
          },

          {
            id: "py-api-process",
            title: "处理：Processor",
            desc: "Processor，*NodeConfig，make_processor_node。",
            body: `
              <div class="hero">
                <div class="hero__kicker">API · 处理</div>
                <h1 class="hero__title">Processor</h1>
                <p class="hero__desc">PCM -&gt; PCM：identity / resample / gain / delay / compressor；也支持创建 pipeline 节点。</p>
              </div>

              <section class="section">
                <h2>Processor（通用）</h2>
                <p>Processor 用于 <b>PCM -&gt; PCM</b> 处理。你可以用下面 5 个构造函数创建不同处理器。</p>
                <h3>通用方法</h3>
                <ul>
                  <li><b>put_frame(pcm, pts=None, format=None)</b>：输入一帧 PCM。若输入格式未知，第一次必须提供 <code>format</code>。</li>
                  <li><b>flush()</b></li>
                  <li><b>get_frame(force=False, layout="planar")</b></li>
                  <li><b>reset()</b></li>
                  <li><b>output_format()</b></li>
                </ul>
                <h3>通用方法参数说明</h3>
                <ul>
                  <li><b>put_frame.pcm</b>：2D numpy。布局/ dtype 需与 format（或已锁定的输入格式）一致。</li>
                  <li><b>put_frame.pts</b>：可选时间戳（int）。</li>
                  <li><b>put_frame.format</b>：可选 <code>AudioFormat</code>。当 Processor 以 format=None 推断模式创建时，第一次 put_frame 必须传入。</li>
                  <li><b>get_frame.force</b>：是否强制 flush 尾巴（不同 processor 行为可能略有差异）。</li>
                  <li><b>get_frame.layout</b>：输出 numpy 布局：<code>planar</code>/<code>interleaved</code>。</li>
                </ul>
              </section>

              <section class="section">
                <h2>make_processor_node</h2>
                <pre><code class="language-python">ast.make_processor_node(kind: str, config: Any) -&gt; DynNode</code></pre>
                <p><b>kind</b>：<code>identity</code>/<code>resample</code>/<code>gain</code>/<code>delay</code>/<code>compressor</code></p>
                <ul>
                  <li><b>kind</b>：节点类型字符串（见上）。</li>
                  <li><b>config</b>：对应的 *NodeConfig。</li>
                </ul>
                <p>相关类型请见：<a href="#python/py-api-pipeline-io">Pipeline / IO</a></p>
              </section>

                            <section class="section">
                <h2>*NodeConfig（用于 make_processor_node）</h2>
                <h3>IdentityNodeConfig</h3>
                <pre><code class="language-python">ast.IdentityNodeConfig(kind: str)  # "pcm" | "packet"</code></pre>
                <ul>
                  <li><b>kind</b>：节点类型：<code>"pcm"</code> 或 <code>"packet"</code>。</li>
                </ul>
                <h3>ResampleNodeConfig</h3>
                <pre><code class="language-python">ast.ResampleNodeConfig(in_format: Optional[AudioFormat], out_format: AudioFormat, out_chunk_samples: Optional[int]=None, pad_final: bool=True)</code></pre>
                <ul>
                  <li><b>in_format</b>：输入格式；None=首帧推断。</li>
                  <li><b>out_format</b>：输出格式（必须给定）。</li>
                  <li><b>out_chunk_samples</b>：可选输出重分帧长度（samples/每声道）。</li>
                  <li><b>pad_final</b>：flush 时是否补齐最后一块。</li>
                </ul>
                <h3>GainNodeConfig</h3>
                <pre><code class="language-python">ast.GainNodeConfig(format: Optional[AudioFormat]=None, gain: float=1.0)</code></pre>
                <ul>
                  <li><b>format</b>：输入格式；None=首帧推断。</li>
                  <li><b>gain</b>：线性增益倍率。</li>
                </ul>
                <h3>DelayNodeConfig</h3>
                <pre><code class="language-python">ast.DelayNodeConfig(format: Optional[AudioFormat]=None, delay_ms: float=0.0)</code></pre>
                <ul>
                  <li><b>format</b>：输入格式；None=首帧推断。</li>
                  <li><b>delay_ms</b>：延迟毫秒数（在开头插入静音）。</li>
                </ul>
                <h3>CompressorNodeConfig</h3>
                <pre><code class="language-python">ast.CompressorNodeConfig(
    format: Optional[AudioFormat],
    sample_rate: Optional[float]=None,
    threshold_db: float=-18.0,
    knee_width_db: float=6.0,
    ratio: float=4.0,
    expansion_ratio: float=2.0,
    expansion_threshold_db: float=-60.0,
    attack_time: float=0.01,
    release_time: float=0.10,
    master_gain_db: float=0.0,
)</code></pre>
                <ul>
                  <li><b>format</b>：输入格式；None=首帧推断。</li>
                  <li><b>sample_rate</b>：采样率（Hz）。None 时通常从 format.sample_rate 推断。</li>
                  <li><b>threshold_db</b>：压缩阈值（dB）。</li>
                  <li><b>knee_width_db</b>：knee 宽度（dB）。</li>
                  <li><b>ratio</b>：压缩比。</li>
                  <li><b>expansion_ratio</b>：扩展比。</li>
                  <li><b>expansion_threshold_db</b>：扩展阈值（dB）。</li>
                  <li><b>attack_time</b>：attack时间（秒）。</li>
                  <li><b>release_time</b>：release时间（秒）。</li>
                  <li><b>master_gain_db</b>：总输出增益（dB）。</li>
                </ul>
              </section>

              <section class="section">
                <h2>构造：identity</h2>
                <pre><code class="language-python">ast.Processor.identity(format: Optional[AudioFormat] = None)</code></pre>
                <h3>参数</h3>
                <ul>
                  <li><b>format</b>：可选输入格式。None 表示首帧推断并锁定。</li>
                </ul>
                <h3>行为</h3>
                <ul>
                  <li><b>作用</b>：不做任何处理（透传）。常用于占位/调试/做 “pcm pipeline 的 identity 节点”。</li>
                </ul>
              </section>

              <section class="section">
                <h2>构造：resample</h2>
                <pre><code class="language-python">ast.Processor.resample(in_format: Optional[AudioFormat], out_format: AudioFormat, out_chunk_samples: Optional[int] = None, pad_final: bool = True)</code></pre>
                <h3>参数</h3>
                <ul>
                  <li><b>in_format</b>：输入格式；None 表示首帧推断并锁定。</li>
                  <li><b>out_format</b>：输出格式（必须给定）。</li>
                  <li><b>out_chunk_samples</b>：可选“输出重分帧”长度（samples/每声道）。例如 Opus 常用 960@48k（20ms）。</li>
                  <li><b>pad_final</b>：flush 时如果不足 out_chunk_samples，是否补 0 到整块（True 推荐用于严格按帧长编码的下游）。</li>
                </ul>
                <h3>行为</h3>
                <ul>
                  <li><b>作用</b>：重采样/通道与布局转换（以 out_format 为准）。</li>
                  <li><b>注意</b>：本库不会在其它 Processor 中自动 resample；如果上下游格式不一致，需要显式插入 resample。</li>
                </ul>
              </section>

              <section class="section">
                <h2>构造：gain</h2>
                <pre><code class="language-python">ast.Processor.gain(format: Optional[AudioFormat] = None, gain: float = 1.0)</code></pre>
                <h3>参数</h3>
                <ul>
                  <li><b>format</b>：输入格式；None 表示首帧推断并锁定。</li>
                  <li><b>gain</b>：线性增益倍率（1.0=不变）。</li>
                </ul>
                <h3>行为</h3>
                <ul>
                  <li><b>作用</b>：对 PCM 做乘法增益。</li>
                </ul>
              </section>

              <section class="section">
                <h2>构造：delay</h2>
                <pre><code class="language-python">ast.Processor.delay(format: Optional[AudioFormat] = None, delay_ms: float = 0.0)</code></pre>
                <h3>参数</h3>
                <ul>
                  <li><b>format</b>：输入格式；None 表示首帧推断并锁定。</li>
                  <li><b>delay_ms</b>：延迟毫秒数（在开头插入静音）。</li>
                </ul>
                <h3>行为</h3>
                <ul>
                  <li><b>作用</b>：在音频开头插入静音，实现整体延迟。</li>
                </ul>
              </section>

              <section class="section">
                <h2>构造：compressor</h2>
                <pre><code class="language-python">ast.Processor.compressor(format: Optional[AudioFormat], sample_rate: float, threshold_db: float, knee_width_db: float, ratio: float, expansion_ratio: float, expansion_threshold_db: float, attack_time: float, release_time: float, master_gain_db: float)</code></pre>
                <h3>参数</h3>
                <ul>
                  <li><b>format</b>：输入格式（部分版本可传 None 表示首帧推断）。</li>
                  <li><b>sample_rate</b>：采样率（Hz），用于计算 attack/release 等时间常数（通常应与 format.sample_rate 一致）。</li>
                  <li><b>threshold_db</b>：阈值（dB）。</li>
                  <li><b>knee_width_db</b>：knee 宽度（dB）。</li>
                  <li><b>ratio</b>：压缩比（&gt;=1）。</li>
                  <li><b>expansion_ratio</b>：扩展比（噪声门/向下扩展；常见 &gt;=1）。</li>
                  <li><b>expansion_threshold_db</b>：扩展阈值（dB）。</li>
                  <li><b>attack_time</b>：attack时间（秒）。</li>
                  <li><b>release_time</b>：release 时间（秒）。</li>
                  <li><b>master_gain_db</b>：总输出增益（dB）。</li>
                </ul>
                <h3>行为</h3>
                <ul>
                  <li><b>作用</b>：动态范围压缩/扩展。</li>
                </ul>
              </section>
            `,
          },

          {
            id: "py-api-pipeline-io",
            title: "Pipeline / IO",
            desc: "Packet/NodeBuffer/DynNode、Pipeline/Runner、文件读写与自定义节点。",
            body: `
              <div class="hero">
                <div class="hero__kicker">API · Pipeline / IO</div>
                <h1 class="hero__title">Pipeline / IO</h1>
                <p class="hero__desc">用于拼装节点链路、在 Python 中自定义节点，以及文件读写。</p>
              </div>

              <section class="section">
                <h2>Packet</h2>
                <pre><code class="language-python">ast.Packet(data, time_base_num=1, time_base_den=48000, pts=None, dts=None, duration=None, flags=0)</code></pre>
                <p>用于 Decoder 输入、Pipeline 交互、或你自己维护时间戳。</p>
                <h3>参数说明</h3>
                <ul>
                  <li><b>data</b>：packet payload（bytes）。</li>
                  <li><b>time_base_num/time_base_den</b>：时间基（Rational）。实际 time_base = num/den。</li>
                  <li><b>pts</b>：可选显示时间戳（int）。单位由 time_base 决定。</li>
                  <li><b>dts</b>：可选解码时间戳（int）。</li>
                  <li><b>duration</b>：可选持续时间（int）。单位由 time_base 决定；音频里常用 samples。</li>
                  <li><b>flags</b>：内部 flags bitmask（u32 透传）。</li>
                </ul>
              </section>

              <section class="section">
                <h2>NodeBuffer</h2>
                <pre><code class="language-python">ast.NodeBuffer.pcm(pcm, format: AudioFormat, pts=None, time_base_num=None, time_base_den=None)
ast.NodeBuffer.packet(pkt: Packet)</code></pre>
                <h3>参数说明</h3>
                <ul>
                  <li><b>NodeBuffer.pcm.pcm</b>：2D numpy。planar=True 时 (channels,samples)，planar=False 时 (samples,channels)。</li>
                  <li><b>NodeBuffer.pcm.format</b>：该 PCM 的 AudioFormat。</li>
                  <li><b>NodeBuffer.pcm.pts</b>：可选时间戳（int）。</li>
                  <li><b>NodeBuffer.pcm.time_base_num/time_base_den</b>：可选时间基。None 时使用默认 time_base（通常是 1/sample_rate）。</li>
                  <li><b>NodeBuffer.packet.pkt</b>：一个 Packet。</li>
                  <li><b>kind</b>：<code>pcm</code>/<code>packet</code></li>
                  <li><b>as_pcm/as_pcm_with_layout/as_packet/pcm_info</b></li>
                  <li><b>重要</b>：被 Pipeline/Runner 消费后不可复用（move 语义）。</li>
                </ul>
                <h3>方法参数说明</h3>
                <ul>
                  <li><b>as_pcm()</b>：无参数。返回 numpy（默认 planar）或 None（若不是 pcm）。</li>
                  <li><b>as_pcm_with_layout(layout)</b>：<b>layout</b> 为 <code>"planar"</code> 或 <code>"interleaved"</code>。</li>
                  <li><b>as_packet()</b>：无参数。返回 Packet 或 None（若不是 packet）。</li>
                  <li><b>pcm_info()</b>：无参数。返回 (AudioFormat, pts, (time_base_num,time_base_den)) 或 None。</li>
                </ul>
              </section>

              <section class="section">
                <h2>DynNode</h2>
                <p>动态节点：通常由 <code>make_*_node</code> 创建。</p>
                <ul>
                  <li><b>name</b> / <b>input_kind</b> / <b>output_kind</b>（只读）</li>
                  <li><b>重要</b>：构建 Pipeline/Runner 时会被 move，同一个 DynNode 不能复用。</li>
                </ul>
              </section>

              <section class="section">
                <h2>make_identity_node</h2>
                <pre><code class="language-python">ast.make_identity_node(kind: str) -&gt; DynNode  # kind: "pcm"|"packet"</code></pre>
                <ul>
                  <li><b>kind</b>：<code>"pcm"</code> 或 <code>"packet"</code>。</li>
                </ul>
              </section>

              <section class="section">
                <h2>make_tap_node（Tap / Tee：旁路复制 + 透传）</h2>
                <p>
                  TapNode 的作用是在 pipeline 中“分叉一份数据”给某个 <b>AudioSink</b>（旁路），同时把原始 frame <b>原样透传</b>给下游节点。
                  典型用途：在中间节点插一个“监控/统计/写盘/抓包”旁路，而不改变主链路行为。
                </p>

                <pre><code class="language-python">ast.make_tap_node(sink, kind: str = "pcm") -&gt; DynNode</code></pre>

                <h3>参数介绍</h3>
                <ul>
                  <li>
                    <b>sink</b>：一个 Python 对象（通常继承 <code>ast.AudioSink</code> 仅用于类型提示即可），需要实现：
                    <ul>
                      <li><b>push(buf: ast.NodeBuffer) -&gt; None</b>：接收旁路数据。</li>
                      <li><b>finalize() -&gt; None</b>：在 stream 结束（flush）时调用一次。</li>
                    </ul>
                  </li>
                  <li>
                    <b>kind</b>：tap 的输入/输出 kind（必须与相邻节点匹配），仅支持：
                    <ul>
                      <li><code>"pcm"</code>：旁路与主链路传递 PCM。</li>
                      <li><code>"packet"</code>：旁路与主链路传递 Packet。</li>
                    </ul>
                  </li>
                </ul>

                <h3>返回值</h3>
                <ul>
                  <li><b>DynNode</b>：可直接插入 <code>AsyncDynPipeline(nodes=[...])</code>/<code>AsyncDynRunner(...)</code> 的 nodes 列表。</li>
                </ul>

                <h3>行为说明</h3>
                <ul>
                  <li><b>透传</b>：每个输入的 <code>NodeBuffer</code> 会被原样传给下游（pull 输出与输入一致）。</li>
                  <li><b>旁路复制</b>：同时会对该 <code>NodeBuffer</code> 做一次 <code>clone</code> 并调用 <code>sink.push(clone)</code>。</li>
                  <li><b>flush/finalize</b>：当 pipeline/runner flush（输入结束）时，TapNode 会调用一次 <code>sink.finalize()</code>。</li>
                </ul>

                <h3>重要注意事项</h3>
                <ul>
                  <li><b>性能/阻塞</b>：TapNode 调用的是同步 <code>sink.push()</code>。如果你的 sink 很慢（例如 Python 写文件/网络），会拖慢主链路。建议把旁路 sink 做成“内部有队列/后台线程”的实现。</li>
                  <li><b>move 语义</b>：返回的 DynNode 在构建 Pipeline/Runner 时会被 move（同一个 DynNode 不能复用）。</li>
                  <li><b>重复 finalize</b>：如果你在别处也手动调用了同一个 sink 的 <code>finalize()</code>，可能出现重复 finalize；建议让 TapNode 统一负责 finalize。</li>
                </ul>

                <h3>示例：旁路收集 + 主链路继续处理</h3>
                <pre><code class="language-python">import pyaudiostream as ast
import numpy as np

fmt = ast.AudioFormat(sample_rate=48000, channels=1, sample_type="f32", planar=True)

class Src:
    def __init__(self, n=5):
        self.i = 0
        self.n = n
    def pull(self):
        if self.i &gt;= self.n:
            return None
        self.i += 1
        pcm = np.zeros((1, 960), dtype=np.float32)
        return ast.NodeBuffer.pcm(pcm, fmt)

class TapSink(ast.AudioSink):
    def __init__(self):
        self.count = 0
    def push(self, buf: ast.NodeBuffer):
        # buf 是旁路复制出来的 NodeBuffer
        if buf.as_pcm() is not None:
            self.count += 1
    def finalize(self):
        print("tap saw", self.count, "frames")

class Dst(ast.AudioSink):
    def push(self, buf: ast.NodeBuffer):
        # 主链路末端 sink
        pass
    def finalize(self):
        pass

nodes = [
    ast.make_tap_node(TapSink(), kind="pcm"),
    ast.make_processor_node("gain", ast.GainNodeConfig(fmt, gain=1.1)),
]

r = ast.AsyncDynRunner(Src(), nodes, Dst())
r.run()</code></pre>
              </section>

              <section class="section">
                <h2>make_python_node</h2>
                <pre><code class="language-python">ast.make_python_node(obj, input_kind: str, output_kind: str, name: str = "py-node") -&gt; DynNode</code></pre>
                <h3>参数说明</h3>
                <ul>
                  <li><b>obj</b>：你的 Python 节点对象。需要实现 <code>push</code>/<code>pull</code>（可选 <code>flush</code>）。</li>
                  <li><b>input_kind</b>：<code>"pcm"</code> 或 <code>"packet"</code>。</li>
                  <li><b>output_kind</b>：<code>"pcm"</code> 或 <code>"packet"</code>。</li>
                  <li><b>name</b>：节点名称（调试/日志用）。</li>
                </ul>
                <h3>Python 侧约定</h3>
                <ul>
                  <li><b>push(nb: NodeBuffer)</b></li>
                  <li><b>pull() -&gt; Optional[NodeBuffer]</b></li>
                  <li><b>flush()</b></li>
                  <li><b>reset()</b></li>
                </ul>
                <h3>控制流异常</h3>
                <ul>
                  <li><b>BlockingIOError</b>：表示 “Again / 暂无输出”。</li>
                  <li><b>EOFError</b>：表示 “Eof / 结束”。</li>
                </ul>
              </section>

              <section class="section">
                <h2>make_async_fork_join_node（Fork/Join + Python Reduce）</h2>
                <p>
                  这是一个<b>并行分支</b>节点：把同一份输入 <b>fork</b> 到多条分支 pipeline（每条分支是一组 DynNode），
                  然后在末端把各分支的输出 <b>join</b> 并调用你提供的 <b>Python reduce</b> 回调合并为一个输出。
                  返回值是一个 <code>DynNode</code>，可直接插入 <code>AsyncDynPipeline(nodes=[...])</code>/<code>AsyncDynRunner(...)</code>。
                </p>
                <pre><code class="language-python">ast.make_async_fork_join_node(
    pipelines: list[list[DynNode]],
    reduce: Callable[[list[NodeBuffer]], NodeBuffer],
    name: str = "async-fork-join",
) -&gt; DynNode</code></pre>

                <h3>参数介绍</h3>
                <ul>
                  <li>
                    <b>pipelines</b>：分支列表。每个分支是一条 <code>DynNode</code> 链（<code>list[DynNode]</code>），至少包含 1 个节点。
                    <ul>
                      <li><b>分支内连接校验</b>：会校验每条分支内部相邻节点的 <code>output_kind</code> 与下一个节点的 <code>input_kind</code> 一致。</li>
                      <li><b>分支间一致性校验</b>：会校验所有分支的入口 <code>input_kind</code> 一致、出口 <code>output_kind</code> 一致。</li>
                      <li><b>move 语义</b>：这里传入的 DynNode 会被 move 到 Rust 内部；<b>同一个 DynNode 不能在多个地方复用</b>。</li>
                    </ul>
                  </li>
                  <li>
                    <b>reduce</b>：合并函数（callable）。签名为 <code>reduce(items: list[NodeBuffer]) -&gt; NodeBuffer</code>。
                    <ul>
                      <li><b>items</b>：长度=分支数；每次从每条分支各取 1 个输出，组成 items 后调用一次 reduce。</li>
                      <li><b>返回值</b>：必须返回 NodeBuffer（不能是 None），且 kind 必须与各分支 <code>output_kind</code> 一致。</li>
                      <li><b>来源</b>：可以是你自定义的 Python 函数/闭包；也可以是下面列出的 Rust 内置 reduce（它们同样是 callable）。</li>
                    </ul>
                  </li>
                  <li><b>name</b>：节点名称（调试/日志用）。默认 <code>"async-fork-join"</code>。</li>
                </ul>
                <h3>重要约束</h3>
                <ul>
                  <li><b>分支一致性</b>：所有分支的 <code>input_kind</code> 必须一致，<code>output_kind</code> 也必须一致。</li>
                  <li><b>对齐 join</b>：每次输出会从每条分支各取 1 个输出，凑成 <code>items</code> 后调用一次 reduce，产出 1 个输出。</li>
                  <li><b>reduce 返回值</b>：必须返回 <code>NodeBuffer</code>（不能是 None），并且 kind 必须与分支 <code>output_kind</code> 一致。</li>
                  <li><b>运行环境</b>：该节点需要在 tokio runtime 内运行（也就是把它用于 <code>AsyncDynPipeline/AsyncDynRunner</code>）。</li>
                  <li><b>move 语义</b>：<code>pipelines</code> 里的 DynNode 会被 move；同一个 DynNode 不能复用到别处。</li>
                  <li><b>内置 reduce 的限制</b>：见下方 “内置 reduce（Rust 预制）”。</li>
                </ul>

                <h3>reduce 回调签名</h3>
                <ul>
                  <li><b>reduce(items)</b>：<code>items: list[NodeBuffer]</code>，长度=分支数；返回一个新的 <code>NodeBuffer</code>。</li>
                  <li><b>建议</b>：不要“原地复用”输入的 NodeBuffer；建议从 <code>items[i]</code> 中提取数据后构造新的 <code>NodeBuffer.pcm(...)</code>/<code>NodeBuffer.packet(...)</code> 作为输出。</li>
                  <li><b>也可以直接用 Rust 内置 reduce</b>：例如 <code>ast.ReduceSum(weight=[...])</code> / <code>ast.ReduceMean()</code> / <code>ast.ReduceConcat()</code> 等（它们本质上也是可调用对象）。</li>
                </ul>

                <h3>内置 reduce（Rust 预制）</h3>
                <p>以下 reduce 都是 Python 可直接调用的对象（callable），可以直接传给 <code>make_async_fork_join_node(..., reduce=...)</code>。</p>
                <ul>
                  <li>
                    <b>ast.ReduceSum(weight: Optional[list[float]]=None)</b>
                    <ul>
                      <li><b>作用</b>：对 PCM 做<b>按样本加权求和</b>：<code>out = sum(items[i] * weight[i])</code>。</li>
                      <li><b>weight</b>：None 表示全 1；否则长度必须等于分支数。</li>
                      <li><b>支持</b>：<code>NodeBuffer.pcm</code>（支持 <code>"u8"|"i16"|"i32"|"i64"|"f32"|"f64"</code>，planar/interleaved 均可）。</li>
                      <li><b>约束</b>：要求各分支 PCM 的 <code>AudioFormat</code> / <code>nb_samples</code> / <code>time_base</code> 一致。</li>
                    </ul>
                  </li>
                  <li>
                    <b>ast.ReduceProduct(weight: Optional[list[float]]=None)</b>
                    <ul>
                      <li><b>作用</b>：对 PCM 做<b>按样本加权求积</b>：<code>out = Π (items[i] * weight[i])</code>。</li>
                      <li><b>weight</b>：None 表示全 1；否则长度必须等于分支数。</li>
                      <li><b>支持</b>：<code>NodeBuffer.pcm</code>（支持 <code>"u8"|"i16"|"i32"|"i64"|"f32"|"f64"</code>，planar/interleaved 均可）。</li>
                      <li><b>约束</b>：要求各分支 PCM 的 <code>AudioFormat</code> / <code>nb_samples</code> / <code>time_base</code> 一致。</li>
                    </ul>
                  </li>
                  <li>
                    <b>ast.ReduceMean()</b>
                    <ul>
                      <li><b>作用</b>：对 PCM 做<b>按样本平均</b>：<code>out = (items[0]+...+items[N-1]) / N</code>。</li>
                      <li><b>支持</b>：<code>NodeBuffer.pcm</code>（支持 <code>"u8"|"i16"|"i32"|"i64"|"f32"|"f64"</code>，planar/interleaved 均可）。</li>
                      <li><b>约束</b>：要求各分支 PCM 的 <code>AudioFormat</code> / <code>nb_samples</code> / <code>time_base</code> 一致。</li>
                    </ul>
                  </li>
                  <li>
                    <b>ast.ReduceMax()</b> / <b>ast.ReduceMin()</b>
                    <ul>
                      <li><b>作用</b>：对 PCM 做<b>按样本取最大/最小</b>（跨分支）。</li>
                      <li><b>支持</b>：<code>NodeBuffer.pcm</code>（支持 <code>"u8"|"i16"|"i32"|"i64"|"f32"|"f64"</code>，planar/interleaved 均可）。</li>
                      <li><b>约束</b>：要求各分支 PCM 的 <code>AudioFormat</code> / <code>nb_samples</code> / <code>time_base</code> 一致。</li>
                    </ul>
                  </li>
                  <li>
                    <b>ast.ReduceConcat()</b>
                    <ul>
                      <li><b>作用</b>：拼接（concat）。支持两种载荷：</li>
                      <li><b>Packet concat</b>：把各分支的 <code>packet.data</code> 顺序拼到一起。</li>
                      <li><b>PCM concat</b>：把各分支的 PCM 按“时间轴/样本”顺序拼接成更长的一帧（等价于 numpy 沿 samples 维度拼接）。</li>
                      <li><b>约束（Packet）</b>：仅支持 <code>NodeBuffer.packet</code>；并要求各 packet 的 <code>time_base</code> 一致。</li>
                      <li><b>约束（PCM）</b>：仅支持 <code>NodeBuffer.pcm</code>；要求各分支 PCM 的 <code>AudioFormat</code> / <code>time_base</code> 一致（<code>nb_samples</code> 可不同）。</li>
                      <li><b>注意</b>：Packet 输出的时间戳/flags 等元数据不会自动合并（当前实现只保留 time_base，并生成新的 data）。PCM 输出的 pts/time_base 取第一帧作为基准。</li>
                    </ul>
                  </li>
                  <li>
                    <b>ast.ReduceXor()</b>
                    <ul>
                      <li><b>作用</b>：对 Packet 做<b>按字节异或</b>：<code>out[i] = pkt0[i] ^ pkt1[i] ^ ...</code>。</li>
                      <li><b>限制</b>：仅支持 <code>NodeBuffer.packet</code>；要求各 packet 的 <code>time_base</code> 一致，且 <code>data</code> 长度一致。</li>
                      <li><b>注意</b>：输出 packet 的时间戳/flags 等元数据不会自动合并（当前实现只保留 time_base，并生成新的 data）。</li>
                    </ul>
                  </li>
                </ul>

                <h3>示例：两分支 Packet → Python reduce 合并</h3>
                <pre><code class="language-python">import pyaudiostream as ast

# 两条分支：这里只用 identity 做示例（真实场景可放 encoder/decoder/processor 等）
b1 = [ast.make_identity_node("packet")]
b2 = [ast.make_identity_node("packet")]

def my_reduce(items: list[ast.NodeBuffer]) -&gt; ast.NodeBuffer:
    # items[0], items[1] 都是 packet
    p1 = items[0].as_packet()
    p2 = items[1].as_packet()
    if p1 is None or p2 is None:
        raise ValueError("expects packet")

    # 示例：把两个 packet 的首字节相加，输出一个新 packet
    s = (p1.data[0] + p2.data[0]) &amp; 0xFF
    out = ast.Packet(data=bytes([s]), time_base_num=p1.time_base_num, time_base_den=p1.time_base_den,
                     pts=p1.pts, dts=p1.dts, duration=p1.duration, flags=p1.flags)
    return ast.NodeBuffer.packet(out)

fj = ast.make_async_fork_join_node([b1, b2], my_reduce, name="fj")

# fj 本身是 DynNode，可直接插入 pipeline
p = ast.AsyncDynPipeline([fj])</code></pre>
              </section>

              <section class="section">
                <h2>AsyncDynPipeline</h2>
                <pre><code class="language-python">ast.AsyncDynPipeline(nodes: list[DynNode])</code></pre>
                <ul>
                  <li><b>nodes</b>：DynNode 列表。会被 move，不能复用。</li>
                  <li><b>push(buf=None)</b>：推入一帧输入。</li>
                  <li><b>try_get()</b>：非阻塞取输出。</li>
                  <li><b>get()</b>：阻塞等待输出或 EOF（内部会释放 GIL）。</li>
                  <li><b>reset(force=False)</b>：重置 pipeline（从起点向终点 reset，直到完成）。</li>
                </ul>
                <h3>push 参数说明</h3>
                <ul>
                  <li><b>buf</b>：<code>NodeBuffer</code> 或 None。None 等价于 flush（输入结束）。buf.kind 必须与 pipeline input_kind 匹配。</li>
                </ul>
                <h3>reset 参数说明</h3>
                <ul>
                  <li><b>force</b>：
                    <ul>
                      <li><code>False</code>：如果 Node 内有尚未结束的处理 flow，不会强制停止；会等待其处理到边界后再 reset。</li>
                      <li><code>True</code>：强制 reset（丢弃内部缓存/残留）。</li>
                    </ul>
                  </li>
                </ul>
              </section>

              <section class="section">
                <h2>AsyncDynRunner</h2>
                <pre><code class="language-python">ast.AsyncDynRunner(source, nodes: list[DynNode], sink)</code></pre>
                <ul>
                  <li><b>source</b>：实现 <code>pull()</code> 的对象（返回 NodeBuffer 或 None）。</li>
                  <li><b>nodes</b>：DynNode 列表（会被 move）。</li>
                  <li><b>sink</b>：实现 <code>push(buf)</code> + <code>finalize()</code> 的对象。</li>
                  <li><b>source.pull() -&gt; Optional[NodeBuffer]</b></li>
                  <li><b>sink.push(buf: NodeBuffer)</b></li>
                  <li><b>sink.finalize()</b></li>
                  <li><b>run()</b>：同步阻塞执行到完成（释放 GIL）。</li>
                </ul>
              </section>

              <section class="section">
                <h2>AudioFileReader</h2>
                <pre><code class="language-python">ast.AudioFileReader(path: str, format: str, chunk_samples: Optional[int] = None)</code></pre>
                <ul>
                  <li><b>path</b>：文件路径。</li>
                  <li><b>format</b>：容器格式字符串：wav/mp3/aac_adts/flac/opus_ogg（支持别名）。</li>
                  <li><b>chunk_samples</b>：每次读出 PCM 的 samples/每声道（目前主要对 wav 生效）。</li>
                  <li><b>next_frame(layout="planar")</b>：手动读 PCM。</li>
                  <li><b>pull()</b>：Runner 兼容（输出 PCM）。</li>
                  <li><b>format</b> 支持：wav/mp3/aac_adts/flac/opus_ogg（aac/adts、opus/ogg_opus 也可作为别名）。</li>
                </ul>
                <h3>next_frame 参数说明</h3>
                <ul>
                  <li><b>layout</b>：返回 PCM numpy 的布局：<code>"planar"</code> 或 <code>"interleaved"</code>。</li>
                </ul>
              </section>

              <section class="section">
                <h2>AudioFileWriter</h2>
                <pre><code class="language-python">ast.AudioFileWriter(path, format, input_format=None, bitrate=None, compression_level=None, wav_output_format=None)</code></pre>
                <ul>
                  <li><b>path</b>：输出文件路径。</li>
                  <li><b>format</b>：输出格式：wav/mp3/aac_adts/flac/opus_ogg。</li>
                  <li><b>input_format</b>：输入 PCM 格式。None 表示首帧推断（建议通过 <code>push(NodeBuffer)</code> 方式写入）。</li>
                  <li><b>bitrate</b>：可选码率（bps），用于 mp3/aac/opus。</li>
                  <li><b>compression_level</b>：可选压缩等级，用于 flac。</li>
                  <li><b>wav_output_format</b>：wav 输出样本格式：<code>"pcm16le"</code>（默认）或 <code>"f32le"</code>。</li>
                  <li><b>opus_ogg</b>：要求 48kHz 且 interleaved（<code>input_format.planar=False</code>）。</li>
                  <li><b>write_pcm(pcm)</b>：直接写入 numpy（要求已知 input_format；input_format=None 时不支持）。</li>
                  <li><b>push(buf)</b>：Runner 兼容写入（buf 必须是 pcm）。input_format=None 时会从首帧推断。</li>
                  <li><b>finalize()</b>：写尾/flush。</li>
                </ul>
                <h3>write_pcm 参数说明</h3>
                <ul>
                  <li><b>pcm</b>：2D numpy。shape/dtype 必须与 input_format 对齐。</li>
                </ul>
                <h3>push 参数说明</h3>
                <ul>
                  <li><b>buf</b>：NodeBuffer（必须是 pcm）。当 input_format=None 时会用首帧的 format 进行推断并锁定。</li>
                </ul>
              </section>

              <section class="section">
                <h2>ParallelAudioWriter</h2>
                <pre><code class="language-python">ast.ParallelAudioWriter(writers: list[AudioFileWriter | LineAudioWriter])</code></pre>
                <p>
                  把多个 <code>AudioFileWriter</code> 绑定成一个 writer，并在每次 <code>push()</code> / <code>finalize()</code> 时并行执行所有 writer。
                  适合“一路处理结果写多份文件”的场景。
                </p>
                <ul>
                  <li><b>writers</b>：要绑定的 writer 列表（支持 <code>AudioFileWriter</code>/<code>LineAudioWriter</code>，会被 move）。</li>
                  <li><b>bind(writer)</b>：追加绑定一个 writer（支持 <code>AudioFileWriter</code>/<code>LineAudioWriter</code>，同样会被 move）。</li>
                  <li><b>push(buf)</b>：Runner 兼容写入（buf 必须是 pcm）。</li>
                  <li><b>finalize()</b>：并行 finalize 所有 writer。</li>
                  <li><b>len</b>：当前绑定数量。</li>
                </ul>
                <h3>重要限制</h3>
                <ul>
                  <li><b>仅支持已初始化的 AudioFileWriter</b>：也就是构造 <code>AudioFileWriter</code> 时显式传入 <code>input_format</code> 的那种。</li>
                  <li><b>绑定后原 AudioFileWriter 不可再用</b>：内部 writer 会被取走（move）。</li>
                </ul>
                <h3>示例</h3>
                <pre><code class="language-python">import numpy as np
import pyaudiostream as ast

fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
w1 = ast.AudioFileWriter("a.wav", "wav", input_format=fmt)
w2 = ast.AudioFileWriter("b.wav", "wav", input_format=fmt)

pw = ast.ParallelAudioWriter([w1, w2])
pcm = np.zeros((2, 960), dtype=np.float32)
pw.push(ast.NodeBuffer.pcm(pcm, fmt))
pw.finalize()
                </code></pre>
              </section>

              <section class="section">
                <h2>LineAudioWriter</h2>
                <pre><code class="language-python">ast.LineAudioWriter(writer: AudioFileWriter, processors: Optional[list[Processor]] = None)</code></pre>
                <p>
                  <b>线性链路写端</b>：把多个 <code>Processor(PCM-&gt;PCM)</code> 串起来，最后把结果写入一个 <code>AudioFileWriter</code>。
                  适合由于使用ParallelAudioWriter时，需要对每个writer进行不同的前处理。比如需要对每个writer进行不同的重采样（因为不同的writer可能需要不同的重采样率和数据格式）。
                  注意：由于LineAudioWriter是同步的，所以不建议将长处理流程放在LineAudioWriter中，否则会导致阻塞！这种场景请使用AsyncPipelineAudioSink！
                </p>
                <ul>
                  <li><b>writer</b>：最终落地 writer（当前仅支持 <code>AudioFileWriter</code>）。会被 move。</li> 
                  <li><b>processors</b>：按顺序执行的一组 <code>Processor</code>。会被 move。</li>
                  <li><b>add_processor(p)</b>：追加一个 processor（同样会 move）。</li>
                  <li><b>push(buf)</b>：Runner 兼容写入（buf 必须是 pcm）。</li>
                  <li><b>finalize()</b>：flush processors 并 finalize writer。</li>
                </ul>
                <h3>示例：读文件 → 增益 → 并行写文件（由于不同writer需要不同的重采样率和数据格式，所以需要对每个writer进行不同的重采样）</h3>
                <pre><code class="language-python">import pyaudiostream as ast
import pyaudiostream as ast

src = ast.AudioFileReader("test.wav", "wav")  # 读文件并产出 PCM

out_fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="i16", planar=True)
out_fmt_mp3 = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="i16", planar=True) 
out_fmt_flac = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="i16", planar=False) 

dst = ast.AudioFileWriter("out_2.flac", "flac", compression_level=8, input_format=out_fmt_flac)
dst_2 = ast.AudioFileWriter("out_2.mp3", "mp3", bitrate=320_000, input_format=out_fmt_mp3)

dst = ast.LineAudioWriter(dst, [ast.Processor.resample(None, out_fmt_flac)])
dst_2 = ast.LineAudioWriter(dst_2, [ast.Processor.resample(None, out_fmt_mp3)])

pw = ast.ParallelAudioWriter([dst, dst_2])

nodes = [
    ast.make_processor_node(
        "gain",
        ast.GainNodeConfig(
            gain=1.2,
        ),
    )
]

r = ast.AsyncDynRunner(src, nodes, pw)
r.run()
                </code></pre>
              </section>

              <section class="section">
                <h2>AsyncPipelineAudioSink</h2>
                <pre><code class="language-python">ast.AsyncPipelineAudioSink(writer: AudioFileWriter, nodes: list[DynNode | Processor], queue_capacity: int = 8, handle_capacity: int = 32)</code></pre>
                <p>
                  <b>异步 pipeline 写入汇（推荐用于“长链路/重计算”场景）</b>：把多个 <code>Processor/Encoder/Deocder(PCM/Packet-&gt;PCM)</code> 拆成多段并行 stage（pipeline parallel），
                  最终顺序写入一个 <code>AudioFileWriter</code>。
                  和 <code>LineAudioWriter</code> 不同，它不会在 <code>AsyncDynRunner</code> 的输出侧形成长时间阻塞。
                </p>
                <h3>重要说明</h3>
                <ul>
                  <li><b>主要用法</b>：作为 <code>AsyncDynRunner(..., sink=...)</code> 的 sink 传入（由 Runner 的 tokio runtime 驱动）。</li>
                  <li><b>也可以直接 push</b>：用 <code>with ... as h</code> 或显式 <code>h = sink.start()</code> 获取 handle（同步接口 + 背压）。</li>
                  <li><b>handle_capacity</b>：Python -&gt; 后台线程的有界队列容量。满了会阻塞 <code>h.push()</code>（背压）。</li>
                  <li><b>queue_capacity</b>：sink 内部 pipeline 各 stage 之间的有界队列容量（背压）。越大吞吐更高但占用更多内存。</li>
                  <li><b>两者关系</b>：这是两层队列，整体排队/内存/延迟由两者共同决定；一般先调 <code>handle_capacity</code>（吸收调用侧突发），再按吞吐需要调 <code>queue_capacity</code>。</li>
                </ul>
                <h3>API（方法与参数）</h3>
                <ul>
                  <li><b>__init__(writer, nodes, queue_capacity=8, handle_capacity=32)</b>
                    <ul>
                      <li><b>writer</b>：最终写入的 <code>AudioFileWriter</code>（会被 move/搬空）。</li>
                      <li><b>nodes</b>：<code>DynNode</code>/<code>Processor</code> 列表（会被 move/搬空）。<code>Processor</code> 会自动包装成 PCM-&gt;PCM 的 DynNode；Encoder/Decoder 请用 <code>make_encoder_node</code>/<code>make_decoder_node</code>。只要相邻 input_kind/output_kind 匹配即可；最后一个节点必须输出 PCM。</li>
                      <li><b>queue_capacity</b>：内部 pipeline stage 之间队列容量（背压）。</li>
                      <li><b>handle_capacity</b>：默认用于 <code>with</code> / <code>__enter__</code> 的 handle 队列容量。</li>
                    </ul>
                  </li>
                  <li><b>start(handle_capacity=32) -&gt; AsyncPipelineAudioSinkHandle</b>：启动后台线程 + tokio runtime，返回可直接 push 的 handle。</li>
                  <li><b>stop() -&gt; None</b>：停止内部 handle（等价于 handle.finalize + join）。</li>
                  <li><b>__enter__() -&gt; AsyncPipelineAudioSinkHandle</b>：进入 <code>with</code> 时自动 start，并返回 handle。</li>
                  <li><b>__exit__(exc_type=None, exc=None, tb=None) -&gt; bool</b>：退出 <code>with</code> 时自动 stop（不吞异常）。</li>
                  <li><b>push(buf) / finalize()</b>：为接口兼容保留，但<b>不建议直接调用</b>（请用 Runner 或 handle）。</li>
                </ul>
                <h3>AsyncPipelineAudioSinkHandle（直接调用的 handle）</h3>
                <ul>
                  <li><b>push(buf: NodeBuffer) -&gt; None</b>：同步入队；队列满会阻塞（背压）。</li>
                  <li><b>finalize() -&gt; None</b>：同步等待后台 sink 完成 flush + finalize。</li>
                </ul>
                <h3>示例：长处理链路写文件（用 AsyncPipelineAudioSink 避免阻塞）</h3>
                <pre><code class="language-python">import pyaudiostream as ast

src = ast.AudioFileReader("test.wav", "wav")
out_fmt = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
dst = ast.AudioFileWriter("out_async.flac", "flac", compression_level=8, input_format=out_fmt)

# nodes 会被 move
sink = ast.AsyncPipelineAudioSink(
    dst,
    nodes=[
        ast.make_processor_node("resample", ast.ResampleNodeConfig(None, out_fmt, out_chunk_samples=960, pad_final=True)),
        ast.make_processor_node("gain", ast.GainNodeConfig(out_fmt, gain=1.1)),
    ],
    queue_capacity=8,
)

nodes = [ast.make_identity_node("pcm")]
r = ast.AsyncDynRunner(src, nodes, sink)
r.run()
                </code></pre>
                <h3>示例：with 管理（自动 start/stop，返回 handle 可直接 push）</h3>
                <pre><code class="language-python">import pyaudiostream as ast

dst = ast.AudioFileWriter("out_async2.flac", "flac", compression_level=8, input_format=out_fmt)
with ast.AsyncPipelineAudioSink(dst, [ast.make_processor_node("resample", ast.ResampleNodeConfig(None, out_fmt, out_chunk_samples=960, pad_final=True))]) as h:
    h.push(ast.NodeBuffer.pcm(pcm, out_fmt))
    h.finalize()
                </code></pre>
              </section>

              <section class="section">
                <h2>AsyncParallelAudioSink</h2>
                <pre><code class="language-python">ast.AsyncParallelAudioSink(sinks: list[AudioSink], handle_capacity: int = 32)</code></pre>
                <p>
                  <b>异步并行 fan-out 汇</b>：把每个输入同时发送到多个 sink，并发执行所有 sink 的 <code>push()</code>/<code>finalize()</code>。
                  适合把同一路输出“异步写多份结果”的场景（尤其当绑定的 sink 本身是异步的，比如 <code>AsyncPipelineAudioSink</code>）。
                </p>
                <h3>重要说明</h3>
                <ul>
                  <li><b>主要用法</b>：作为 <code>AsyncDynRunner(..., sink=...)</code> 的 sink 传入（由 Runner 的 tokio runtime 驱动）。</li>
                  <li><b>也可以直接 push</b>：用 <code>with ... as h</code> 或显式 <code>h = sink.start()</code> 获取 handle（同步接口 + 背压）。</li>
                </ul>
                <h3>API（方法与参数）</h3>
                <ul>
                  <li><b>__init__(sinks, handle_capacity=32)</b>
                    <ul>
                      <li><b>sinks</b>：要 fan-out 的 sink 列表（可以混合：<code>AsyncPipelineAudioSink</code> / <code>AsyncParallelAudioSink</code> / 自定义 Python sink）。</li>
                      <li><b>handle_capacity</b>：默认用于 <code>with</code> / <code>__enter__</code> 的 handle 队列容量。</li>
                    </ul>
                  </li>
                  <li><b>start(handle_capacity=32) -&gt; AsyncParallelAudioSinkHandle</b>：启动后台线程 + tokio runtime，返回可直接 push 的 handle。</li>
                  <li><b>stop() -&gt; None</b>：停止内部 handle（等价于 handle.finalize + join）。</li>
                  <li><b>__enter__() -&gt; AsyncParallelAudioSinkHandle</b>：进入 <code>with</code> 时自动 start，并返回 handle。</li>
                  <li><b>__exit__(exc_type=None, exc=None, tb=None) -&gt; bool</b>：退出 <code>with</code> 时自动 stop（不吞异常）。</li>
                  <li><b>push(buf) / finalize()</b>：为接口兼容保留，但<b>不建议直接调用</b>（请用 Runner 或 handle）。</li>
                </ul>
                <h3>AsyncParallelAudioSinkHandle（直接调用的 handle）</h3>
                <ul>
                  <li><b>push(buf: NodeBuffer) -&gt; None</b>：同步入队；队列满会阻塞（背压）。</li>
                  <li><b>finalize() -&gt; None</b>：同步等待后台 sink 完成并行 fan-out + finalize。</li>
                </ul>
                <h3>示例：异步写两份（每份不同的 processors/格式）</h3>
                <pre><code class="language-python">import pyaudiostream as ast

src = ast.AudioFileReader("test.wav", "wav")

fmt1 = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="f32", planar=True)
fmt2 = ast.AudioFormat(sample_rate=48000, channels=2, sample_type="i16", planar=False)

w1 = ast.AudioFileWriter("a.flac", "flac", compression_level=8, input_format=fmt1)
w2 = ast.AudioFileWriter("b.mp3", "mp3", bitrate=320_000, input_format=fmt2)

s1 = ast.AsyncPipelineAudioSink(w1, [ast.make_processor_node("resample", ast.ResampleNodeConfig(None, fmt1, out_chunk_samples=960, pad_final=True))], queue_capacity=8)
s2 = ast.AsyncPipelineAudioSink(w2, [ast.make_processor_node("resample", ast.ResampleNodeConfig(None, fmt2, out_chunk_samples=960, pad_final=True))], queue_capacity=8)

sink = ast.AsyncParallelAudioSink([s1, s2])
r = ast.AsyncDynRunner(src, [ast.make_identity_node("pcm")], sink)
r.run()
                </code></pre>
                <h3>示例：with 管理（自动 start/stop，返回 handle 可直接 push）</h3>
                <pre><code class="language-python">import pyaudiostream as ast

sink = ast.AsyncParallelAudioSink([s1, s2])
with sink as h:
    h.push(ast.NodeBuffer.pcm(pcm, out_fmt))
    h.finalize()
                </code></pre>
              </section>

              <section class="section">
                <h2>基类（仅用于类型提示/继承）</h2>
                <ul>
                  <li><b>ast.Node</b> / <b>ast.AudioSource</b> / <b>ast.AudioSink</b></li>
                </ul>
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


