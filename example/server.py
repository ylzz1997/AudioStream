from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import time
import wave
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np


def _import_ast():
    import pyaudiostream as ast  # type: ignore
    return ast


HTML = r"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>AudioStream - MP3/AAC chunk streaming demo</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }
      .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
      button { padding: 8px 12px; }
      code { background: #f5f5f5; padding: 2px 6px; border-radius: 6px; }
      .log { white-space: pre-wrap; background: #0b1020; color: #d7e1ff; padding: 12px; border-radius: 8px; min-height: 120px; }
      .hint { color: #555; }
    </style>
  </head>
  <body>
    <h2>MP3/AAC 分帧推流 → 浏览器解码 → 流式播放</h2>
    <div class="hint">
      说明：这个页面会通过 <code>WebSocket</code> 接收服务端推送的编码帧（默认 <code>mp3</code>），并用 <code>MediaSource</code> 逐段 append 实现边下边播。<br/>
      注意：AAC(ADTS) 在不同浏览器的 MSE 支持不一致，推荐先用 MP3 验证链路。
    </div>

    <div class="row" style="margin-top: 12px;">
      <label>codec:
        <select id="codec">
          <option value="mp3" selected>mp3</option>
          <option value="aac">aac(adts)</option>
        </select>
      </label>
      <button id="btnStart">开始播放</button>
      <button id="btnStop" disabled>停止</button>
    </div>

    <div style="margin-top: 12px;">
      <audio id="audio" controls></audio>
    </div>

    <div style="margin-top: 12px;" class="log" id="log"></div>

    <script>
      const logEl = document.getElementById("log");
      const audioEl = document.getElementById("audio");
      const codecEl = document.getElementById("codec");
      const btnStart = document.getElementById("btnStart");
      const btnStop = document.getElementById("btnStop");

      function log(s) {
        logEl.textContent += s + "\n";
        logEl.scrollTop = logEl.scrollHeight;
      }

      function guessMime(codec) {
        if (codec === "mp3") return "audio/mpeg";
        // AAC(ADTS) 在 MSE 上经常不可用；这里尽力尝试
        return "audio/aac";
      }

      let ws = null;
      let ms = null;
      let sb = null;
      let queue = [];
      let meta = null;

      function reset() {
        try { if (ws) ws.close(); } catch {}
        ws = null;
        meta = null;
        queue = [];
        sb = null;
        ms = null;
        audioEl.removeAttribute("src");
        audioEl.load();
      }

      function pump() {
        if (!sb) return;
        if (sb.updating) return;
        if (queue.length === 0) return;
        const chunk = queue.shift();
        try {
          sb.appendBuffer(chunk);
        } catch (e) {
          log("[mse] appendBuffer error: " + e);
        }
      }

      btnStart.onclick = async () => {
        reset();
        const codec = codecEl.value;
        const url = `ws://${location.host}/ws?codec=${encodeURIComponent(codec)}`;
        log("[ui] connecting: " + url);

        ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
          btnStart.disabled = true;
          btnStop.disabled = false;
          log("[ws] open");
        };

        ws.onclose = () => {
          log("[ws] close");
          btnStart.disabled = false;
          btnStop.disabled = true;
          try { if (ms && ms.readyState === "open") ms.endOfStream(); } catch {}
        };

        ws.onerror = (e) => log("[ws] error: " + e);

        ws.onmessage = (ev) => {
          if (typeof ev.data === "string") {
            meta = JSON.parse(ev.data);
            const mime = meta.mime || guessMime(meta.codec);
            log("[meta] " + JSON.stringify(meta));
            log("[mse] mime=" + mime + " supported=" + MediaSource.isTypeSupported(mime));

            ms = new MediaSource();
            audioEl.src = URL.createObjectURL(ms);

            ms.addEventListener("sourceopen", () => {
              try {
                sb = ms.addSourceBuffer(mime);
              } catch (e) {
                log("[mse] addSourceBuffer failed: " + e);
                return;
              }
              sb.mode = "sequence";
              sb.addEventListener("updateend", pump);
              pump();
            });
            return;
          }

          // binary chunk
          const u8 = new Uint8Array(ev.data);
          queue.push(u8);
          pump();

          // 让它尽早开始播放（需要用户手势触发的 start 点击已经有了）
          if (audioEl.paused && queue.length > 2) {
            audioEl.play().catch((e) => log("[audio] play() blocked: " + e));
          }
        };
      };

      btnStop.onclick = () => {
        log("[ui] stop");
        reset();
        btnStart.disabled = false;
        btnStop.disabled = true;
      };
    </script>
  </body>
</html>
"""


@dataclass
class StreamCfg:
    host: str
    http_port: int
    ingest_port: int
    mode: str  # "server" | "sender" | "demo"
    codec: str  # "mp3" | "aac"
    sample_rate: int
    channels: int
    chunk_samples: int
    bitrate: int
    seconds: float
    wav: str


def _pcm_f32_planar_from_i16_interleaved(raw_i16: np.ndarray, channels: int) -> np.ndarray:
    """
    raw_i16: shape=(samples*channels,) interleaved
    returns: shape=(channels, samples) float32 in [-1,1]
    """
    if raw_i16.dtype != np.int16:
        raw_i16 = raw_i16.astype(np.int16, copy=False)
    if channels <= 0:
        raise ValueError("channels must be > 0")
    if raw_i16.size % channels != 0:
        raise ValueError("wav data size not divisible by channels")
    frames = raw_i16.reshape(-1, channels)  # (samples, ch)
    planar = frames.T  # (ch, samples)
    return (planar.astype(np.float32) / 32768.0).astype(np.float32, copy=False)


def iter_pcm_chunks(cfg: StreamCfg) -> AsyncIterator[np.ndarray]:
    """
    产出 numpy float32 planar PCM: shape=(channels, n<=chunk_samples)
    """

    async def _from_wav(path: str) -> AsyncIterator[np.ndarray]:
        with wave.open(path, "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            if ch != cfg.channels:
                raise ValueError(f"wav channels={ch} 与 cfg.channels={cfg.channels} 不一致")
            if sr != cfg.sample_rate:
                raise ValueError(f"wav sample_rate={sr} 与 cfg.sample_rate={cfg.sample_rate} 不一致")
            if sampwidth != 2:
                raise ValueError(f"仅支持 16-bit PCM wav(sampwidth=2)，当前 sampwidth={sampwidth}")

            need_samples = int(cfg.seconds * cfg.sample_rate) if cfg.seconds > 0 else wf.getnframes()
            left = need_samples
            while left > 0:
                want = min(cfg.chunk_samples, left)
                raw = wf.readframes(want)
                if not raw:
                    break
                i16 = np.frombuffer(raw, dtype=np.int16)
                pcm = _pcm_f32_planar_from_i16_interleaved(i16, cfg.channels)
                yield pcm
                left -= pcm.shape[1]

    async def _sine() -> AsyncIterator[np.ndarray]:
        total = int(cfg.seconds * cfg.sample_rate) if cfg.seconds > 0 else cfg.sample_rate * 10
        phase = 0.0
        freq = 440.0
        amp = 0.2
        produced = 0
        while produced < total:
            n = min(cfg.chunk_samples, total - produced)
            t = (np.arange(n, dtype=np.float64) / float(cfg.sample_rate)).astype(np.float64)
            # 相位连续的正弦：sin(2πf(t + phase))
            s = np.sin(2.0 * math.pi * freq * t + phase) * amp
            phase = float((phase + 2.0 * math.pi * freq * (n / float(cfg.sample_rate))) % (2.0 * math.pi))
            mono = s.astype(np.float32, copy=False)[None, :]
            pcm = np.tile(mono, (cfg.channels, 1))
            yield pcm
            produced += n

    if cfg.wav:
        return _from_wav(cfg.wav)
    return _sine()


async def iter_encoded_frames(cfg: StreamCfg) -> AsyncIterator[bytes]:
    ast = _import_ast()

    fmt = ast.AudioFormat(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        sample_type="f32",
        planar=True,
    )
    if cfg.codec == "mp3":
        enc_cfg = ast.Mp3EncoderConfig(fmt, cfg.chunk_samples, bitrate=cfg.bitrate)
    elif cfg.codec == "aac":
        enc_cfg = ast.AacEncoderConfig(fmt, cfg.chunk_samples, bitrate=cfg.bitrate)
    else:
        raise ValueError("codec 仅支持 mp3/aac")

    enc = ast.Encoder(cfg.codec, enc_cfg)

    start = time.perf_counter()
    sent_pcm_samples = 0

    async for pcm in iter_pcm_chunks(cfg):
        enc.put_frame(pcm)
        sent_pcm_samples += int(pcm.shape[1])

        # 把 encoder 当前可取出的 packet 全部吐出来
        get_force = False
        while True:
            pkt = enc.get_frame(force=get_force)
            if pkt is None:
                if enc.state() == "empty":
                    break
                elif enc.state() == "need_more":
                    get_force = True
                    continue
                else:
                    raise ValueError(f"encoder state: {enc.state()}")
            yield bytes(pkt)

        # 近似按实时节奏推送，避免浏览器一次性堆积太多 buffer
        expected = sent_pcm_samples / float(cfg.sample_rate)
        now = time.perf_counter() - start
        delay = expected - now
        if delay > 0:
            await asyncio.sleep(min(delay, 0.2))

    # flush
    while True:
        pkt = enc.get_frame(force=True)
        if pkt is None:
            break
        yield bytes(pkt)


def _codec_to_mime(codec: str) -> str:
    # 浏览器端主要保证 mp3 可播
    if codec == "mp3":
        return "audio/mpeg"
    # AAC(ADTS) 的 MSE 支持不稳定：这里尽力而为
    return "audio/aac"


async def _broadcast(clients: set, frame: bytes) -> None:
    if not clients:
        return
    # 并发发送，单个慢 client 不要拖垮整体（失败就剔除）
    ws_list = [ws for ws in list(clients) if not ws.closed]
    tasks = [ws.send_bytes(frame) for ws in ws_list]
    if not tasks:
        return
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 清理异常/关闭的连接
    for ws, r in zip(ws_list, results):
        if ws.closed:
            clients.discard(ws)
        elif isinstance(r, Exception):
            clients.discard(ws)


async def _recv_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    buf = await reader.readexactly(n)
    return buf


async def _recv_frame(reader: asyncio.StreamReader) -> bytes | None:
    header = await _recv_exact(reader, 4)
    ln = int.from_bytes(header, "big")
    if ln == 0:
        return None
    return await _recv_exact(reader, ln)


def _send_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(len(payload).to_bytes(4, "big") + payload)


async def run_server(cfg: StreamCfg) -> int:
    try:
        from aiohttp import web
    except Exception as e:
        raise SystemExit(
            "缺少依赖：aiohttp。\n"
            "请先安装：\n"
            "  python -m pip install -U aiohttp\n"
            f"\n原始错误：{e!r}\n"
        )

    routes = web.RouteTableDef()

    @routes.get("/")
    async def index(_req: web.Request) -> web.Response:
        # aiohttp: charset 不能写在 content_type 里，要单独用 charset 参数
        return web.Response(text=HTML, content_type="text/html", charset="utf-8")

    clients: set = set()

    @routes.get("/ws")
    async def ws(req: web.Request) -> web.WebSocketResponse:
        codec = (req.query.get("codec") or cfg.codec).lower()
        if codec not in ("mp3", "aac"):
            codec = "mp3"

        ws = web.WebSocketResponse(autoping=True, heartbeat=20)
        await ws.prepare(req)
        clients.add(ws)

        meta = {
            "codec": codec,
            "mime": _codec_to_mime(codec),
            "sample_rate": cfg.sample_rate,
            "channels": cfg.channels,
            "chunk_samples": cfg.chunk_samples,
            "bitrate": cfg.bitrate,
        }
        await ws.send_str(json.dumps(meta, ensure_ascii=False))

        try:
            async for _msg in ws:
                # 当前不需要客户端消息；只是保持连接。
                pass
        finally:
            clients.discard(ws)
            await ws.close()
        return ws

    app = web.Application()
    app.add_routes(routes)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, cfg.host, cfg.http_port)
    await site.start()

    async def handle_ingest(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        print(f"[ingest] connected from {peer}", flush=True)
        try:
            while True:
                frame = await _recv_frame(reader)
                if frame is None:
                    break
                await _broadcast(clients, frame)
        except asyncio.IncompleteReadError:
            pass
        except Exception as e:
            print(f"[ingest] error: {e!r}", flush=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            print(f"[ingest] disconnected {peer}", flush=True)

    ingest_srv = await asyncio.start_server(handle_ingest, cfg.host, cfg.ingest_port)

    print(f"[server] http://{cfg.host}:{cfg.http_port}  (open in browser)", flush=True)
    print(f"[server] websocket: ws://{cfg.host}:{cfg.http_port}/ws?codec=mp3", flush=True)
    print(f"[server] ingest tcp: {cfg.host}:{cfg.ingest_port} (4-byte big-endian length + payload; 0=EOF)", flush=True)
    print("[server] Ctrl+C to stop", flush=True)

    # keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        ingest_srv.close()
        await ingest_srv.wait_closed()
        await runner.cleanup()
    return 0


async def run_sender(cfg: StreamCfg) -> int:
    """
    用 pyaudiostream 编码出 mp3/aac 帧
    """
    # 先连上 ingest
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    # Windows 下 Proactor 对 start_server/stream 更友好；这里保持纯 asyncio
    for _ in range(200):
        try:
            reader, writer = await asyncio.open_connection(cfg.host, cfg.ingest_port)
            break
        except OSError:
            await asyncio.sleep(0.05)
    else:
        print("[sender] connect failed", flush=True)
        return 2

    print(f"[sender] connected to ingest {cfg.host}:{cfg.ingest_port}", flush=True)
    if cfg.wav:
        print(f"[sender] source wav: {cfg.wav}", flush=True)
    else:
        print(f"[sender] source: sine {cfg.seconds:.2f}s", flush=True)

    try:
        async for frame in iter_encoded_frames(cfg):
            _send_frame(writer, frame)
            await writer.drain()
        # EOF
        _send_frame(writer, b"")
        await writer.drain()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
    print("[sender] done", flush=True)
    return 0


async def run_demo(cfg: StreamCfg) -> int:
    """
    单文件自测：同一进程里起 server + sender。
    """
    server_task = asyncio.create_task(run_server(cfg))
    try:
        # 给 http/ws/ingest 一点启动时间
        await asyncio.sleep(0.3)
        _ = await run_sender(cfg)
        # demo 默认保持 server 持续运行（方便多次点“开始播放”）
        await server_task
    finally:
        if not server_task.done():
            server_task.cancel()
            with contextlib.suppress(Exception):
                await server_task
    return 0


def parse_args() -> StreamCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["server", "sender", "demo"], default="demo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=23456)
    p.add_argument("--ingest-port", type=int, default=23457)
    p.add_argument("--codec", choices=["mp3", "aac"], default="mp3")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--chunk-samples", type=int, default=1152, help="PCM chunk 大小（越小延迟越低，越大码率更稳定）")
    p.add_argument("--bitrate", type=int, default=128_000)
    p.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="推流时长；<=0 表示尽可能读完整个 wav（如果 --wav 指定）",
    )
    p.add_argument("--wav", default="", help="可选：16-bit PCM wav 文件路径（不填则发送正弦波）")
    args = p.parse_args()

    wav_path = args.wav.strip()
    if wav_path:
        wav_path = os.path.abspath(wav_path)
        if not os.path.exists(wav_path):
            raise SystemExit(f"wav not found: {wav_path}")

    return StreamCfg(
        host=args.host,
        http_port=args.http_port,
        ingest_port=args.ingest_port,
        mode=args.mode,
        codec=args.codec,
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk_samples=args.chunk_samples,
        bitrate=args.bitrate,
        seconds=float(args.seconds),
        wav=wav_path,
    )


def main() -> int:
    cfg = parse_args()
    if cfg.mode == "server":
        return asyncio.run(run_server(cfg))
    if cfg.mode == "sender":
        return asyncio.run(run_sender(cfg))
    return asyncio.run(run_demo(cfg))


if __name__ == "__main__":
    raise SystemExit(main())


