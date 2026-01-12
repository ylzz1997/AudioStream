<p align="center">
<img src="logo/logo.png" alt="AudioStream" width="720" />
</p>

[中文README(Chinese README)](README_CN.md)

# AudioStream

A Rust library designed for **audio stream processing**, exposed to Python via PyO3: You can push **PCM (numpy)** chunks to the `Encoder` to obtain **encoded frames (bytes)**; or push **encoded frames (bytes)** to the `Decoder` to retrieve **PCM (numpy)** chunks.

Currently, the most common scenarios on the Python side are:

* **Python Generates/Reads PCM → Encodes to MP3/AAC Frames → Pushes Stream over Network**
* **Python/Network Receives MP3/AAC Frames → Decodes to PCM → Further Processing/Saving to Disk**

> Note: MP3/AAC encoding/decoding usually depends on the `ffmpeg` feature (see installation instructions below).

---

## Features Overview

* **Python API (PyO3)**: `Encoder` / `Decoder`
* **Input/Output**:
* **Encoder**: `put_frame(pcm_numpy)` → `get_frame() -> bytes`
* **Decoder**: `put_frame(frame_bytes)` → `get_frame() -> pcm_numpy`


* **Chunk Semantics**:
* Internally buffers input into a FIFO; only produces an output frame when accumulated data reaches `chunk_samples`.
* `get_frame(force=True)` flushes residual data at the end (including EOF flush for the underlying codec).



---

## Dependencies & Build (Python)

**Warning⚠️: FFMPEG MUST BE INSTALLED FIRST!!!!!**

**Otherwise, the MP3/AAC encoder will report that ffmpeg is required.**

In the project root directory:

```bash
python -m pip install -U maturin numpy
maturin develop -F python -F ffmpeg

```

After installation, you can use it in Python:

```python
import pyaudiostream as ast

```

---

## Python API Quick Start

### Encoder (PCM → Encoded Frame bytes)

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

# It is recommended to flush at the end
while True:
    last = enc.get_frame(force=True)
    if last is None:
        break

```

### Decoder (Encoded Frame bytes → PCM numpy)

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

## Example 1: `example/server.py` (Python Encoding/Streaming → Browser Decoding/Playback)

This example includes three modes covering the full pipeline:

* **server**: Provides a Web page + WebSocket broadcast; also opens a TCP ingest port to receive "frame-based" pushes.
* **sender**: Uses `pyaudiostream` to encode PCM (wav or sine wave) into MP3/AAC frames and pushes them to the ingest port via custom framing.
* **demo**: Runs both server and sender in the same process (convenient for one-command self-testing).

### Install Dependencies

```bash
python -m pip install -U aiohttp

```

(Ensure you have run `maturin develop -F python -F ffmpeg`)

### One-Command Self-Test (Recommended)

```bash
python example/server.py --mode demo --codec mp3

```

Open in browser: `http://127.0.0.1:23456` and click "Start Playback".

### Running Separately (server / sender)

Start server (HTTP+WS + TCP ingest):

```bash
python example/server.py --mode server --codec mp3 --http-port 23456 --ingest-port 23457

```

Start sender (Encode from wav and push):

```bash
python example/server.py --mode sender --codec mp3 --host 127.0.0.1 --ingest-port 23457 --wav "/abs/path/to/input.wav" --seconds 0

```

#### Key Parameters

* `--wav`
* Currently, the example requires **16-bit PCM WAV** (sampwidth=2).
* The wav's **sample rate/channels** must match `--sample-rate/--channels`, otherwise it will error out (to avoid "pitch shifting").


* `--chunk-samples`
* PCM chunk size; smaller means lower latency, but the encoder/browser side may suffer from buffer underrun.
* MP3 usually uses 1152 samples per channel (example default is 1152).
* AAC usually uses 1024.



### Browser Note

* **MP3**: Uses MSE (MediaSource) with `audio/mpeg` streaming append; has the best compatibility.
* **AAC(ADTS)**: MSE support for `audio/aac` varies across browsers; it is recommended to prioritize MP3 for pipeline verification.

## Framework Design: AudioFlowModel

AudioFlowModel is an abstraction layer designed by the author for **streaming audio computation**: It treats continuously arriving audio data as a "Stream" and modules like encoding, decoding, and DSP processing as composable "Operators," building a flexible audio processing pipeline through a standardized protocol.

### 1. Core Design Philosophy

We use a **Dataflow Model** to unify the description of the audio processing workflow:

* **Stream**: The continuous flow of data along the time axis. In this project, the payload of a stream is strictly typed into two categories of `NodeBuffer`:
* `PCM`: Raw audio frames (`AudioFrame`).
* `Packet`: Encoded data packets (`CodecPacket`).


* **Operator/Node**: The computation unit that transforms the stream. Operators are **Stateful**, potentially containing encoder latency, filter states, or internal caches.
* **Composition**: Operators follow a unified input/output protocol and can be chained like building blocks to form an end-to-end processing pipeline.

### 2. Component Architecture

#### 2.1 Codec (Core Operator)

This is the atomic computation unit for audio processing, acting directly on the data content.

* **Encoder**: Receives PCM input, maintains internal state (e.g., Lookahead, Bit Reservoir), and produces encoded packets. (Compresses audio signals into bitstreams: `PCM -> CodecPacket`)
* **Decoder**: Receives encoded packet input and produces PCM audio. (Restores bitstreams to audio signals: `CodecPacket -> PCM`)
* **Processor**: PCM to PCM transformation. (Processes audio signals in the time domain, such as gain, mixing, resampling).
* *Special Implementation* `IdentityProcessor`: A "pass-through" operator (`id: PCM -> PCM`) that transmits data without modification. Often used for placeholders, pipeline testing, or interface alignment.

#### 2.2 IO (Source & Sink)

Responsible for connecting the computation graph to the external world:

* **AudioReader (Source)**: The **Producer** of the stream. Responsible for reading data from external sources (files, network streams, devices) and converting it into system `PCM` or `Packet` streams to introduce into the computation graph.
* **AudioWriter (Sink)**: The **Consumer** of the stream. Responsible for receiving `PCM` or `Packet` output from the computation graph and writing it to external storage or sending it to a network/sound card.

#### 2.3 Node (Wrapper Layer)

`Node` is the standardized encapsulation of the aforementioned Codec and IO, allowing different operators to be uniformly scheduled by the Pipeline. It defines the lifecycle and interaction protocol for stream processing:

* **Protocol (Push/Pull)**: External callers Push data to the node and Pull results from the node.
* **Backpressure**:
* `Again`: Returns this signal to let the scheduler retry later when the node needs more input to produce output (e.g., encoder buffer not full) or when output is temporarily unavailable.


* **Lifecycle (EOF & Flush)**:
* When the input stream ends, a **Flush** operation is triggered by Pushing `None`, forcing the node to drain all internal residual buffers.
* When Flush is complete and there is no more output, `Eof` is returned.



**Implementation Forms:**

* **DynNode (Dynamic)**: Uses the `NodeBuffer` enum for type erasure, allowing arbitrarily complex pipelines to be assembled dynamically at runtime via `Vec<Box<dyn DynNode>>` (This interface is exposed to Python).
* **StaticNode (Static)**: Uses the Rust generic system to determine types at compile time, providing stronger type safety and less runtime overhead (Not usable via Python interface!).
* *Special Implementation* `IdentityNode`: A "straight pipe" at the node layer. It transfers input Buffer ownership to output as-is (Zero-copy move). Often used to build test scaffolds to verify Runner driver logic and backpressure propagation mechanisms.

#### 2.4 Pipeline

Pipeline is the abstraction for "Node Composition". It connects a set of `Node`s in sequence (Chaining), automatically handling data transfer between nodes.

Typical pipeline examples:

```text
[Source] -> PCM --(Processor)--> PCM --(Encoder)--> Packet -> [Sink]

```

Or:

```text
[Source] -> Packet --(Decoder)--> PCM --(Processor)--> PCM -> [Sink]

```

* **AsyncPipeline**: A pipeline designed for asynchronous environments (Tokio). It typically splits the pipeline into a Producer (input end) and Consumer (output end), supporting non-blocking Push/Pull, facilitating parallel driving in multi-tasking environments.

In short, a Pipeline must have an input and an output, but the input/output can be `PCM` or `Packet` (The Runner doesn't strictly need a Pipeline, but requires a Source and Sink to be specified).

#### 2.5 Runner (Scheduler)

The Runner is the execution engine that gives "life" to the static computation graph. It acts as the driver for the entire Pipeline:

1. Reads data from Source and Pushes into Pipeline.
2. Continuously Pulls Pipeline output and writes to Sink.
3. Correctly handles `Again` (wait/yield time slice).
4. Propagates the Flush signal when input ends until all nodes finish processing (`Eof`).

* **AsyncRunner**: The asynchronous implementation of Runner. It combines computation logic with the async nature of underlying I/O (like Network Ingest, WebSocket Broadcast, File I/O) to build end-to-end real-time stream processing applications.

Refer to the diagram below for the scheduling method:
![调度流程图](logo/i1.jpg)