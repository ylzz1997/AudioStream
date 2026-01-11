from __future__ import annotations

from typing import Optional, Literal, Any

import numpy as np
from numpy.typing import NDArray

SampleType = Literal["u8", "i16", "i32", "i64", "f32", "f64"]
Codec = Literal["wav", "pcm", "mp3", "aac", "opus", "flac"]


class AudioFormat:
    sample_rate: int
    channels: int
    sample_type: str
    planar: bool
    channel_layout_mask: int

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        sample_type: SampleType,
        planar: bool = True,
        channel_layout_mask: int = 0,
    ) -> None: ...


class WavEncoderConfig:
    input_format: Optional[AudioFormat]
    chunk_samples: int

    def __init__(self, input_format: Optional[AudioFormat] = None, chunk_samples: int = 1024) -> None: ...


class Mp3EncoderConfig:
    input_format: Optional[AudioFormat]
    chunk_samples: int
    bitrate: Optional[int]

    def __init__(
        self,
        input_format: Optional[AudioFormat] = None,
        chunk_samples: int = 1152,
        bitrate: Optional[int] = 128000,
    ) -> None: ...


class AacEncoderConfig:
    input_format: Optional[AudioFormat]
    chunk_samples: int
    bitrate: Optional[int]

    def __init__(
        self,
        input_format: Optional[AudioFormat] = None,
        chunk_samples: int = 1024,
        bitrate: Optional[int] = None,
    ) -> None: ...


class WavDecoderConfig:
    output_format: AudioFormat
    chunk_samples: int

    def __init__(self, output_format: AudioFormat, chunk_samples: int) -> None: ...


class Mp3DecoderConfig:
    chunk_samples: int
    packet_time_base_den: int

    def __init__(self, chunk_samples: int, packet_time_base_den: int = 48000) -> None: ...


class AacDecoderConfig:
    chunk_samples: int
    packet_time_base_den: int

    def __init__(self, chunk_samples: int, packet_time_base_den: int = 48000) -> None: ...

class OpusEncoderConfig:
    input_format: Optional[AudioFormat]
    chunk_samples: int
    bitrate: Optional[int]

    def __init__(
        self,
        input_format: Optional[AudioFormat] = None,
        chunk_samples: int = 960,
        bitrate: Optional[int] = 96000,
    ) -> None: ...


class FlacEncoderConfig:
    input_format: Optional[AudioFormat]
    chunk_samples: int
    compression_level: Optional[int]

    def __init__(
        self,
        input_format: Optional[AudioFormat] = None,
        chunk_samples: int = 4096,
        compression_level: Optional[int] = None,
    ) -> None: ...


class OpusDecoderConfig:
    chunk_samples: int
    packet_time_base_den: int
    extradata: Optional[bytes]

    def __init__(
        self,
        chunk_samples: int,
        packet_time_base_den: int = 48000,
        extradata: Optional[bytes] = None,
    ) -> None: ...


class FlacDecoderConfig:
    chunk_samples: int
    packet_time_base_den: int

    def __init__(self, chunk_samples: int, packet_time_base_den: int = 48000) -> None: ...


class Encoder:
    codec: str
    chunk_samples: int

    def __init__(
        self,
        codec: Codec,
        config: WavEncoderConfig | Mp3EncoderConfig | AacEncoderConfig | OpusEncoderConfig | FlacEncoderConfig,
    ) -> None: ...

    def put_frame(self, pcm: NDArray[np.generic], pts: Optional[int] = None, format: Optional[AudioFormat] = None) -> None: ...

    def get_frame(self, force: bool = False) -> Optional[bytes]: ...

    def get_packet(self, force: bool = False) -> Optional[Packet]: ...

    def reset(self) -> None: ...

    def pending_samples(self) -> int: ...

    def state(self) -> Literal["ready", "need_more", "empty"]: ...


class Decoder:
    codec: str
    chunk_samples: int

    def __init__(
        self,
        codec: Codec,
        config: WavDecoderConfig | Mp3DecoderConfig | AacDecoderConfig | OpusDecoderConfig | FlacDecoderConfig,
    ) -> None: ...

    def put_frame(self, frame: bytes) -> None: ...

    def put_packet(self, pkt: Packet) -> None: ...

    def get_frame(
        self,
        force: bool = False,
        layout: Literal["planar", "interleaved"] = "planar",
    ) -> Optional[NDArray[np.generic]]: ...

    def get_frame_info(
        self,
        force: bool = False,
        layout: Literal["planar", "interleaved"] = "planar",
    ) -> Optional[tuple[NDArray[np.generic], Optional[int], tuple[int, int]]]: ...

    def reset(self) -> None: ...

    def pending_samples(self) -> int: ...

    def state(self) -> Literal["ready", "need_more", "empty"]: ...


class Packet:
    data: bytes
    time_base_num: int
    time_base_den: int
    pts: Optional[int]
    dts: Optional[int]
    duration: Optional[int]
    flags: int

    def __init__(
        self,
        data: bytes,
        time_base_num: int = 1,
        time_base_den: int = 48000,
        pts: Optional[int] = None,
        dts: Optional[int] = None,
        duration: Optional[int] = None,
        flags: int = 0,
    ) -> None: ...


class NodeBuffer:
    kind: Literal["pcm", "packet"]

    @staticmethod
    def pcm(
        pcm: NDArray[np.generic],
        format: AudioFormat,
        pts: Optional[int] = None,
        time_base_num: Optional[int] = None,
        time_base_den: Optional[int] = None,
    ) -> NodeBuffer: ...

    @staticmethod
    def packet(pkt: Packet) -> NodeBuffer: ...

    def as_pcm(self) -> Optional[NDArray[np.generic]]: ...

    def as_pcm_with_layout(self, layout: Literal["planar", "interleaved"] = "planar") -> Optional[NDArray[np.generic]]: ...

    def as_packet(self) -> Optional[Packet]: ...

    def pcm_info(self) -> Optional[tuple[AudioFormat, Optional[int], tuple[int, int]]]: ...


class Node:

    def push(self, buf: Optional[NodeBuffer]) -> None: ...

    def pull(self) -> Optional[NodeBuffer]: ...


class AudioSource:

    def pull(self) -> Optional[NodeBuffer]: ...


class AudioSink:

    def push(self, buf: NodeBuffer) -> None: ...

    def finalize(self) -> None: ...


class DynNode:
    name: str
    input_kind: Literal["pcm", "packet"]
    output_kind: Literal["pcm", "packet"]


def make_identity_node(kind: Literal["pcm", "packet"]) -> DynNode: ...


class IdentityNodeConfig:
    kind: Literal["pcm", "packet"]

    def __init__(self, kind: Literal["pcm", "packet"]) -> None: ...

class ResampleNodeConfig:
    in_format: Optional[AudioFormat]
    out_format: AudioFormat
    out_chunk_samples: Optional[int]
    pad_final: bool

    def __init__(
        self,
        in_format: Optional[AudioFormat],
        out_format: AudioFormat,
        out_chunk_samples: Optional[int] = ...,
        pad_final: bool = ...,
    ) -> None: ...

class GainNodeConfig:
    format: Optional[AudioFormat]
    gain: float

    def __init__(self, format: Optional[AudioFormat] = None, gain: float = ...) -> None: ...

class CompressorNodeConfig:
    format: Optional[AudioFormat]
    sample_rate: Optional[float]
    threshold_db: float
    knee_width_db: float
    ratio: float
    expansion_ratio: float
    expansion_threshold_db: float
    attack_time: float
    release_time: float
    master_gain_db: float

    def __init__(
        self,
        format: Optional[AudioFormat],
        sample_rate: Optional[float] = ...,
        threshold_db: float = ...,
        knee_width_db: float = ...,
        ratio: float = ...,
        expansion_ratio: float = ...,
        expansion_threshold_db: float = ...,
        attack_time: float = ...,
        release_time: float = ...,
        master_gain_db: float = ...,
    ) -> None: ...

def make_processor_node(
    kind: Literal["identity", "resample", "gain", "compressor"],
    config: IdentityNodeConfig | ResampleNodeConfig | GainNodeConfig | CompressorNodeConfig,
) -> DynNode: ...


def make_encoder_node(
    codec: Codec,
    config: WavEncoderConfig | Mp3EncoderConfig | AacEncoderConfig | OpusEncoderConfig | FlacEncoderConfig,
) -> DynNode: ...


def make_decoder_node(
    codec: Codec,
    config: WavDecoderConfig | Mp3DecoderConfig | AacDecoderConfig | OpusDecoderConfig | FlacDecoderConfig,
) -> DynNode: ...

def make_python_node(
    obj: Node,
    input_kind: Literal["pcm", "packet"],
    output_kind: Literal["pcm", "packet"],
    name: str = "py-node",
) -> DynNode: ...


class AsyncDynPipeline:
    input_kind: Literal["pcm", "packet"]
    output_kind: Literal["pcm", "packet"]

    def __init__(self, nodes: list[DynNode]) -> None: ...

    def push(self, buf: Optional[NodeBuffer]) -> None: ...

    def flush(self) -> None: ...

    def try_get(self) -> Optional[NodeBuffer]: ...

    def get(self) -> Optional[NodeBuffer]: ...


class AsyncDynRunner:
    def __init__(self, source: AudioSource, nodes: list[DynNode], sink: AudioSink) -> None: ...

    def run(self) -> None: ...


class AudioFileReader:
    # chunk_samples 目前仅对 wav 生效；默认行为是按 1024 samples/帧切块（最后一帧可能更短）。
    def __init__(self, path: str, format: Literal["wav", "mp3", "aac_adts", "flac", "opus_ogg"], chunk_samples: Optional[int] = 1024) -> None: ...

    def next_frame(
        self,
        layout: Literal["planar", "interleaved"] = ...,
    ) -> Optional[NDArray[np.generic]]: ...

    # AsyncDynRunner 兼容：pull() -> Optional[NodeBuffer]
    def pull(self) -> Optional[NodeBuffer]: ...


class AudioFileWriter:
    def __init__(
        self,
        path: str,
        format: Literal["wav", "mp3", "aac_adts", "flac", "opus_ogg"],
        input_format: Optional[AudioFormat] = None,
        bitrate: Optional[int] = None,
        compression_level: Optional[int] = None,
        # None 等价于 "pcm16le"
        wav_output_format: Optional[Literal["pcm16le", "f32le"]] = "pcm16le",
    ) -> None: ...

    def write_pcm(self, pcm: NDArray[np.generic]) -> None: ...

    # AsyncDynRunner 兼容：push(buf) + finalize()
    def push(self, buf: NodeBuffer) -> None: ...

    def finalize(self) -> None: ...


class Processor:
    name: str

    @staticmethod
    def identity(format: Optional[AudioFormat] = None) -> Processor: ...

    @staticmethod
    def resample(
        in_format: Optional[AudioFormat],
        out_format: AudioFormat,
        out_chunk_samples: Optional[int] = None,
        pad_final: bool = True,
    ) -> Processor: ...

    @staticmethod
    def gain(
        format: Optional[AudioFormat] = None,
        gain: float = ...,
    ) -> Processor: ...

    @staticmethod
    def compressor(
        format: Optional[AudioFormat],
        sample_rate: float,
        threshold_db: float,
        knee_width_db: float,
        ratio: float,
        expansion_ratio: float,
        expansion_threshold_db: float,
        attack_time: float,
        release_time: float,
        master_gain_db: float,
    ) -> Processor: ...

    def put_frame(self, pcm: NDArray[np.generic], pts: Optional[int] = None, format: Optional[AudioFormat] = None) -> None: ...

    def flush(self) -> None: ...

    def get_frame(self, force: bool = False, layout: Literal["planar", "interleaved"] = "planar") -> Optional[NDArray[np.generic]]: ...

    def reset(self) -> None: ...

    def output_format(self) -> Optional[AudioFormat]: ...


