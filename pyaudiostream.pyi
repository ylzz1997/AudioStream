from __future__ import annotations

from typing import Optional, Literal

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
        planar: bool = ...,
        channel_layout_mask: int = ...,
    ) -> None: ...


class WavEncoderConfig:
    input_format: AudioFormat
    chunk_samples: int

    def __init__(self, input_format: AudioFormat, chunk_samples: int) -> None: ...


class Mp3EncoderConfig:
    input_format: AudioFormat
    chunk_samples: int
    bitrate: Optional[int]

    def __init__(
        self,
        input_format: AudioFormat,
        chunk_samples: int,
        bitrate: Optional[int] = ...,
    ) -> None: ...


class AacEncoderConfig:
    input_format: AudioFormat
    chunk_samples: int
    bitrate: Optional[int]

    def __init__(
        self,
        input_format: AudioFormat,
        chunk_samples: int,
        bitrate: Optional[int] = ...,
    ) -> None: ...


class WavDecoderConfig:
    output_format: AudioFormat
    chunk_samples: int

    def __init__(self, output_format: AudioFormat, chunk_samples: int) -> None: ...


class Mp3DecoderConfig:
    chunk_samples: int
    packet_time_base_den: int

    def __init__(self, chunk_samples: int, packet_time_base_den: int = ...) -> None: ...


class AacDecoderConfig:
    chunk_samples: int
    packet_time_base_den: int

    def __init__(self, chunk_samples: int, packet_time_base_den: int = ...) -> None: ...

class OpusEncoderConfig:
    input_format: AudioFormat
    chunk_samples: int
    bitrate: Optional[int]

    def __init__(
        self,
        input_format: AudioFormat,
        chunk_samples: int,
        bitrate: Optional[int] = ...,
    ) -> None: ...


class FlacEncoderConfig:
    input_format: AudioFormat
    chunk_samples: int
    compression_level: Optional[int]

    def __init__(
        self,
        input_format: AudioFormat,
        chunk_samples: int,
        compression_level: Optional[int] = ...,
    ) -> None: ...


class OpusDecoderConfig:
    chunk_samples: int
    packet_time_base_den: int
    extradata: Optional[bytes]

    def __init__(
        self,
        chunk_samples: int,
        packet_time_base_den: int = ...,
        extradata: Optional[bytes] = ...,
    ) -> None: ...


class FlacDecoderConfig:
    chunk_samples: int
    packet_time_base_den: int

    def __init__(self, chunk_samples: int, packet_time_base_den: int = ...) -> None: ...


class Encoder:
    codec: str
    chunk_samples: int

    def __init__(
        self,
        codec: Codec,
        config: WavEncoderConfig | Mp3EncoderConfig | AacEncoderConfig | OpusEncoderConfig | FlacEncoderConfig,
    ) -> None: ...

    def put_frame(self, pcm: NDArray[np.generic]) -> None: ...

    def get_frame(self, force: bool = ...) -> Optional[bytes]: ...

    def get_packet(self, force: bool = ...) -> Optional[Packet]: ...

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
        force: bool = ...,
        layout: Literal["planar", "interleaved"] = ...,
    ) -> Optional[NDArray[np.generic]]: ...

    def get_frame_info(
        self,
        force: bool = ...,
        layout: Literal["planar", "interleaved"] = ...,
    ) -> Optional[tuple[NDArray[np.generic], Optional[int], tuple[int, int]]]: ...

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
        time_base_num: int = ...,
        time_base_den: int = ...,
        pts: Optional[int] = ...,
        dts: Optional[int] = ...,
        duration: Optional[int] = ...,
        flags: int = ...,
    ) -> None: ...


class NodeBuffer:
    kind: Literal["pcm", "packet"]

    @staticmethod
    def pcm(
        pcm: NDArray[np.generic],
        format: AudioFormat,
        pts: Optional[int] = ...,
        time_base_num: Optional[int] = ...,
        time_base_den: Optional[int] = ...,
    ) -> NodeBuffer: ...

    @staticmethod
    def packet(pkt: Packet) -> NodeBuffer: ...

    def as_pcm(self) -> Optional[NDArray[np.generic]]: ...

    def as_pcm_with_layout(self, layout: Literal["planar", "interleaved"] = ...) -> Optional[NDArray[np.generic]]: ...

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


def make_resample_node(
    in_format: AudioFormat,
    out_format: AudioFormat,
    out_chunk_samples: Optional[int] = ...,
    pad_final: bool = ...,
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
    name: str = ...,
) -> DynNode: ...


class AsyncDynPipeline:
    input_kind: Literal["pcm", "packet"]
    output_kind: Literal["pcm", "packet"]

    def __init__(self, nodes: list[DynNode]) -> None: ...

    def push(self, buf: NodeBuffer) -> None: ...

    def flush(self) -> None: ...

    def try_get(self) -> Optional[NodeBuffer]: ...

    def get(self) -> Optional[NodeBuffer]: ...


class AsyncDynRunner:
    def __init__(self, source: AudioSource, nodes: list[DynNode], sink: AudioSink) -> None: ...

    def run(self) -> None: ...


class AudioFileReader:
    def __init__(self, path: str, format: Literal["wav", "mp3", "aac_adts", "flac", "opus_ogg"]) -> None: ...

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
        input_format: AudioFormat,
        bitrate: Optional[int] = ...,
        compression_level: Optional[int] = ...,
    ) -> None: ...

    def write_pcm(self, pcm: NDArray[np.generic]) -> None: ...

    # AsyncDynRunner 兼容：push(buf) + finalize()
    def push(self, buf: NodeBuffer) -> None: ...

    def finalize(self) -> None: ...


class Processor:
    name: str

    @staticmethod
    def identity(format: Optional[AudioFormat] = ...) -> Processor: ...

    @staticmethod
    def resample(
        in_format: AudioFormat,
        out_format: AudioFormat,
        out_chunk_samples: Optional[int] = ...,
        pad_final: bool = ...,
    ) -> Processor: ...

    def put_frame(self, pcm: NDArray[np.generic], pts: Optional[int] = ...) -> None: ...

    def flush(self) -> None: ...

    def get_frame(self, layout: Literal["planar", "interleaved"] = ...) -> Optional[NDArray[np.generic]]: ...

    def output_format(self) -> Optional[AudioFormat]: ...


