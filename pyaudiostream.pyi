from __future__ import annotations

from typing import Optional, Literal

import numpy as np
from numpy.typing import NDArray

SampleType = Literal["u8", "i16", "i32", "i64", "f32", "f64"]
Codec = Literal["wav", "pcm", "mp3", "aac"]


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


class Encoder:
    codec: str
    chunk_samples: int

    def __init__(
        self,
        codec: Codec,
        config: WavEncoderConfig | Mp3EncoderConfig | AacEncoderConfig,
    ) -> None: ...

    def put_frame(self, pcm: NDArray[np.generic]) -> None: ...

    def get_frame(self, force: bool = ...) -> Optional[bytes]: ...

    def pending_samples(self) -> int: ...

    def state(self) -> Literal["ready", "need_more", "empty"]: ...


class Decoder:
    codec: str
    chunk_samples: int

    def __init__(
        self,
        codec: Codec,
        config: WavDecoderConfig | Mp3DecoderConfig | AacDecoderConfig,
    ) -> None: ...

    def put_frame(self, frame: bytes) -> None: ...

    def get_frame(self, force: bool = ...) -> Optional[NDArray[np.generic]]: ...

    def pending_samples(self) -> int: ...

    def state(self) -> Literal["ready", "need_more", "empty"]: ...


