//! 静态版 Node 接口：用关联类型保证 pipeline 连接类型正确（编译期校验）。

use crate::codec::decoder::decoder_interface::AudioDecoder;
use crate::codec::encoder::encoder_interface::AudioEncoder;
use crate::codec::error::{CodecError, CodecResult};
use crate::codec::packet::CodecPacket;
use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::audio::audio::{AudioFrame, AudioFrameView};
use std::collections::VecDeque;

/// 静态 node：In/Out 在类型层面固定，`pull()` 以 `CodecError::{Again,Eof}` 表示背压/结束。
pub trait StaticNode {
    type In;
    type Out;

    fn name(&self) -> &'static str;
    fn push(&mut self, input: Option<Self::In>) -> CodecResult<()>;
    fn pull(&mut self) -> CodecResult<Self::Out>;
}

pub struct ProcessorStaticNode<P: AudioProcessor> {
    p: P,
}

impl<P: AudioProcessor> ProcessorStaticNode<P> {
    pub fn new(p: P) -> Self {
        Self { p }
    }
}

impl<P: AudioProcessor> StaticNode for ProcessorStaticNode<P> {
    type In = AudioFrame;
    type Out = AudioFrame;

    fn name(&self) -> &'static str {
        self.p.name()
    }

    fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
        match input {
            None => self.p.send_frame(None),
            Some(f) => self.p.send_frame(Some(&f as &dyn AudioFrameView)),
        }
    }

    fn pull(&mut self) -> CodecResult<Self::Out> {
        self.p.receive_frame()
    }
}

pub struct EncoderStaticNode<E: AudioEncoder> {
    e: E,
}

impl<E: AudioEncoder> EncoderStaticNode<E> {
    pub fn new(e: E) -> Self {
        Self { e }
    }
}

impl<E: AudioEncoder> StaticNode for EncoderStaticNode<E> {
    type In = AudioFrame;
    type Out = CodecPacket;

    fn name(&self) -> &'static str {
        self.e.name()
    }

    fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
        match input {
            None => self.e.send_frame(None),
            Some(f) => self.e.send_frame(Some(&f as &dyn AudioFrameView)),
        }
    }

    fn pull(&mut self) -> CodecResult<Self::Out> {
        self.e.receive_packet()
    }
}

pub struct DecoderStaticNode<D: AudioDecoder> {
    d: D,
}

impl<D: AudioDecoder> DecoderStaticNode<D> {
    pub fn new(d: D) -> Self {
        Self { d }
    }
}

impl<D: AudioDecoder> StaticNode for DecoderStaticNode<D> {
    type In = CodecPacket;
    type Out = AudioFrame;

    fn name(&self) -> &'static str {
        self.d.name()
    }

    fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
        self.d.send_packet(input)
    }

    fn pull(&mut self) -> CodecResult<Self::Out> {
        self.d.receive_frame()
    }
}

/// Identity 静态节点：把输入原样 move 到输出（零拷贝，不做任何处理）。
///
/// 适合用于：
/// - 静态 pipeline 中“占位/透传”一段，便于后续插入 processor/filter。
/// - 统一使用 `StaticNode` 的 `push/pull + Again/Eof + flush(None)` 驱动语义。
pub struct IdentityStaticNode<T> {
    q: VecDeque<T>,
    flushed: bool,
}

impl<T> IdentityStaticNode<T> {
    pub fn new() -> Self {
        Self {
            q: VecDeque::new(),
            flushed: false,
        }
    }
}

impl<T> Default for IdentityStaticNode<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StaticNode for IdentityStaticNode<T> {
    type In = T;
    type Out = T;

    fn name(&self) -> &'static str {
        "identity"
    }

    fn push(&mut self, input: Option<Self::In>) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidState("already flushed"));
        }
        match input {
            None => {
                self.flushed = true;
                Ok(())
            }
            Some(v) => {
                self.q.push_back(v);
                Ok(())
            }
        }
    }

    fn pull(&mut self) -> CodecResult<Self::Out> {
        if let Some(v) = self.q.pop_front() {
            return Ok(v);
        }
        if self.flushed {
            return Err(CodecError::Eof);
        }
        Err(CodecError::Again)
    }
}

/// 一个固定 3 段的静态 pipeline（Processor -> Encoder -> Decoder）。
pub struct Pipeline3<N1, N2, N3>
where
    N1: StaticNode,
    N2: StaticNode<In = N1::Out>,
    N3: StaticNode<In = N2::Out>,
{
    pub n1: N1,
    pub n2: N2,
    pub n3: N3,
    done1: bool,
    done2: bool,
}

impl<N1, N2, N3> Pipeline3<N1, N2, N3>
where
    N1: StaticNode,
    N2: StaticNode<In = N1::Out>,
    N3: StaticNode<In = N2::Out>,
{
    pub fn new(n1: N1, n2: N2, n3: N3) -> Self {
        Self {
            n1,
            n2,
            n3,
            done1: false,
            done2: false,
        }
    }

    /// 推入一个输入（或 flush），并尽可能把数据跑到末端，返回末端所有可用输出。
    pub fn push_and_drain(&mut self, input: Option<N1::In>) -> CodecResult<Vec<N3::Out>> {
        self.n1.push(input)?;
        self.drain_all()
    }

    pub fn drain_all(&mut self) -> CodecResult<Vec<N3::Out>> {
        // n1 -> n2
        if !self.done1 {
            loop {
                match self.n1.pull() {
                    Ok(v) => self.n2.push(Some(v))?,
                    Err(CodecError::Again) => break,
                    Err(CodecError::Eof) => {
                        self.done1 = true;
                        self.n2.push(None)?;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        // n2 -> n3
        if !self.done2 {
            loop {
                match self.n2.pull() {
                    Ok(v) => self.n3.push(Some(v))?,
                    Err(CodecError::Again) => break,
                    Err(CodecError::Eof) => {
                        self.done2 = true;
                        self.n3.push(None)?;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        // drain n3
        let mut outs = Vec::new();
        loop {
            match self.n3.pull() {
                Ok(v) => outs.push(v),
                Err(CodecError::Again) | Err(CodecError::Eof) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(outs)
    }
}


