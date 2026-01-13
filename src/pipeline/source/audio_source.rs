use crate::common::audio::audio::AudioFrame;
use crate::common::io::io::AudioReader;
use crate::pipeline::node::node_interface::NodeBuffer;
use crate::runner::error::RunnerResult;
use async_trait::async_trait;
use std::collections::VecDeque;

/// 音频源（只出）：Runner 会不断拉取直到返回 `Ok(None)` 表示结束。
pub trait AudioSource {
    type Out;

    fn name(&self) -> &'static str {
        "audio-source"
    }

    fn pull(&mut self) -> RunnerResult<Option<Self::Out>>;
}

/// 异步音频源（只出）：允许在 tokio runtime 中 `await` 拉取输入。
#[async_trait]
pub trait AsyncAudioSource: Send {
    type Out: Send + 'static;

    fn name(&self) -> &'static str {
        "async-audio-source"
    }

    async fn pull(&mut self) -> RunnerResult<Option<Self::Out>>;
}

/// 同步 source 自动适配为 async source（在 async 上下文里直接同步执行）。
#[async_trait]
impl<T> AsyncAudioSource for T
where
    T: AudioSource + Send,
    T::Out: Send + 'static,
{
    type Out = T::Out;

    fn name(&self) -> &'static str {
        AudioSource::name(self)
    }

    async fn pull(&mut self) -> RunnerResult<Option<Self::Out>> {
        AudioSource::pull(self)
    }
}

/// 让现有的 `AudioReader` 自动成为 `AudioSource<Out = AudioFrame>`。
impl<T> AudioSource for T
where
    T: AudioReader,
{
    type Out = AudioFrame;

    fn name(&self) -> &'static str {
        "audio-reader"
    }

    fn pull(&mut self) -> RunnerResult<Option<Self::Out>> {
        Ok(self.next_frame()?)
    }
}

/// 把 `AudioSource<Out = AudioFrame>` 适配为 `AudioSource<Out = NodeBuffer>`（PCM）。
pub struct PcmSource<S> {
    inner: S,
}

impl<S> PcmSource<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S> AudioSource for PcmSource<S>
where
    S: AudioSource<Out = AudioFrame>,
{
    type Out = NodeBuffer;

    fn name(&self) -> &'static str {
        "pcm-source"
    }

    fn pull(&mut self) -> RunnerResult<Option<Self::Out>> {
        Ok(self.inner.pull()?.map(NodeBuffer::Pcm))
    }
}

/// 在一个 Source 前面追加若干条预读数据（常用于“为了拿 format 先读一帧”）。
pub struct PrependSource<S>
where
    S: AudioSource,
{
    q: VecDeque<S::Out>,
    inner: S,
}

impl<S> PrependSource<S>
where
    S: AudioSource,
{
    pub fn new(inner: S, items: impl IntoIterator<Item = S::Out>) -> Self {
        Self {
            q: items.into_iter().collect(),
            inner,
        }
    }

    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S> AudioSource for PrependSource<S>
where
    S: AudioSource,
{
    type Out = S::Out;

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn pull(&mut self) -> RunnerResult<Option<Self::Out>> {
        if let Some(v) = self.q.pop_front() {
            return Ok(Some(v));
        }
        self.inner.pull()
    }
}


