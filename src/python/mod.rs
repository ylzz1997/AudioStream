//! Python bindings (PyO3) for streaming encoder/decoder.
mod errors;
mod format;
mod encoder;
mod decoder;
mod processor;
mod io;

pub use crate::python::format::AudioFormat;
pub use crate::python::encoder::{
    AacEncoderConfigPy, Encoder, FlacEncoderConfigPy, Mp3EncoderConfigPy, OpusEncoderConfigPy, WavEncoderConfigPy,
    make_encoder_node,
};
pub use crate::python::decoder::{
    AacDecoderConfigPy, Decoder, FlacDecoderConfigPy, Mp3DecoderConfigPy, OpusDecoderConfigPy, WavDecoderConfigPy,
    make_decoder_node,
};
pub use crate::python::processor::{
    CompressorNodeConfigPy, GainNodeConfigPy, IdentityNodeConfigPy, ProcessorPy, ResampleNodeConfigPy, make_processor_node,
};
pub use crate::python::io::{
    AsyncDynPipelinePy, AsyncDynRunnerPy, AudioFileReaderPy, AudioFileWriterPy, AudioSinkBase, AudioSourceBase,
    AsyncParallelAudioSinkHandlePy, AsyncParallelAudioSinkPy, AsyncPipelineAudioSinkHandlePy, AsyncPipelineAudioSinkPy,
    DynNodePy, LineAudioWriterPy, NodeBase, NodeBufferPy, PacketPy, ParallelAudioWriterPy,
    ReduceConcatPy, ReduceMaxPy, ReduceMeanPy, ReduceMinPy, ReduceProductPy, ReduceSumPy, ReduceXorPy,
    make_async_fork_join_node, make_identity_node, make_python_node, make_tap_node,
};

use pyo3::prelude::*;

#[pymodule]
fn pyaudiostream(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // format
    m.add_class::<AudioFormat>()?;

    // encoder configs + class
    m.add_class::<WavEncoderConfigPy>()?;
    m.add_class::<Mp3EncoderConfigPy>()?;
    m.add_class::<AacEncoderConfigPy>()?;
    m.add_class::<OpusEncoderConfigPy>()?;
    m.add_class::<FlacEncoderConfigPy>()?;
    m.add_class::<Encoder>()?;

    // decoder configs + class
    m.add_class::<WavDecoderConfigPy>()?;
    m.add_class::<Mp3DecoderConfigPy>()?;
    m.add_class::<AacDecoderConfigPy>()?;
    m.add_class::<OpusDecoderConfigPy>()?;
    m.add_class::<FlacDecoderConfigPy>()?;
    m.add_class::<Decoder>()?;

    // processor
    m.add_class::<ProcessorPy>()?;
    m.add_class::<IdentityNodeConfigPy>()?;
    m.add_class::<ResampleNodeConfigPy>()?;
    m.add_class::<GainNodeConfigPy>()?;
    m.add_class::<CompressorNodeConfigPy>()?;

    // io/pipeline
    m.add_class::<PacketPy>()?;
    m.add_class::<NodeBufferPy>()?;
    m.add_class::<DynNodePy>()?;
    // built-in reduce callables
    m.add_class::<ReduceSumPy>()?;
    m.add_class::<ReduceProductPy>()?;
    m.add_class::<ReduceMeanPy>()?;
    m.add_class::<ReduceMaxPy>()?;
    m.add_class::<ReduceMinPy>()?;
    m.add_class::<ReduceConcatPy>()?;
    m.add_class::<ReduceXorPy>()?;
    m.add_class::<NodeBase>()?;
    m.add_class::<AudioSourceBase>()?;
    m.add_class::<AudioSinkBase>()?;
    m.add_class::<AsyncDynPipelinePy>()?;
    m.add_class::<AsyncDynRunnerPy>()?;
    m.add_class::<AudioFileReaderPy>()?;
    m.add_class::<AudioFileWriterPy>()?;
    m.add_class::<ParallelAudioWriterPy>()?;
    m.add_class::<LineAudioWriterPy>()?;
    m.add_class::<AsyncPipelineAudioSinkPy>()?;
    m.add_class::<AsyncPipelineAudioSinkHandlePy>()?;
    m.add_class::<AsyncParallelAudioSinkPy>()?;
    m.add_class::<AsyncParallelAudioSinkHandlePy>()?;

    // functions
    m.add_function(wrap_pyfunction!(io::make_identity_node, m)?)?;
    m.add_function(wrap_pyfunction!(processor::make_processor_node, m)?)?;
    m.add_function(wrap_pyfunction!(encoder::make_encoder_node, m)?)?;
    m.add_function(wrap_pyfunction!(decoder::make_decoder_node, m)?)?;
    m.add_function(wrap_pyfunction!(io::make_python_node, m)?)?;
    m.add_function(wrap_pyfunction!(io::make_tap_node, m)?)?;
    m.add_function(wrap_pyfunction!(io::make_async_fork_join_node, m)?)?;

    // Backward-compatible alias: LineWriter -> LineAudioWriter
    m.setattr("LineWriter", m.getattr("LineAudioWriter")?)?;
    Ok(())
}


