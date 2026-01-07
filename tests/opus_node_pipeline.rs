#![cfg(feature = "ffmpeg")]

use audiostream::codec::decoder::opus_decoder::OpusDecoder;
use audiostream::codec::encoder::opus_encoder::{OpusEncoder, OpusEncoderConfig};
use audiostream::codec::node::dynamic_node_interface::{DecoderNode, DynPipeline, EncoderNode, ProcessorNode};
use audiostream::codec::node::node_interface::NodeBuffer;
use audiostream::codec::node::static_node_interface::{
    DecoderStaticNode, EncoderStaticNode, Pipeline3, ProcessorStaticNode,
};
use audiostream::codec::processor::resample_processor::ResampleProcessor;
use audiostream::common::audio::audio::{AudioFormat, AudioFrame, AudioFrameView, ChannelLayout, Rational, SampleFormat};

fn build_opus_head(channels: u8, input_sample_rate: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(19);
    v.extend_from_slice(b"OpusHead");
    v.push(1);
    v.push(channels);
    v.extend_from_slice(&0u16.to_le_bytes()); // pre_skip
    v.extend_from_slice(&input_sample_rate.to_le_bytes());
    v.extend_from_slice(&0i16.to_le_bytes()); // output_gain
    v.push(0); // mapping_family=0
    v
}

fn f32_plane_to_bytes(p: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(p.len() * 4);
    for &v in p {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

fn make_sine_frame_f32_interleaved_stereo(
    sample_rate: u32,
    start_sample: usize,
    nb_samples: usize,
    freq_hz: f32,
) -> AudioFrame {
    let tb = Rational::new(1, sample_rate as i32);
    let fmt = AudioFormat {
        sample_rate,
        sample_format: SampleFormat::F32 { planar: false },
        channel_layout: ChannelLayout::stereo(),
    };

    let mut interleaved = Vec::with_capacity(nb_samples * 2);
    for i in 0..nb_samples {
        let t = (start_sample + i) as f32 / sample_rate as f32;
        let s = (2.0 * core::f32::consts::PI * freq_hz * t).sin() * 0.2;
        interleaved.push(s); // L
        interleaved.push(s); // R
    }

    let planes = vec![f32_plane_to_bytes(&interleaved)];
    AudioFrame::from_planes(fmt, nb_samples, tb, Some(start_sample as i64), planes).unwrap()
}

#[test]
fn opus_pipeline_dynamic_and_static() {
    let in_fmt = AudioFormat {
        sample_rate: 44_100,
        sample_format: SampleFormat::F32 { planar: false },
        channel_layout: ChannelLayout::stereo(),
    };
    let mid_fmt = AudioFormat {
        sample_rate: 48_000,
        sample_format: SampleFormat::F32 { planar: false },
        channel_layout: ChannelLayout::stereo(),
    };

    // -------- dynamic pipeline --------
    let mut resample = ResampleProcessor::new_auto(in_fmt, mid_fmt).unwrap();
    // Opus 常用 20ms@48k => 960 samples；flush 时补齐尾巴，避免非法帧长
    resample.set_output_chunker(Some(960), true).unwrap();
    let enc = OpusEncoder::new(OpusEncoderConfig {
        input_format: mid_fmt,
        bitrate: Some(64_000),
    })
    .unwrap();
    let extradata = enc
        .extradata()
        .unwrap_or_else(|| build_opus_head(2, 48_000));
    let dec = OpusDecoder::new_with_extradata(&extradata).unwrap();

    let mut pipe = DynPipeline::new(vec![
        Box::new(ProcessorNode::new(resample)),
        Box::new(EncoderNode::new(enc)),
        Box::new(DecoderNode::new(dec)),
    ])
    .unwrap();

    let total_in = 44_100 / 10; // 0.1s
    let chunk = 1024;
    let mut got_samples = 0usize;

    let mut pos = 0usize;
    while pos < total_in {
        let nb = (total_in - pos).min(chunk);
        let f = make_sine_frame_f32_interleaved_stereo(44_100, pos, nb, 440.0);
        let outs = pipe
            .push_and_drain(Some(NodeBuffer::Pcm(f)))
            .unwrap();
        for o in outs {
            if let NodeBuffer::Pcm(of) = o {
                assert_eq!(of.format().sample_rate, 48_000);
                assert_eq!(of.format().channels(), 2);
                got_samples += of.nb_samples();
            }
        }
        pos += nb;
    }

    // flush 并尽量 drain
    let mut outs = pipe.push_and_drain(None).unwrap();
    for _ in 0..8 {
        let more = pipe.drain_all().unwrap();
        if more.is_empty() {
            break;
        }
        outs.extend(more);
    }
    for o in outs {
        if let NodeBuffer::Pcm(of) = o {
            assert_eq!(of.format().sample_rate, 48_000);
            assert_eq!(of.format().channels(), 2);
            got_samples += of.nb_samples();
        }
    }
    assert!(got_samples > 0);

    // -------- static pipeline --------
    let mut resample = ResampleProcessor::new_auto(in_fmt, mid_fmt).unwrap();
    resample.set_output_chunker(Some(960), true).unwrap();
    let enc = OpusEncoder::new(OpusEncoderConfig {
        input_format: mid_fmt,
        bitrate: Some(64_000),
    })
    .unwrap();
    let extradata = enc
        .extradata()
        .unwrap_or_else(|| build_opus_head(2, 48_000));
    let dec = OpusDecoder::new_with_extradata(&extradata).unwrap();

    let mut pipe = Pipeline3::new(
        ProcessorStaticNode::new(resample),
        EncoderStaticNode::new(enc),
        DecoderStaticNode::new(dec),
    );

    let mut got_samples2 = 0usize;
    let mut pos = 0usize;
    while pos < total_in {
        let nb = (total_in - pos).min(chunk);
        let f = make_sine_frame_f32_interleaved_stereo(44_100, pos, nb, 440.0);
        let outs = pipe.push_and_drain(Some(f)).unwrap();
        for of in outs {
            assert_eq!(of.format().sample_rate, 48_000);
            assert_eq!(of.format().channels(), 2);
            got_samples2 += of.nb_samples();
        }
        pos += nb;
    }
    let mut outs = pipe.push_and_drain(None).unwrap();
    for _ in 0..8 {
        let more = pipe.drain_all().unwrap();
        if more.is_empty() {
            break;
        }
        outs.extend(more);
    }
    for of in outs {
        assert_eq!(of.format().sample_rate, 48_000);
        assert_eq!(of.format().channels(), 2);
        got_samples2 += of.nb_samples();
    }
    assert!(got_samples2 > 0);
}


