#[cfg(not(feature = "ffmpeg"))]
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 {
        eprintln!("当前未启用 ffmpeg feature：只支持 WAV->WAV（PCM）测试。");
        eprintln!("如需 AAC/MP3：cargo run --features ffmpeg -- <in> <out>");
        // wav->wav 也统一走异步 pipeline（后台线程驱动）。
        if let Err(e) = transcode_no_ffmpeg(&args[1], &args[2]).await {
            eprintln!("transcode failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    println!("用法：audiostream <in> <out>");
    println!("例如：cargo run --features ffmpeg -- input.wav out.mp3");
}

#[cfg(feature = "ffmpeg")]
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 {
        transcode_ffmpeg(&args[1], &args[2]).await?;
        return Ok(());
    }
    eprintln!("用法：audiostream <in> <out>");
    eprintln!("支持：.wav  .mp3  .adts/.aac  .opus(Ogg Opus)  .flac");
    eprintln!("例如：cargo run --features ffmpeg -- input.wav out.mp3");
    Ok(())
}

fn ext_lower(path: &str) -> Option<String> {
    std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
}

#[cfg(not(feature = "ffmpeg"))]
async fn transcode_no_ffmpeg(input: &str, output: &str) -> Result<(), String> {
    use audiostream::common::io::file::{AudioFileReadConfig, AudioFileReader, AudioFileWriteConfig, AudioFileWriter};
    use audiostream::common::io::AudioReader;
    use audiostream::common::audio::audio::AudioFrameView;
    use audiostream::common::io::file::{WavReaderConfig, WavWriterConfig};
    use audiostream::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
    use audiostream::pipeline::node::dynamic_node_interface::IdentityNode;
    use audiostream::pipeline::node::node_interface::NodeBufferKind;
    use audiostream::runner::async_dynamic_runner::AsyncDynRunner;
    use audiostream::runner::async_runner_interface::AsyncRunner;
    use audiostream::runner::audio_sink::PcmSink;
    use audiostream::runner::audio_source::{PcmSource, PrependSource};

    let in_ext = ext_lower(input).ok_or("missing input extension")?;
    let out_ext = ext_lower(output).ok_or("missing output extension")?;
    if in_ext != "wav" || out_ext != "wav" {
        return Err("no-ffmpeg build supports only wav->wav".into());
    }
    let mut r = AudioFileReader::open(input, AudioFileReadConfig::Wav(WavReaderConfig::default())).map_err(|e| format!("{e:?}"))?;
    let first = r.next_frame().map_err(|e| format!("{e:?}"))?.ok_or("empty input")?;
    let fmt = first.format();
    let mut w = AudioFileWriter::create(output, AudioFileWriteConfig::Wav(WavWriterConfig::pcm16le(fmt)))
        .map_err(|e| format!("{e:?}"))?;

    // 统一走异步 pipeline：这里无需处理时，用 identity 节点零拷贝搬运。
    let nodes = vec![Box::new(IdentityNode::new(NodeBufferKind::Pcm)) as Box<dyn audiostream::pipeline::node::dynamic_node_interface::DynNode>];
    let p = AsyncDynPipeline::new(nodes).map_err(|e| format!("{e:?}"))?;

    // 先读一帧用于拿格式：用 PrependSource 把它“塞回”source，再由 runner 统一驱动。
    let source = PcmSource::new(PrependSource::new(r, vec![first]));
    let sink = PcmSink::new(w);
    // 使用 “endpoints 并行驱动” 的异步 Runner（输入/输出各一个任务）
    let mut runner = AsyncDynRunner::new(source, p, sink);
    runner
        .execute()
        .await
        .map_err(|e| format!("{e}"))?;
    Ok(())
}

#[cfg(feature = "ffmpeg")]
async fn transcode_ffmpeg(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    use audiostream::codec::encoder::aac_encoder::AacEncoderConfig;
    use audiostream::codec::encoder::flac_encoder::FlacEncoderConfig;
    use audiostream::codec::encoder::opus_encoder::OpusEncoderConfig;
    use audiostream::codec::processor::resample_processor::ResampleProcessor;
    use audiostream::common::io::file::{
        AudioFileReadConfig, AudioFileReader, AudioFileWriteConfig, AudioFileWriter,
    };
    use audiostream::common::io::AudioReader;
    use audiostream::common::io::file::{AacAdtsWriterConfig, FlacWriterConfig, Mp3WriterConfig, OpusOggWriterConfig, WavReaderConfig, WavWriterConfig};
    use audiostream::common::audio::audio::{AudioFormat, AudioFrameView, SampleFormat};
    use audiostream::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
    use audiostream::pipeline::node::dynamic_node_interface::{ProcessorNode, DynNode};
    use audiostream::pipeline::node::node_interface::{NodeBufferKind, IdentityNode};
    use audiostream::runner::async_dynamic_runner::AsyncDynRunner;
    use audiostream::runner::async_runner_interface::AsyncRunner;
    use audiostream::runner::audio_sink::PcmSink;
    use audiostream::runner::audio_source::{PcmSource, PrependSource};

    let in_ext = ext_lower(input).ok_or("missing input extension")?;
    let out_ext = ext_lower(output).ok_or("missing output extension")?;

    // ---- open reader ----
    let in_cfg = match in_ext.as_str() {
        "wav" => AudioFileReadConfig::Wav(WavReaderConfig::default()),
        "mp3" => AudioFileReadConfig::Mp3,
        "aac" | "adts" => AudioFileReadConfig::AacAdts,
        "opus" => AudioFileReadConfig::OpusOgg,
        "flac" => AudioFileReadConfig::Flac,
        _ => return Err(format!("unsupported input extension: {in_ext}").into()),
    };
    let mut r = AudioFileReader::open(input, in_cfg)?;
    let first = r.next_frame()?.ok_or("empty input")?;
    let fmt = first.format();

    // ---- write ----
    // Opus 输出：需要 48k + 常见 flt(packed)；这里统一转到 f32 interleaved，并按 960 samples 分帧
    let mut resampler: Option<ResampleProcessor> = None;
    let out_cfg = match out_ext.as_str() {
        "opus" => {
            let target_fmt = AudioFormat {
                sample_rate: 48_000,
                sample_format: SampleFormat::F32 { planar: false },
                channel_layout: fmt.channel_layout,
            };

            let mut proc = ResampleProcessor::new(fmt, target_fmt)?;
            // Opus 常用 20ms@48k => 960 samples；并在 flush 时 pad 到 960
            proc.set_output_chunker(Some(960), true)?;
            resampler = Some(proc);

            AudioFileWriteConfig::OpusOgg(OpusOggWriterConfig {
                encoder: OpusEncoderConfig {
                    input_format: Some(target_fmt),
                    bitrate: Some(96_000),
                },
            })
        }
        "flac" => {
            // FFmpeg FLAC encoder 常见只支持 s16/s32（packed）。我们统一转成 s16 packed，避免输入是 fltp（WAV reader 默认输出）时失败。
            let target_fmt = AudioFormat {
                sample_rate: fmt.sample_rate,
                sample_format: SampleFormat::I16 { planar: false },
                channel_layout: fmt.channel_layout,
            };
            if fmt != target_fmt {
                let proc = ResampleProcessor::new(fmt, target_fmt)?;
                resampler = Some(proc);
            }
            AudioFileWriteConfig::Flac(FlacWriterConfig {
                encoder: FlacEncoderConfig::new(target_fmt),
            })
        }
        "wav" => AudioFileWriteConfig::Wav(WavWriterConfig::pcm16le(fmt)),
        "mp3" => AudioFileWriteConfig::Mp3(Mp3WriterConfig {
            encoder: audiostream::codec::encoder::mp3_encoder::Mp3EncoderConfig::new(fmt),
        }),
        "aac" | "adts" => AudioFileWriteConfig::AacAdts(AacAdtsWriterConfig {
            encoder: AacEncoderConfig {
                input_format: Some(fmt),
                bitrate: Some(128_000),
            },
        }),
        _ => return Err(format!("unsupported output extension: {out_ext}").into()),
    };
    let w = AudioFileWriter::create(output, out_cfg)?;

    let mut nodes: Vec<Box<dyn DynNode>> = Vec::new();
    if let Some(proc) = resampler {
        nodes.push(Box::new(ProcessorNode::new(proc)));
    } else {
        nodes.push(Box::new(IdentityNode::new(NodeBufferKind::Pcm)));
    }
    let p = AsyncDynPipeline::new(nodes)?;

    let source = PcmSource::new(PrependSource::new(r, vec![first]));
    let sink = PcmSink::new(w);
    let mut runner = AsyncDynRunner::new(source, p, sink);
    runner.execute().await.map_err(|e| format!("{e}"))?;
    Ok(())
}
