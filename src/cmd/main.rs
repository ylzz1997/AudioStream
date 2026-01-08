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
    eprintln!("支持：.wav  .mp3  .adts/.aac  .opus(Ogg Opus) ");
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
    use audiostream::common::io::{AudioReader, AudioWriter};
    use audiostream::common::io::file::WavWriterConfig;
    use audiostream::common::audio::audio::AudioFrameView;
    use audiostream::codec::error::CodecError;
    use audiostream::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
    use audiostream::pipeline::node::dynamic_node_interface::IdentityNode;
    use audiostream::pipeline::node::node_interface::{AsyncPipeline, NodeBuffer, NodeBufferKind};

    let in_ext = ext_lower(input).ok_or("missing input extension")?;
    let out_ext = ext_lower(output).ok_or("missing output extension")?;
    if in_ext != "wav" || out_ext != "wav" {
        return Err("no-ffmpeg build supports only wav->wav".into());
    }
    let mut r = AudioFileReader::open(input, AudioFileReadConfig::Wav).map_err(|e| format!("{e:?}"))?;
    let first = r.next_frame().map_err(|e| format!("{e:?}"))?.ok_or("empty input")?;
    let fmt = first.format();
    let mut w = AudioFileWriter::create(output, AudioFileWriteConfig::Wav(WavWriterConfig::pcm16le(fmt.sample_rate, fmt.channels())))
        .map_err(|e| format!("{e:?}"))?;

    // 统一走异步 pipeline：这里无需处理时，用 identity 节点零拷贝搬运。
    let nodes = vec![Box::new(IdentityNode::new(NodeBufferKind::Pcm)) as Box<dyn audiostream::pipeline::node::dynamic_node_interface::DynNode>];
    let mut p = AsyncDynPipeline::new(nodes).map_err(|e| format!("{e:?}"))?;

    // first
    p.push_frame(NodeBuffer::Pcm(first)).map_err(|e| format!("{e:?}"))?;
    loop {
        match p.try_get_frame() {
            Ok(NodeBuffer::Pcm(of)) => w.write_frame(&of as &dyn AudioFrameView).map_err(|e| format!("{e:?}"))?,
            Ok(_) => return Err("unexpected node buffer kind".into()),
            Err(CodecError::Again) => break,
            Err(e) => return Err(format!("{e:?}")),
        }
    }

    // rest
    while let Some(f) = r.next_frame().map_err(|e| format!("{e:?}"))? {
        p.push_frame(NodeBuffer::Pcm(f)).map_err(|e| format!("{e:?}"))?;
        loop {
            match p.try_get_frame() {
                Ok(NodeBuffer::Pcm(of)) => w.write_frame(&of as &dyn AudioFrameView).map_err(|e| format!("{e:?}"))?,
                Ok(_) => return Err("unexpected node buffer kind".into()),
                Err(CodecError::Again) => break,
                Err(e) => return Err(format!("{e:?}")),
            }
        }
    }

    // flush
    p.flush().map_err(|e| format!("{e:?}"))?;
    loop {
        match p.get_frame().await {
            Ok(NodeBuffer::Pcm(of)) => w.write_frame(&of as &dyn AudioFrameView).map_err(|e| format!("{e:?}"))?,
            Ok(_) => return Err("unexpected node buffer kind".into()),
            Err(CodecError::Eof) => break,
            Err(e) => return Err(format!("{e:?}")),
        }
    }
    w.finalize().map_err(|e| format!("{e:?}"))?;
    Ok(())
}

#[cfg(feature = "ffmpeg")]
async fn transcode_ffmpeg(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    use audiostream::codec::encoder::aac_encoder::AacEncoderConfig;
    use audiostream::codec::encoder::opus_encoder::OpusEncoderConfig;
    use audiostream::codec::processor::resample_processor::ResampleProcessor;
    use audiostream::codec::error::CodecError;
    use audiostream::common::io::file::{
        AudioFileReadConfig, AudioFileReader, AudioFileWriteConfig, AudioFileWriter,
    };
    use audiostream::common::io::{AudioReader, AudioWriter};
    use audiostream::common::io::file::{Mp3WriterConfig, WavWriterConfig};
    use audiostream::common::audio::audio::AudioFrameView;
    use audiostream::common::audio::audio::{AudioFormat, SampleFormat};
    use audiostream::pipeline::node::async_dynamic_node_interface::AsyncDynPipeline;
    use audiostream::pipeline::node::dynamic_node_interface::{IdentityNode, ProcessorNode, DynNode};
    use audiostream::pipeline::node::node_interface::{AsyncPipeline, NodeBuffer, NodeBufferKind};

    let in_ext = ext_lower(input).ok_or("missing input extension")?;
    let out_ext = ext_lower(output).ok_or("missing output extension")?;

    // ---- open reader ----
    let in_cfg = match in_ext.as_str() {
        "wav" => AudioFileReadConfig::Wav,
        "mp3" => AudioFileReadConfig::Mp3,
        "aac" | "adts" => AudioFileReadConfig::AacAdts,
        "opus" => AudioFileReadConfig::OpusOgg,
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

            let mut proc = ResampleProcessor::new_auto(fmt, target_fmt)?;
            // Opus 常用 20ms@48k => 960 samples；并在 flush 时 pad 到 960
            proc.set_output_chunker(Some(960), true)?;
            resampler = Some(proc);

            AudioFileWriteConfig::OpusOgg(OpusEncoderConfig {
                input_format: target_fmt,
                bitrate: Some(96_000),
            })
        }
        "wav" => AudioFileWriteConfig::Wav(WavWriterConfig::pcm16le(fmt.sample_rate, fmt.channels())),
        "mp3" => AudioFileWriteConfig::Mp3(Mp3WriterConfig::new(fmt)),
        "aac" | "adts" => AudioFileWriteConfig::AacAdts(AacEncoderConfig { input_format: fmt, bitrate: Some(128_000) }),
        _ => return Err(format!("unsupported output extension: {out_ext}").into()),
    };
    let mut w = AudioFileWriter::create(output, out_cfg)?;

    // 构建异步 pipeline：
    // - opus：插入 ResampleProcessor
    // - 其它：走 identity（零拷贝搬运），但依旧统一遵循 pipeline 驱动逻辑
    let mut nodes: Vec<Box<dyn DynNode>> = Vec::new();
    if let Some(proc) = resampler {
        nodes.push(Box::new(ProcessorNode::new(proc)));
    } else {
        nodes.push(Box::new(IdentityNode::new(NodeBufferKind::Pcm)));
    }
    let mut p = AsyncDynPipeline::new(nodes)?;

    // first
    p.push_frame(NodeBuffer::Pcm(first))?;
    loop {
        match p.try_get_frame() {
            Ok(NodeBuffer::Pcm(of)) => w.write_frame(&of as &dyn AudioFrameView)?,
            Ok(_) => return Err("unexpected node buffer kind".into()),
            Err(CodecError::Again) => break,
            Err(e) => return Err(Box::new(e)),
        }
    }

    // rest
    while let Some(f) = r.next_frame()? {
        p.push_frame(NodeBuffer::Pcm(f))?;
        loop {
            match p.try_get_frame() {
                Ok(NodeBuffer::Pcm(of)) => w.write_frame(&of as &dyn AudioFrameView)?,
                Ok(_) => return Err("unexpected node buffer kind".into()),
                Err(CodecError::Again) => break,
                Err(e) => return Err(Box::new(e)),
            }
        }
    }

    // flush pipeline
    p.flush()?;
    loop {
        match p.get_frame().await {
            Ok(NodeBuffer::Pcm(of)) => w.write_frame(&of as &dyn AudioFrameView)?,
            Ok(_) => return Err("unexpected node buffer kind".into()),
            Err(CodecError::Eof) => break,
            Err(e) => return Err(Box::new(e)),
        }
    }

    w.finalize()?;
    Ok(())
}
