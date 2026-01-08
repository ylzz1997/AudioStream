#[cfg(not(feature = "ffmpeg"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 {
        eprintln!("当前未启用 ffmpeg feature：只支持 WAV->WAV（PCM）测试。");
        eprintln!("如需 AAC/MP3：cargo run --features ffmpeg -- <in> <out>");
        // 仍然允许 wav->wav 走纯 Rust 路径
        if let Err(e) = transcode_no_ffmpeg(&args[1], &args[2]) {
            eprintln!("transcode failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    println!("用法：audiostream <in> <out>");
    println!("例如：cargo run --features ffmpeg -- input.wav out.mp3");
}

#[cfg(feature = "ffmpeg")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 {
        transcode_ffmpeg(&args[1], &args[2])?;
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
fn transcode_no_ffmpeg(input: &str, output: &str) -> Result<(), String> {
    use audiostream::common::io::file::{AudioFileReadConfig, AudioFileReader, AudioFileWriteConfig, AudioFileWriter};
    use audiostream::common::io::{AudioReader, AudioWriter};
    use audiostream::common::io::file::WavWriterConfig;
    use audiostream::common::audio::audio::AudioFrameView;

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
    w.write_frame(&first).map_err(|e| format!("{e:?}"))?;
    while let Some(f) = r.next_frame().map_err(|e| format!("{e:?}"))? {
        w.write_frame(&f).map_err(|e| format!("{e:?}"))?;
    }
    w.finalize().map_err(|e| format!("{e:?}"))?;
    Ok(())
}

#[cfg(feature = "ffmpeg")]
fn transcode_ffmpeg(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    use audiostream::codec::encoder::aac_encoder::AacEncoderConfig;
    use audiostream::codec::encoder::opus_encoder::OpusEncoderConfig;
    use audiostream::codec::processor::resample_processor::ResampleProcessor;
    use audiostream::codec::error::CodecError;
    use audiostream::codec::processor::processor_interface::AudioProcessor;
    use audiostream::common::io::file::{
        AudioFileReadConfig, AudioFileReader, AudioFileWriteConfig, AudioFileWriter,
    };
    use audiostream::common::io::{AudioReader, AudioWriter};
    use audiostream::common::io::file::{Mp3WriterConfig, WavWriterConfig};
    use audiostream::common::audio::audio::AudioFrameView;
    use audiostream::common::audio::audio::{AudioFormat, SampleFormat};

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

    // 共享写入逻辑：可选经过 ResampleProcessor（opus），否则直接写入（wav/mp3/aac）
    let mut write_via_resampler = |f: &audiostream::common::audio::audio::AudioFrame| -> Result<(), Box<dyn std::error::Error>> {
        let Some(proc) = resampler.as_mut() else {
            w.write_frame(f as &dyn AudioFrameView)?;
            return Ok(());
        };

        proc.send_frame(Some(f as &dyn AudioFrameView))?;
        loop {
            match proc.receive_frame() {
                Ok(of) => w.write_frame(&of as &dyn AudioFrameView)?,
                Err(CodecError::Again) => break,
                Err(e) => return Err(Box::new(e)),
            }
        }
        Ok(())
    };

    // first
    write_via_resampler(&first)?;
    // rest
    while let Some(f) = r.next_frame()? {
        write_via_resampler(&f)?;
    }
    // flush resampler if enabled
    if let Some(proc) = resampler.as_mut() {
        proc.send_frame(None)?;
        loop {
            match proc.receive_frame() {
                Ok(of) => w.write_frame(&of as &dyn AudioFrameView)?,
                Err(CodecError::Eof) => break,
                Err(CodecError::Again) => break,
                Err(e) => return Err(Box::new(e)),
            }
        }
    }

    w.finalize()?;
    Ok(())
}
