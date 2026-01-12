use audiostream::codec::processor::identity_processor::IdentityProcessor;
use audiostream::common::audio::audio::{
    AudioFormat, AudioFrame, AudioFrameView, AudioFrameViewMut, ChannelLayout, Rational, SampleFormat,
};
use audiostream::common::io::io::{AudioIOResult, AudioWriter};
use audiostream::common::io::LineAudioWriter;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct CollectState {
    frames: Vec<AudioFrame>,
    finalized: bool,
}

struct CollectWriter {
    state: Arc<Mutex<CollectState>>,
}

impl AudioWriter for CollectWriter {
    fn write_frame(&mut self, frame: &dyn AudioFrameView) -> AudioIOResult<()> {
        // 为了测试简单：把 frame copy 成 owned AudioFrame 存起来
        let fmt = frame.format();
        let nb_samples = frame.nb_samples();
        let plane_count = frame.plane_count();
        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(plane_count);
        for i in 0..plane_count {
            planes.push(frame.plane(i).unwrap().to_vec());
        }
        let owned = AudioFrame::from_planes(fmt, nb_samples, frame.time_base(), frame.pts(), planes)
            .map_err(|_| audiostream::common::io::io::AudioIOError::Format("failed to copy frame"))?;
        self.state.lock().unwrap().frames.push(owned);
        Ok(())
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        self.state.lock().unwrap().finalized = true;
        Ok(())
    }
}

fn make_test_frame(nb_samples: usize) -> AudioFrame {
    let fmt = AudioFormat {
        sample_rate: 48000,
        sample_format: SampleFormat::F32 { planar: false },
        channel_layout: ChannelLayout::stereo(),
    };
    let mut f = AudioFrame::new_alloc(fmt, nb_samples).unwrap();
    f.set_time_base(Rational::new(1, 48000)).unwrap();
    f
}

#[test]
fn line_writer_no_processor_passthrough() {
    let state = Arc::new(Mutex::new(CollectState::default()));
    let collect = Box::new(CollectWriter { state: state.clone() });
    let mut lw = LineAudioWriter::new(collect);

    let f1 = make_test_frame(256);
    let f2 = make_test_frame(512);

    lw.write_frame(&f1 as &dyn AudioFrameView).unwrap();
    lw.write_frame(&f2 as &dyn AudioFrameView).unwrap();
    lw.finalize().unwrap();

    let st = state.lock().unwrap();
    assert_eq!(st.frames.len(), 2);
    assert!(st.finalized);
    assert_eq!(st.frames[0].nb_samples(), 256);
    assert_eq!(st.frames[1].nb_samples(), 512);
}

#[test]
fn line_writer_with_identity_processor() {
    let state = Arc::new(Mutex::new(CollectState::default()));
    let collect = Box::new(CollectWriter { state: state.clone() });
    let mut lw =
        LineAudioWriter::with_processors(vec![Box::new(IdentityProcessor::new().unwrap())], collect);

    let f1 = make_test_frame(128);
    let f2 = make_test_frame(128);

    lw.write_frame(&f1 as &dyn AudioFrameView).unwrap();
    lw.write_frame(&f2 as &dyn AudioFrameView).unwrap();
    lw.finalize().unwrap();

    let st = state.lock().unwrap();
    assert_eq!(st.frames.len(), 2);
    assert!(st.finalized);
}

