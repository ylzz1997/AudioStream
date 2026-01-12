use crate::common::io::io::AudioWriter;

/// Extension point for "multi writer" implementations.
pub trait MultiAudioWriter {
    /// Bind (append) a writer.
    fn bind(&mut self, w: Box<dyn AudioWriter + Send>);

    /// Bind multiple writers.
    fn extend(&mut self, ws: impl IntoIterator<Item = Box<dyn AudioWriter + Send>>) {
        for w in ws {
            self.bind(w);
        }
    }

    /// Number of bound writers.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

