use crate::codec::processor::processor_interface::AudioProcessor;
use crate::common::io::io::AudioWriter;

/// Extension point for "processor chain -> writer" implementations.
///
/// A writer chain is conceptually:
/// `processors (PCM->PCM)* -> writer`
pub trait AudioWriterChain {
    /// Append a processor to the chain (right before the final writer).
    fn push_processor(&mut self, p: Box<dyn AudioProcessor>);

    /// Append multiple processors.
    fn extend_processors(&mut self, ps: impl IntoIterator<Item = Box<dyn AudioProcessor>>) {
        for p in ps {
            self.push_processor(p);
        }
    }

    /// Get the processors list (in-order).
    fn processors(&self) -> &[Box<dyn AudioProcessor>];

    /// Get the processors list (mutable).
    fn processors_mut(&mut self) -> &mut [Box<dyn AudioProcessor>];

    /// Get the final writer (read-only).
    fn writer(&self) -> &dyn AudioWriter;

    /// Get the final writer (mutable).
    fn writer_mut(&mut self) -> &mut dyn AudioWriter;

    /// Number of processors in the chain.
    fn len(&self) -> usize {
        self.processors().len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

