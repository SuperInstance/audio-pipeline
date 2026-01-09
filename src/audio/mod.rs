//! Audio stream capture and processing

pub mod buffer;
pub mod capture;

use std::time::Duration;

/// Audio sample rate in Hertz
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hz(pub u32);

/// Audio sample format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 32-bit float (range: [-1.0, 1.0])
    F32,
    /// 16-bit integer (range: [-32768, 32767])
    I16,
}

/// Continuous audio stream
///
/// Captures audio from microphone or API at specified sample rate and format.
/// Uses ring buffer for continuous processing.
pub struct AudioStream {
    sample_rate: Hz,
    format: SampleFormat,
    chunk_size: usize,
    buffer: buffer::AudioRingBuffer,
}

impl AudioStream {
    /// Create new audio stream with default settings
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (default: 16000 for speech)
    /// * `format` - Audio sample format (F32 or I16)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use audio_pipeline::{AudioStream, Hz, SampleFormat};
    ///
    /// let stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
    /// ```
    pub fn new(sample_rate: Hz, format: SampleFormat) -> Result<Self> {
        Ok(Self {
            sample_rate,
            format,
            chunk_size: crate::DEFAULT_FRAME_SIZE,
            buffer: buffer::AudioRingBuffer::new(crate::DEFAULT_BUFFER_SIZE)?,
        })
    }

    /// Create audio stream with builder
    pub fn builder() -> AudioStreamBuilder {
        AudioStreamBuilder::new()
    }

    /// Start audio capture (async)
    pub async fn start(&mut self) -> Result<()> {
        capture::start_capture(self).await
    }

    /// Get next audio frame
    ///
    /// Returns chunk_size samples (default: 512 samples, 32ms at 16kHz)
    pub async fn next_frame(&mut self) -> Result<Vec<f32>> {
        self.buffer.read_frame().await
    }

    /// Stop audio capture
    pub async fn stop(&mut self) -> Result<()> {
        capture::stop_capture().await
    }
}

/// Audio stream builder
pub struct AudioStreamBuilder {
    sample_rate: Option<Hz>,
    format: Option<SampleFormat>,
    chunk_size: Option<usize>,
    buffer_size: Option<usize>,
}

impl AudioStreamBuilder {
    pub fn new() -> Self {
        Self {
            sample_rate: None,
            format: None,
            chunk_size: None,
            buffer_size: None,
        }
    }

    pub fn sample_rate(mut self, rate: Hz) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    pub fn format(mut self, format: SampleFormat) -> Self {
        self.format = Some(format);
        self
    }

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    pub fn buffer_size(mut self, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(AudioPipelineError::ConfigError(
                "Buffer size cannot be zero".to_string(),
            ));
        }
        self.buffer_size = Some(size);
        Ok(self)
    }

    pub fn build(self) -> Result<AudioStream> {
        Ok(AudioStream {
            sample_rate: self.sample_rate.unwrap_or(crate::DEFAULT_SAMPLE_RATE),
            format: self.format.unwrap_or(SampleFormat::F32),
            chunk_size: self.chunk_size.unwrap_or(crate::DEFAULT_FRAME_SIZE),
            buffer: buffer::AudioRingBuffer::new(
                self.buffer_size.unwrap_or(crate::DEFAULT_BUFFER_SIZE),
            )?,
        })
    }
}

impl Default for AudioStreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_stream_creation() {
        let stream = AudioStream::new(Hz(16000), SampleFormat::F32);
        assert!(stream.is_ok());
    }

    #[test]
    fn test_builder() {
        let stream = AudioStream::builder()
            .sample_rate(Hz(16000))
            .format(SampleFormat::F32)
            .chunk_size(512)
            .buffer_size(16000)
            .build();

        assert!(stream.is_ok());
    }
}
