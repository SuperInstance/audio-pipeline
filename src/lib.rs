//! audio-pipeline: Real-time audio processing for interruption detection
//!
//! This library provides Voice Activity Detection (VAD), Automatic Speech Recognition (ASR),
//! and sentiment analysis from audio streams, enabling <5ms interruption detection for
//! the equilibrium-tokens architecture.
//!
//! # Performance
//!
//! - **VAD Latency**: <1ms (Silero VAD on CPU)
//! - **ASR Latency**: <100ms end-to-end (CAIMAN-ASR streaming)
//! - **Sentiment Analysis**: <5ms with GPU acceleration
//! - **Throughput**: >100K audio frames/second
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use audio_pipeline::{AudioStream, VADDetector, Hz, SampleFormat};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize VAD (Silero: <1ms)
//!     let vad = VADDetector::silero()?;
//!
//!     // Initialize audio stream (16kHz, f32)
//!     let mut audio_stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
//!     audio_stream.start().await?;
//!
//!     // Process audio frames
//!     loop {
//!         let frame = audio_stream.next_frame().await?;
//!         let vad_result = vad.detect(&frame)?;
//!
//!         if vad_result.is_speech {
//!             println!("Speech detected!");
//!         }
//!     }
//! }
//! ```

pub mod audio;
pub mod asr;
pub mod gpu;
pub mod pipeline;
pub mod sentiment;
pub mod vad;

// Re-export core types
pub use audio::{AudioStream, Hz, SampleFormat};
pub use asr::{ASREngine, ASRResult};
pub use pipeline::{AudioPipeline, PipelineResult};
pub use sentiment::{SentimentAnalyzer, VADScores};
pub use vad::{VADDetector, VADResult};

pub mod error;
pub use error::{AudioPipelineError, Result};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default sample rate for speech processing (16kHz)
///
/// Based on the Nyquist-Shannon sampling theorem:
/// To capture frequency f, sample rate must be > 2f
/// Human speech: 80 Hz - 8 kHz
/// Sample rate: 16 kHz (2 × 8 kHz)
pub const DEFAULT_SAMPLE_RATE: Hz = Hz(16000);

/// Default frame duration (32ms)
///
/// 32ms at 16kHz = 512 samples
/// Optimal balance for real-time speech processing
pub const DEFAULT_FRAME_DURATION_MS: u64 = 32;

/// Default frame size (512 samples)
///
/// 16kHz × 32ms = 512 samples
pub const DEFAULT_FRAME_SIZE: usize = 512;

/// Default ring buffer size (1 second)
///
/// 16kHz × 1s = 16000 samples
pub const DEFAULT_BUFFER_SIZE: usize = 16000;
