//! Sentiment analysis from audio

use crate::error::Result;
use std::time::Duration;

pub mod gpu;

/// VAD sentiment scores (Valence-Arousal-Dominance)
#[derive(Debug, Clone)]
pub struct VADScores {
    /// Valence (positive/negative) [0.0, 1.0]
    pub valence: f32,
    /// Arousal (calm/excited) [0.0, 1.0]
    pub arousal: f32,
    /// Dominance (weak/strong) [0.0, 1.0]
    pub dominance: f32,
}

/// Sentiment analyzer trait
pub trait SentimentAnalyzer: Send + Sync {
    /// Analyze sentiment from audio frame
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples (16kHz, f32 format)
    ///
    /// # Returns
    ///
    /// VAD sentiment scores
    fn analyze(&self, audio: &[f32]) -> Result<VADScores>;

    /// Batch analysis for efficiency
    fn analyze_batch(&self, audio_batch: &[Vec<f32>]) -> Result<Vec<VADScores>> {
        // Default implementation: sequential analysis
        audio_batch
            .iter()
            .map(|audio| self.analyze(audio))
            .collect()
    }
}

/// Create GPU-accelerated sentiment analyzer (default)
///
/// GPU acceleration: <5ms latency with CUDA Graph
pub fn gpu_accelerated() -> Result<Box<dyn SentimentAnalyzer>> {
    Ok(Box::new(gpu::GPUSentimentAnalyzer::new()?))
}
