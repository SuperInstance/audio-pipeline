//! Voice activity detection

use crate::error::Result;
use std::time::Duration;

pub mod silero;

/// VAD result
#[derive(Debug, Clone)]
pub struct VADResult {
    /// True if speech detected
    pub is_speech: bool,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
    /// Frame timestamp
    pub timestamp: Duration,
}

/// Voice activity detector trait
pub trait VADDetector: Send + Sync {
    /// Detect voice activity in audio frame
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples (16kHz, f32 format)
    ///
    /// # Returns
    ///
    /// VAD result with speech detection and confidence
    fn detect(&self, audio: &[f32]) -> Result<VADResult>;

    /// Reset internal state (for streaming VAD)
    fn reset(&self) -> Result<()>;
}

/// VAD detector builder
pub struct VADDetectorBuilder {
    model: Option<String>,
    threshold: Option<f32>,
}

impl VADDetectorBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            threshold: None,
        }
    }

    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Result<Self> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(AudioPipelineError::ConfigError(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        self.threshold = Some(threshold);
        Ok(self)
    }

    pub fn build(self) -> Result<Box<dyn VADDetector>> {
        let model = self.model.as_deref().unwrap_or("silero");
        let threshold = self.threshold.unwrap_or(0.7);

        match model {
            "silero" => Ok(Box::new(silero::SileroVAD::new(threshold)?)),
            _ => Err(AudioPipelineError::ModelNotFound(format!(
                "Unknown VAD model: {}",
                model
            ))),
        }
    }
}

impl Default for VADDetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create Silero VAD detector (default)
///
/// Silero VAD: <1ms inference, 99.5% accuracy
pub fn silero() -> Result<Box<dyn VADDetector>> {
    VADDetectorBuilder::new().model("silero").build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_builder() {
        let vad = VADDetectorBuilder::new()
            .model("silero")
            .threshold(0.8)
            .build();

        assert!(vad.is_ok());
    }

    #[test]
    fn test_invalid_threshold() {
        let vad = VADDetectorBuilder::new().threshold(1.5);
        assert!(vad.is_err());
    }
}
