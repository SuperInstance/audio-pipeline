//! Automatic speech recognition

use crate::error::Result;
use std::time::Duration;

pub mod caiman;

/// ASR result
#[derive(Debug, Clone)]
pub struct ASRResult {
    /// Transcribed text
    pub text: String,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
    /// Timestamp
    pub timestamp: Duration,
}

/// Automatic speech recognition engine trait
pub trait ASREngine: Send + Sync {
    /// Transcribe audio frame (blocking)
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples (16kHz, f32 format)
    ///
    /// # Returns
    ///
    /// ASR result with transcribed text and confidence
    fn transcribe(&self, audio: &[f32]) -> Result<ASRResult>;

    /// Check if frame contains speech (VAD pre-filter)
    fn has_speech(&self, audio: &[f32]) -> bool {
        // Default implementation: check if audio has energy
        audio.iter().any(|&s| s.abs() > 0.01)
    }
}

/// ASR engine builder
pub struct ASREngineBuilder {
    model: Option<String>,
}

impl ASREngineBuilder {
    pub fn new() -> Self {
        Self { model: None }
    }

    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    pub fn build(self) -> Result<Box<dyn ASREngine>> {
        let model = self.model.as_deref().unwrap_or("caiman");

        match model {
            "caiman" => Ok(Box::new(caiman::CAIMANASR::new()?)),
            _ => Err(AudioPipelineError::ModelNotFound(format!(
                "Unknown ASR model: {}",
                model
            ))),
        }
    }
}

impl Default for ASREngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create CAIMAN-ASR engine (default)
///
/// CAIMAN-ASR: <100ms latency, WER <10%, streaming capable
pub fn caiman() -> Result<Box<dyn ASREngine>> {
    ASREngineBuilder::new().model("caiman").build()
}
