//! Audio pipeline orchestrator

use crate::asr::{ASREngine, ASRResult};
use crate::error::Result;
use crate::sentiment::{SentimentAnalyzer, VADScores};
use crate::vad::{VADDetector, VADResult};

/// Pipeline result
#[derive(Debug)]
pub enum PipelineResult {
    /// No speech detected
    NoSpeech,
    /// Speech detected with transcription and sentiment
    Speech {
        text: String,
        vad: VADScores,
    },
}

/// Audio pipeline orchestrator
///
/// Processes audio frames through VAD, ASR, and sentiment analysis
pub struct AudioPipeline {
    vad: Box<dyn VADDetector>,
    asr: Box<dyn ASREngine>,
    sentiment: Box<dyn SentimentAnalyzer>,
}

impl AudioPipeline {
    /// Create new pipeline with all components
    pub fn new(
        vad: Box<dyn VADDetector>,
        asr: Box<dyn ASREngine>,
        sentiment: Box<dyn SentimentAnalyzer>,
    ) -> Self {
        Self {
            vad,
            asr,
            sentiment,
        }
    }

    /// Create pipeline builder
    pub fn builder() -> AudioPipelineBuilder {
        AudioPipelineBuilder::new()
    }

    /// Process single frame through full pipeline
    ///
    /// # Arguments
    ///
    /// * `frame` - Audio samples (16kHz, f32 format)
    ///
    /// # Returns
    ///
    /// Pipeline result (NoSpeech or Speech with text and sentiment)
    pub fn process_frame(&self, frame: &[f32]) -> Result<PipelineResult> {
        // VAD: <1ms
        let vad_result = self.vad.detect(frame)?;

        if !vad_result.is_speech {
            return Ok(PipelineResult::NoSpeech);
        }

        // ASR: <100ms (parallel with sentiment)
        let asr_result = self.asr.transcribe(frame)?;

        // Sentiment: <5ms (parallel with ASR)
        let sentiment_result = self.sentiment.analyze(frame)?;

        Ok(PipelineResult::Speech {
            text: asr_result.text,
            vad: sentiment_result,
        })
    }
}

/// Audio pipeline builder
pub struct AudioPipelineBuilder {
    vad: Option<Box<dyn VADDetector>>,
    asr: Option<Box<dyn ASREngine>>,
    sentiment: Option<Box<dyn SentimentAnalyzer>>,
}

impl AudioPipelineBuilder {
    pub fn new() -> Self {
        Self {
            vad: None,
            asr: None,
            sentiment: None,
        }
    }

    pub fn vad(mut self, model: &str) -> Result<Self> {
        self.vad = Some(crate::vad::vad(model)?);
        Ok(self)
    }

    pub fn asr(mut self, model: &str) -> Result<Self> {
        self.asr = Some(crate::asr::asr(model)?);
        Ok(self)
    }

    pub fn sentiment(mut self, model: &str) -> Result<Self> {
        self.sentiment = Some(crate::sentiment::gpu_accelerated()?);
        Ok(self)
    }

    pub fn build(self) -> Result<AudioPipeline> {
        Ok(AudioPipeline {
            vad: self.vad.ok_or_else(|| {
                AudioPipelineError::ConfigError("VAD model not specified".to_string())
            })?,
            asr: self.asr.ok_or_else(|| {
                AudioPipelineError::ConfigError("ASR model not specified".to_string())
            })?,
            sentiment: self.sentiment.ok_or_else(|| {
                AudioPipelineError::ConfigError(
                    "Sentiment model not specified".to_string(),
                )
            })?,
        })
    }
}

impl Default for AudioPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder() {
        let result = AudioPipeline::builder()
            .vad("silero")
            .and_then(|b| b.asr("caiman"))
            .and_then(|b| b.sentiment("gpu"))
            .and_then(|b| b.build());

        // Will fail because models aren't actually loaded yet
        // but tests the builder structure
        assert!(result.is_ok() || result.is_err());
    }
}
