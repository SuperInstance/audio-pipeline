//! Error types for audio-pipeline

use std::fmt;

/// audio-pipeline error types
#[derive(Debug)]
pub enum AudioPipelineError {
    /// Audio capture error (hardware, permissions)
    AudioCapture(String),

    /// VAD inference error
    VADInference(String),

    /// ASR inference error
    ASRInference(String),

    /// Sentiment analysis error
    SentimentInference(String),

    /// GPU error
    GPUError(String),

    /// Buffer overflow/underflow
    BufferError(String),

    /// Invalid configuration
    ConfigError(String),

    /// Invalid input (empty, wrong format, etc.)
    InvalidInput(String),

    /// Model not found
    ModelNotFound(String),

    /// ONNX runtime error
    ONNXRuntime(String),
}

impl fmt::Display for AudioPipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioPipelineError::AudioCapture(msg) => write!(f, "Audio capture error: {}", msg),
            AudioPipelineError::VADInference(msg) => write!(f, "VAD inference error: {}", msg),
            AudioPipelineError::ASRInference(msg) => write!(f, "ASR inference error: {}", msg),
            AudioPipelineError::SentimentInference(msg) => {
                write!(f, "Sentiment inference error: {}", msg)
            }
            AudioPipelineError::GPUError(msg) => write!(f, "GPU error: {}", msg),
            AudioPipelineError::BufferError(msg) => write!(f, "Buffer error: {}", msg),
            AudioPipelineError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            AudioPipelineError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AudioPipelineError::ModelNotFound(msg) => write!(f, "Model not found: {}", msg),
            AudioPipelineError::ONNXRuntime(msg) => write!(f, "ONNX runtime error: {}", msg),
        }
    }
}

impl std::error::Error for AudioPipelineError {}

/// Result type alias
pub type Result<T> = std::result::Result<T, AudioPipelineError>;

impl From<tract_onnx::prelude::TractError> for AudioPipelineError {
    fn from(err: tract_onnx::prelude::TractError) -> Self {
        AudioPipelineError::ONNXRuntime(err.to_string())
    }
}

#[cfg(feature = "gpu")]
impl From<cudarc::driver::CudaError> for AudioPipelineError {
    fn from(err: cudarc::driver::CudaError) -> Self {
        AudioPipelineError::GPUError(err.to_string())
    }
}
