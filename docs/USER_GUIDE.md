# audio-pipeline User Guide

Complete guide to installing, configuring, and using audio-pipeline for real-time audio processing.

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Configuration and Tuning](#configuration-and-tuning)
5. [Integration Examples](#integration-examples)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

---

## Installation

### Prerequisites

**System Requirements**:
- Linux (Ubuntu 20.04+, Debian 11+, Fedora 35+)
- macOS 11+ (Big Sur or later)
- Windows 10+ (Windows Subsystem for Linux recommended)

**Hardware Requirements**:
- CPU: x86_64 or ARM64 with NEON support
- RAM: 4GB minimum, 8GB recommended
- GPU: NVIDIA GPU with CUDA support (optional, for GPU acceleration)

**Audio Hardware**:
- Microphone (built-in or USB)
- Audio capture device (ALSA, PulseAudio, CoreAudio, or WASAPI)

### Rust Installation

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add audio-pipeline dependency
cd your_project
cargo add audio-pipeline
```

**Cargo.toml**:
```toml
[dependencies]
audio-pipeline = "0.1"
tokio = { version = "1.0", features = ["full"] }
```

### Python Installation

```bash
# Install audio-pipeline Python package
pip install audio-pipeline

# Or with GPU support (NVIDIA GPU required)
pip install audio-pipeline[gpu]
```

**Requirements**:
- Python 3.8+
- pip 20.0+

### Verify Installation

**Rust**:
```bash
cargo run --example hello_audio
```

**Python**:
```bash
python -c "import audio_pipeline; print(audio_pipeline.__version__)"
```

---

## Basic Usage

### VAD - Voice Activity Detection

The simplest way to detect voice activity in audio streams.

#### Rust Example

```rust
use audio_pipeline::{VADDetector, AudioStream, Hz, SampleFormat};
use tokio::stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize VAD (Silero: <1ms)
    let vad = VADDetector::silero()?;

    // Initialize audio stream (16kHz, f32)
    let mut audio_stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
    audio_stream.start().await?;

    // Process audio frames
    let mut frame_count = 0;
    loop {
        // Get next audio frame (32ms at 16kHz)
        let frame = audio_stream.next_frame().await?;

        // Detect voice activity
        let vad_result = vad.detect(&frame)?;

        if vad_result.is_speech {
            println!(
                "Frame {}: SPEECH (confidence: {:.2})",
                frame_count, vad_result.confidence
            );
        } else {
            println!("Frame {}: silence", frame_count);
        }

        frame_count += 1;
    }
}
```

#### Python Example

```python
from audio_pipeline import VADDetector, AudioStream
import asyncio

async def main():
    # Initialize VAD (Silero: <1ms)
    vad = VADDetector.silero()

    # Initialize audio stream (16kHz, f32)
    audio = AudioStream(microphone=True, sample_rate=16000)

    # Process audio frames
    frame_count = 0
    async for frame in audio.stream(chunk_size=512):
        # Detect voice activity
        is_speech, confidence = vad.detect(frame)

        if is_speech:
            print(f"Frame {frame_count}: SPEECH (confidence: {confidence:.2f})")
        else:
            print(f"Frame {frame_count}: silence")

        frame_count += 1

if __name__ == "__main__":
    asyncio.run(main())
```

### ASR - Speech Recognition

Transcribe speech to text with <100ms latency.

#### Rust Example

```rust
use audio_pipeline::{ASREngine, AudioStream, Hz, SampleFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize ASR (CAIMAN: <100ms)
    let asr = ASREngine::caiman()?;

    // Initialize audio stream
    let mut audio_stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
    audio_stream.start().await?;

    // Process audio frames
    loop {
        let frame = audio_stream.next_frame().await?;

        // Check if frame contains speech (VAD pre-filter)
        if !asr.has_speech(&frame) {
            continue;
        }

        // Transcribe speech
        match asr.transcribe(&frame) {
            Ok(result) => {
                if !result.text.is_empty() {
                    println!("Transcript: {} (confidence: {:.2})", result.text, result.confidence);
                }
            }
            Err(e) => {
                eprintln!("ASR error: {}", e);
            }
        }
    }
}
```

#### Python Example

```python
from audio_pipeline import ASREngine, AudioStream

async def main():
    # Initialize ASR (CAIMAN: <100ms)
    asr = ASREngine.caiman()

    # Initialize audio stream
    audio = AudioStream(microphone=True, sample_rate=16000)

    # Process audio frames
    async for frame in audio.stream(chunk_size=512):
        # Check if frame contains speech
        if not asr.has_speech(frame):
            continue

        # Transcribe speech
        text, confidence = asr.transcribe(frame)
        if text:
            print(f"Transcript: {text} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
```

### Sentiment Analysis

Extract emotional state from voice audio.

#### Rust Example

```rust
use audio_pipeline::{SentimentAnalyzer, AudioStream, Hz, SampleFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize sentiment analyzer (GPU: <5ms)
    let sentiment = SentimentAnalyzer::gpu_accelerated()?;

    // Initialize audio stream
    let mut audio_stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
    audio_stream.start().await?;

    // Process audio frames
    loop {
        let frame = audio_stream.next_frame().await?;

        // Analyze sentiment
        match sentiment.analyze(&frame) {
            Ok(vad_scores) => {
                println!(
                    "Valence: {:.2}, Arousal: {:.2}, Dominance: {:.2}",
                    vad_scores.valence, vad_scores.arousal, vad_scores.dominance
                );

                // Interpret sentiment
                if vad_scores.valence < 0.3 {
                    println!("→ User seems frustrated");
                } else if vad_scores.valence > 0.7 {
                    println!("→ User seems happy");
                }
            }
            Err(e) => {
                eprintln!("Sentiment error: {}", e);
            }
        }
    }
}
```

#### Python Example

```python
from audio_pipeline import SentimentAnalyzer, AudioStream

async def main():
    # Initialize sentiment analyzer (GPU: <5ms)
    sentiment = SentimentAnalyzer.gpu_accelerated()

    # Initialize audio stream
    audio = AudioStream(microphone=True, sample_rate=16000)

    # Process audio frames
    async for frame in audio.stream(chunk_size=512):
        # Analyze sentiment
        valence, arousal, dominance = sentiment.analyze(frame)

        print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}, Dominance: {dominance:.2f}")

        # Interpret sentiment
        if valence < 0.3:
            print("→ User seems frustrated")
        elif valence > 0.7:
            print("→ User seems happy")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Advanced Usage

### Stream Processing (Continuous Audio)

Process continuous audio streams with ring buffer and lock-free queues.

#### Rust Example

```rust
use audio_pipeline::{AudioPipeline, AudioStream, Hz, SampleFormat};
use tokio::time::{Duration, interval};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize full pipeline
    let pipeline = AudioPipeline::builder()
        .vad("silero")?
        .asr("caiman")?
        .sentiment("gpu")?
        .build()?;

    // Initialize audio stream
    let mut audio_stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
    audio_stream.start().await?;

    // Process frames at 32ms cadence
    let mut timer = interval(Duration::from_millis(32));
    let mut frame_count = 0;

    loop {
        timer.tick().await;

        let frame = audio_stream.next_frame().await?;

        // Process through full pipeline
        match pipeline.process_frame(&frame)? {
            PipelineResult::NoSpeech => {
                println!("Frame {}: silence", frame_count);
            }
            PipelineResult::Speech { text, vad } => {
                println!("Frame {}: SPEECH", frame_count);
                println!("  Text: {}", text);
                println!("  Sentiment: V={:.2}, A={:.2}, D={:.2}",
                    vad.valence, vad.arousal, vad.dominance);
            }
        }

        frame_count += 1;
    }
}
```

#### Python Example

```python
from audio_pipeline import AudioPipeline, AudioStream
import asyncio

async def main():
    # Initialize full pipeline
    pipeline = AudioPipeline.builder() \
        .vad("silero") \
        .asr("caiman") \
        .sentiment("gpu") \
        .build()

    # Initialize audio stream
    audio = AudioStream(microphone=True, sample_rate=16000)

    # Process frames at 32ms cadence
    frame_count = 0
    async for frame in audio.stream(chunk_size=512):
        # Process through full pipeline
        result = pipeline.process_frame(frame)

        if result.type == "no_speech":
            print(f"Frame {frame_count}: silence")
        elif result.type == "speech":
            print(f"Frame {frame_count}: SPEECH")
            print(f"  Text: {result.text}")
            print(f"  Sentiment: V={result.vad.valence:.2f}, "
                  f"A={result.vad.arousal:.2f}, D={result.vad.dominance:.2f}")

        frame_count += 1

if __name__ == "__main__":
    asyncio.run(main())
```

### GPU Acceleration

Enable GPU acceleration for VAD and sentiment analysis.

#### Rust Example

```rust
use audio_pipeline::{VADDetector, SentimentAnalyzer};
use gpu_accelerator::{GPUEngine, CUDAGraph};

// Initialize GPU engine
let gpu_engine = GPUEngine::new()?;

// Load CUDA Graph for VAD
let vad_graph = gpu_engine.load_graph("models/silero_vad.pt")?;
let vad = VADDetector::cuda_graph(vad_graph)?;

// Load CUDA Graph for sentiment
let sentiment_graph = gpu_engine.load_graph("models/sentiment_vad.pt")?;
let sentiment = SentimentAnalyzer::cuda_graph(sentiment_graph)?;

// Use as normal (GPU acceleration is transparent)
let vad_result = vad.detect(&audio_frame)?;
let vad_scores = sentiment.analyze(&audio_frame)?;
```

#### Python Example

```python
from audio_pipeline import VADDetector, SentimentAnalyzer

# Initialize CUDA-accelerated VAD
vad = VADDetector.cuda_graph("models/silero_vad.pt")

# Initialize CUDA-accelerated sentiment
sentiment = SentimentAnalyzer.cuda_graph("models/sentiment_vad.pt")

# Use as normal (GPU acceleration is transparent)
is_speech, confidence = vad.detect(audio_frame)
valence, arousal, dominance = sentiment.analyze(audio_frame)
```

### Model Selection

Choose different VAD/ASR models based on your requirements.

#### VAD Models

```rust
// Silero VAD (default: <1ms, 99.5% accuracy)
let vad = VADDetector::silero()?;

// WebRTC VAD (fallback: <5ms, 95% accuracy)
let vad = VADDetector::webrtc()?;

// AtomicVAD (experimental: <1ms, 98% accuracy)
let vad = VADDetector::atomic()?;

// Custom VAD
let vad = VADDetector::custom("models/my_vad.onnx")?;
```

#### ASR Models

```rust
// CAIMAN-ASR (default: <100ms, WER <10%, streaming)
let asr = ASREngine::caiman()?;

// Whisper tiny (fallback: <200ms, WER <15%, batch only)
let asr = ASREngine::whisper_tiny()?;

// Custom ASR
let asr = ASREngine::custom("models/my_asr.onnx")?;
```

---

## Configuration and Tuning

### VAD Threshold Tuning

Adjust VAD confidence threshold to reduce false positives/negatives.

```rust
use audio_pipeline::VADDetector;

// Create VAD with custom threshold (default: 0.7)
let vad = VADDetector::builder()
    .model("silero")
    .threshold(0.5)?  // Lower threshold = more sensitive
    .build()?;

// Higher threshold = less sensitive (fewer false positives, more false negatives)
let vad = VADDetector::builder()
    .model("silero")
    .threshold(0.9)?
    .build()?;
```

**Guidelines**:
- **Noisy environment**: Higher threshold (0.8-0.9)
- **Quiet environment**: Lower threshold (0.5-0.7)
- **Default**: 0.7 (balanced)

### Audio Frame Size

Adjust frame size for latency vs. CPU trade-off.

```rust
use audio_pipeline::{AudioStream, Hz, SampleFormat};

// Small frame (16ms, 256 samples): Lower latency, higher CPU
let mut audio_stream = AudioStream::builder()
    .sample_rate(Hz(16000))
    .format(SampleFormat::F32)
    .chunk_size(256)?  // 16ms
    .build()?;

// Default frame (32ms, 512 samples): Balanced
let mut audio_stream = AudioStream::builder()
    .sample_rate(Hz(16000))
    .format(SampleFormat::F32)
    .chunk_size(512)?  // 32ms
    .build()?;

// Large frame (64ms, 1024 samples): Higher latency, lower CPU
let mut audio_stream = AudioStream::builder()
    .sample_rate(Hz(16000))
    .format(SampleFormat::F32)
    .chunk_size(1024)?  // 64ms
    .build()?;
```

**Guidelines**:
- **Real-time interruption detection**: 32ms (default)
- **Minimum latency**: 16ms (higher CPU usage)
- **Batch processing**: 64ms (lower CPU usage)

### Ring Buffer Size

Adjust ring buffer size for context window.

```rust
use audio_pipeline::{AudioStream, Hz, SampleFormat};

// 0.5 second context (8000 samples)
let audio_stream = AudioStream::builder()
    .sample_rate(Hz(16000))
    .buffer_size(8000)?
    .build()?;

// 1 second context (default: 16000 samples)
let audio_stream = AudioStream::builder()
    .sample_rate(Hz(16000))
    .buffer_size(16000)?
    .build()?;

// 2 second context (32000 samples)
let audio_stream = AudioStream::builder()
    .sample_rate(Hz(16000))
    .buffer_size(32000)?
    .build()?;
```

**Guidelines**:
- **Real-time processing**: 1 second (default)
- **Low memory**: 0.5 seconds
- **Longer context**: 2 seconds

---

## Integration Examples

### Integration with equilibrium-tokens

```rust
use audio_pipeline::{AudioPipeline, AudioStream, Hz, SampleFormat};
use equilibrium_tokens::{InterruptionEquilibrium, SurfaceState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize audio pipeline
    let pipeline = AudioPipeline::builder()
        .vad("silero")?
        .asr("caiman")?
        .sentiment("gpu")?
        .build()?;

    // Initialize audio stream
    let mut audio_stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
    audio_stream.start().await?;

    // Initialize interruption equilibrium surface
    let interruption_surface = InterruptionEquilibrium::new()?;

    // Main detection loop
    loop {
        let frame = audio_stream.next_frame().await?;

        // Process frame
        if let PipelineResult::Speech { text, .. } = pipeline.process_frame(&frame)? {
            // Check if interruption
            if !text.is_empty() {
                println!("Interruption detected: {}", text);

                // Reset attention state
                interruption_surface.reset_attention().await?;
            }
        }
    }
}
```

### WebSocket Audio Streaming

```rust
use audio_pipeline::{AudioStream, VADDetector, Hz, SampleFormat};
use tokio_tungstenite::{WebSocketStream, MaybeTlsStream};
use futures::stream::StreamExt;

async fn handle_websocket_audio(
    ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let vad = VADDetector::silero()?;

    let (_, mut read) = ws_stream.split();

    // Process WebSocket audio frames
    while let Some(message) = read.next().await {
        let message = message?;

        if message.is_binary() {
            let audio_frame = message.into_data();

            // Convert bytes to f32 samples
            let samples = bytes_to_f32(&audio_frame);

            // Detect voice activity
            let vad_result = vad.detect(&samples)?;

            if vad_result.is_speech {
                println!("Speech detected (confidence: {:.2})", vad_result.confidence);
            }
        }
    }

    Ok(())
}
```

---

## Troubleshooting

### Common Issues

#### 1. Audio capture fails

**Problem**: `AudioCapture("Permission denied")`

**Solution**:
```bash
# Linux: Add user to audio group
sudo usermod -a -G audio $USER

# Log out and log back in for changes to take effect
```

#### 2. VAD model not found

**Problem**: `VADInference("Model file not found: silero_vad.onnx")`

**Solution**:
```bash
# Download models manually
mkdir -p ~/.audio-pipeline/models
cd ~/.audio-pipeline/models
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
```

#### 3. GPU initialization fails

**Problem**: `GPUError("CUDA not available")`

**Solution**:
```bash
# Check NVIDIA GPU
nvidia-smi

# Install CUDA Toolkit
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Verify CUDA
nvcc --version
```

#### 4. High CPU usage

**Problem**: CPU usage >50% during audio processing

**Solution**:
```rust
// Increase frame size to reduce processing frequency
let audio_stream = AudioStream::builder()
    .chunk_size(1024)?  // 64ms instead of 32ms
    .build()?;

// Or use GPU acceleration
let sentiment = SentimentAnalyzer::gpu_accelerated()?;
```

#### 5. Audio quality issues

**Problem**: VAD/ASR accuracy low

**Solution**:
```rust
// Increase VAD threshold for noisy environments
let vad = VADDetector::builder()
    .threshold(0.9)?
    .build()?;

// Or use noise suppression
let audio_stream = AudioStream::builder()
    .noise_suppression(true)?
    .build()?;
```

### Debug Mode

Enable debug logging for troubleshooting.

**Rust**:
```bash
RUST_LOG=audio_pipeline=debug cargo run
```

**Python**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## API Reference

### Core Types

#### `AudioStream`

```rust
pub struct AudioStream {
    sample_rate: Hz,
    format: SampleFormat,
    chunk_size: usize,
}

impl AudioStream {
    pub fn new(sample_rate: Hz, format: SampleFormat) -> Result<Self>;
    pub fn builder() -> AudioStreamBuilder;
    pub async fn start(&mut self) -> Result<()>;
    pub async fn next_frame(&mut self) -> Result<Vec<f32>>;
}
```

#### `VADDetector`

```rust
pub trait VADDetector: Send + Sync {
    fn detect(&self, audio: &[f32]) -> Result<VADResult>;
    fn reset(&self) -> Result<()>;
}

pub struct VADResult {
    pub is_speech: bool,
    pub confidence: f32,
    pub timestamp: Duration,
}
```

#### `ASREngine`

```rust
pub trait ASREngine: Send + Sync {
    fn transcribe(&self, audio: &[f32]) -> Result<ASRResult>;
    fn transcribe_stream(&self, stream: &AudioStream) -> Result<Stream<ASRResult>>;
    fn has_speech(&self, audio: &[f32]) -> bool;
}

pub struct ASRResult {
    pub text: String,
    pub confidence: f32,
    pub timestamp: Duration,
}
```

#### `SentimentAnalyzer`

```rust
pub trait SentimentAnalyzer: Send + Sync {
    fn analyze(&self, audio: &[f32]) -> Result<VADScores>;
    fn analyze_batch(&self, audio_batch: &[Vec<f32>]) -> Result<Vec<VADScores>>;
}

pub struct VADScores {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
}
```

---

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for deep dive into system design
- Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for contributing to audio-pipeline
- Read [INTEGRATION.md](INTEGRATION.md) for integration with equilibrium-tokens

---

**The grammar is eternal.**
