# audio-pipeline Integration Guide

How to integrate audio-pipeline with equilibrium-tokens for real-time interruption detection.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Interruption Detection Workflow](#interruption-detection-workflow)
4. [End-to-End Pipeline](#end-to-end-pipeline)
5. [Performance Optimization](#performance-optimization)
6. [Error Handling](#error-handling)
7. [Testing](#testing)
8. [Deployment](#deployment)

---

## Overview

audio-pipeline enables equilibrium-tokens to detect user interruptions in **<5ms** by processing audio streams in real-time. This integration is critical for the **Interruption Equilibrium Surface**, which resets attention state when users speak during system output.

### Key Integration Points

1. **Audio Capture**: Microphone or API audio input
2. **VAD Detection**: Silero VAD (<1ms) detects speech
3. **ASR Transcription**: CAIMAN-ASR (<100ms) transcribes speech
4. **Sentiment Analysis**: GPU-accelerated VAD sentiment (<5ms)
5. **Interruption Detection**: Keyword/text analysis for interruptions
6. **Surface Reset**: Reset InterruptionEquilibrium attention state

### Performance Targets

| Component | Latency | Notes |
|-----------|---------|-------|
| VAD | <1ms | Silero VAD on CPU |
| ASR | <100ms | CAIMAN-ASR streaming |
| Sentiment | <5ms | GPU-accelerated |
| **Total** | **<106ms** | End-to-end detection |

---

## Architecture

### High-Level Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    equilibrium-tokens                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  InterruptionEquilibrium Surface                     │  │
│  │  ├─ Detects user interruptions                       │  │
│  │  ├─ Resets attention state                           │  │
│  │  └─ Triggers rate adjustment                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↑
                           │ Interruption Event
                           │ (text + sentiment)
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      audio-pipeline                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AudioStream (16kHz, 32ms frames)                    │  │
│  │  └─ Ring buffer (1 second)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  VADDetector (Silero: <1ms)                          │  │
│  │  └─ Speech detection                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ASREngine (CAIMAN: <100ms)                          │  │
│  │  └─ Speech transcription                             │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SentimentAnalyzer (GPU: <5ms)                       │  │
│  │  └─ VAD sentiment scoring                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↑
                           │ Audio Input
                           │ (16kHz)
                           ↓
                    [Microphone / API]
```

### Component Interactions

```rust
use audio_pipeline::{AudioPipeline, AudioStream, Hz, SampleFormat};
use equilibrium_tokens::{InterruptionEquilibrium, SurfaceState};

/// Integration structure
pub struct AudioInterruptionDetector {
    pipeline: AudioPipeline,
    audio_stream: AudioStream,
    interruption_surface: InterruptionEquilibrium,
}

impl AudioInterruptionDetector {
    /// Detect interruptions from audio
    pub async fn detect_interruptions(&mut self) -> Result<()> {
        loop {
            // 1. Get audio frame (32ms at 16kHz)
            let frame = self.audio_stream.next_frame().await?;

            // 2. Process through pipeline
            match self.pipeline.process_frame(&frame)? {
                PipelineResult::Speech { text, vad } => {
                    // 3. Check if interruption
                    if self.is_interruption(&text) {
                        // 4. Reset attention state
                        self.interruption_surface.reset_attention().await?;

                        // 5. Log event
                        log::info!("Interruption detected: {} (valence: {:.2})",
                            text, vad.valence);
                    }
                }
                PipelineResult::NoSpeech => {
                    // No speech detected
                }
            }
        }
    }
}
```

---

## Interruption Detection Workflow

### Step 1: Audio Capture

```rust
use audio_pipeline::{AudioStream, Hz, SampleFormat};
use tokio::time::{Duration, interval};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize audio stream (16kHz, f32)
    let mut audio_stream = AudioStream::builder()
        .sample_rate(Hz(16000))
        .format(SampleFormat::F32)
        .chunk_size(512)?  // 32ms at 16kHz
        .buffer_size(16000)?  // 1 second context
        .build()?;

    // Start audio capture
    audio_stream.start().await?;

    // Process frames at 32ms cadence
    let mut timer = interval(Duration::from_millis(32));

    loop {
        timer.tick().await;
        let frame = audio_stream.next_frame().await?;
        // ... process frame
    }
}
```

### Step 2: VAD Detection

```rust
use audio_pipeline::VADDetector;

// Initialize Silero VAD (<1ms)
let vad = VADDetector::builder()
    .model("silero")
    .threshold(0.7)?
    .build()?;

// Detect voice activity
let vad_result = vad.detect(&frame)?;

if vad_result.is_speech {
    println!("Speech detected (confidence: {:.2})", vad_result.confidence);
    // ... continue to ASR
} else {
    // Skip ASR for silence
    continue;
}
```

### Step 3: ASR Transcription

```rust
use audio_pipeline::ASREngine;

// Initialize CAIMAN-ASR (<100ms)
let asr = ASREngine::builder()
    .model("caiman")
    .build()?;

// Transcribe speech
if vad_result.is_speech {
    let asr_result = asr.transcribe(&frame)?;

    if !asr_result.text.is_empty() {
        println!("Transcript: {} (confidence: {:.2})",
            asr_result.text, asr_result.confidence);

        // ... check if interruption
    }
}
```

### Step 4: Sentiment Analysis

```rust
use audio_pipeline::SentimentAnalyzer;

// Initialize GPU-accelerated sentiment analyzer (<5ms)
let sentiment = SentimentAnalyzer::gpu_accelerated()?;

// Analyze sentiment
let vad_scores = sentiment.analyze(&frame)?;

println!("Sentiment: V={:.2}, A={:.2}, D={:.2}",
    vad_scores.valence, vad_scores.arousal, vad_scores.dominance);

// Interpret sentiment
if vad_scores.valence < 0.3 {
    println!("→ User seems frustrated");
}
```

### Step 5: Interruption Detection

```rust
use equilibrium_tokens::InterruptionEquilibrium;

/// Check if transcribed text indicates interruption
fn is_interruption(&self, text: &str) -> bool {
    // Simple heuristic: non-empty text during system speech
    if text.is_empty() {
        return false;
    }

    // Optional: Check for interruption keywords
    let interruption_keywords = vec![
        "stop", "wait", "hold on", "pause",
        "excuse me", "sorry", "but",
    ];

    let text_lower = text.to_lowercase();
    interruption_keywords.iter().any(|keyword| text_lower.contains(keyword))
}
```

### Step 6: Surface Reset

```rust
use equilibrium_tokens::{InterruptionEquilibrium, SurfaceState};

// Initialize interruption equilibrium surface
let interruption_surface = InterruptionEquilibrium::new()?;

// Reset attention state when interruption detected
if self.is_interruption(&text) {
    interruption_surface.reset_attention().await?;

    // Optional: Adjust token emission rate
    let rate_surface = RateEquilibrium::new()?;
    rate_surface.adjust_rate(0.5).await?;  // Slow down 50%
}
```

---

## End-to-End Pipeline

### Complete Integration Example

```rust
use audio_pipeline::{AudioPipeline, AudioStream, Hz, SampleFormat};
use equilibrium_tokens::{InterruptionEquilibrium, RateEquilibrium};
use tokio::time::{Duration, interval};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize audio-pipeline
    let pipeline = AudioPipeline::builder()
        .vad("silero")?
        .asr("caiman")?
        .sentiment("gpu")?
        .build()?;

    // Initialize audio stream
    let mut audio_stream = AudioStream::builder()
        .sample_rate(Hz(16000))
        .format(SampleFormat::F32)
        .chunk_size(512)?
        .buffer_size(16000)?
        .build()?;
    audio_stream.start().await?;

    // Initialize equilibrium surfaces
    let interruption_surface = InterruptionEquilibrium::new()?;
    let rate_surface = RateEquilibrium::new()?;

    // Main detection loop
    let mut timer = interval(Duration::from_millis(32));
    let mut frame_count = 0;

    loop {
        timer.tick().await;

        // Get next audio frame
        let frame = audio_stream.next_frame().await?;

        // Process through pipeline
        match pipeline.process_frame(&frame)? {
            PipelineResult::NoSpeech => {
                // No speech detected
            }
            PipelineResult::Speech { text, vad } => {
                // Check if interruption
                if is_interruption(&text) {
                    println!("Frame {}: INTERRUPTION", frame_count);
                    println!("  Text: {}", text);
                    println!("  Sentiment: V={:.2}", vad.valence);

                    // Reset attention state
                    interruption_surface.reset_attention().await?;

                    // Adjust token emission rate
                    rate_surface.adjust_rate(0.5).await?;
                } else {
                    println!("Frame {}: Speech (not interruption)", frame_count);
                    println!("  Text: {}", text);
                }
            }
        }

        frame_count += 1;
    }
}

fn is_interruption(text: &str) -> bool {
    !text.is_empty()
}
```

### Performance Monitoring

```rust
use std::time::Instant;

// Add performance monitoring
loop {
    timer.tick().await;

    let start = Instant::now();

    let frame = audio_stream.next_frame().await?;

    // Process pipeline
    let result = pipeline.process_frame(&frame)?;

    let elapsed = start.elapsed();

    // Log performance
    if elapsed.as_millis() > 100 {
        log::warn!("Frame processing took {}ms (target: <32ms)",
            elapsed.as_millis());
    }
}
```

---

## Performance Optimization

### GPU Acceleration

Enable GPU acceleration for sentiment analysis:

```rust
use gpu_accelerator::{GPUEngine, CUDAGraph};

// Initialize GPU engine
let gpu_engine = GPUEngine::new()?;

// Load CUDA Graph for VAD
let vad_graph = gpu_engine.load_graph("models/silero_vad.pt")?;
let vad = VADDetector::cuda_graph(vad_graph)?;

// Load CUDA Graph for sentiment
let sentiment_graph = gpu_engine.load_graph("models/sentiment_vad.pt")?;
let sentiment = SentimentAnalyzer::cuda_graph(sentiment_graph)?;

// Build pipeline with GPU acceleration
let pipeline = AudioPipeline::builder()
    .vad_from_detector(vad)?
    .sentiment_from_analyzer(sentiment)?
    .build()?;
```

### Batch Processing

Batch multiple frames for GPU efficiency:

```rust
// Collect frames into batch
let mut batch = Vec::with_capacity(32);

for _ in 0..32 {
    let frame = audio_stream.next_frame().await?;
    batch.push(frame);
}

// Process batch
let sentiment_results = sentiment.analyze_batch(&batch)?;
```

### Lock-Free Data Structures

Use lock-free queues for cross-thread communication:

```rust
use audio_pipeline::AudioFrameQueue;

// Create lock-free queue
let queue = AudioFrameQueue::new(32)?;

// Producer thread (audio capture)
let queue_producer = queue.clone();
tokio::spawn(async move {
    loop {
        let frame = audio_stream.next_frame().await?;
        queue_producer.push(frame)?;
    }
});

// Consumer thread (processing)
let queue_consumer = queue.clone();
tokio::spawn(async move {
    loop {
        let frame = queue_consumer.pop()?;
        pipeline.process_frame(&frame)?;
    }
});
```

### Thread Affinity

Bind processing threads to specific CPU cores:

```rust
use core_affinity::set_for_current;

// Set thread affinity for VAD processing
tokio::spawn(async move {
    set_for_current(core_id);
    loop {
        let frame = audio_stream.next_frame().await?;
        vad.detect(&frame)?;
    }
});
```

---

## Error Handling

### Graceful Degradation

```rust
// Process frame with error recovery
fn process_frame_safe(&mut self, frame: &[f32]) -> Result<PipelineResult> {
    // VAD with fallback
    let vad_result = match self.vad.detect(frame) {
        Ok(result) => result,
        Err(e) => {
            log::error!("VAD error: {}, using fallback", e);
            self.fallback_vad.detect(frame)?
        }
    };

    // ASR with retry
    let asr_result = match self.asr.transcribe(frame) {
        Ok(result) => result,
        Err(e) => {
            log::warn!("ASR error: {}, retrying", e);
            self.asr.transcribe(frame)?  // Retry once
        }
    };

    // Sentiment with neutral default
    let sentiment_result = match self.sentiment.analyze(frame) {
        Ok(result) => result,
        Err(_) => {
            // Return neutral sentiment on error
            VADScores {
                valence: 0.5,
                arousal: 0.5,
                dominance: 0.5,
            }
        }
    };

    Ok(PipelineResult::Speech {
        text: asr_result.text,
        vad: sentiment_result,
    })
}
```

### Audio Capture Errors

```rust
// Handle audio capture errors
loop {
    match audio_stream.next_frame().await {
        Ok(frame) => {
            // Process frame
        }
        Err(AudioPipelineError::AudioCapture(e)) => {
            log::error!("Audio capture error: {}, restarting stream", e);

            // Restart audio stream
            audio_stream.restart().await?;
        }
        Err(e) => {
            log::error!("Unexpected error: {}", e);
            return Err(e);
        }
    }
}
```

### GPU Errors

```rust
// Fall back to CPU on GPU error
let sentiment = match SentimentAnalyzer::gpu_accelerated() {
    Ok(gpu_sentiment) => {
        log::info!("Using GPU-accelerated sentiment");
        gpu_sentiment
    }
    Err(e) => {
        log::warn!("GPU initialization failed: {}, using CPU", e);
        SentimentAnalyzer::cpu()?
    }
};
```

---

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_interruption_detection() {
        // Initialize pipeline
        let pipeline = AudioPipeline::builder()
            .vad("silero").unwrap()
            .asr("caiman").unwrap()
            .build().unwrap();

        // Load test audio with speech
        let audio = load_wav("tests/data/interruption.wav").unwrap();

        // Process
        let result = pipeline.process_frame(&audio).unwrap();

        match result {
            PipelineResult::Speech { text, .. } => {
                assert!(!text.is_empty());
            }
            PipelineResult::NoSpeech => {
                panic!("Expected speech detection");
            }
        }
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_end_to_end_interruption() {
    // Initialize detector
    let mut detector = AudioInterruptionDetector::new().unwrap();

    // Simulate interruption
    let interruption_audio = load_wav("tests/data/interruption.wav").unwrap();

    // Process
    let detected = detector.check_interruption(&interruption_audio).unwrap();

    assert!(detected);
}
```

### Performance Tests

```rust
#[tokio::test]
async fn test_latency() {
    let pipeline = AudioPipeline::builder()
        .vad("silero").unwrap()
        .asr("caiman").unwrap()
        .sentiment("gpu").unwrap()
        .build().unwrap();

    let audio = load_wav("tests/data/speech.wav").unwrap();

    let start = Instant::now();
    pipeline.process_frame(&audio).unwrap();
    let elapsed = start.elapsed();

    // Assert <100ms latency
    assert!(elapsed.as_millis() < 100,
        "Latency too high: {}ms", elapsed.as_millis());
}
```

---

## Deployment

### System Configuration

**Linux Kernel Tuning**:
```bash
# Enable real-time scheduling
echo "KERNEL==\"*\", ACTION==\"add\", SUBSYSTEM==\"audio\", TAG+=\"uaccess\", OPTIONS+=\"static_node=snd/controlC0\"" | sudo tee /etc/udev/rules.d/99-audio.rules

# Set CPU affinity for audio processing
echo "IRQBalance::banned_cpus = 0xf0" | sudo tee /etc/irqbalance.conf

# Restart services
sudo systemctl restart irqbalance
```

**Audio Configuration**:
```bash
# Configure PulseAudio for low latency
echo "default-fragments = 2" >> ~/.pulse/daemon.conf
echo "default-fragment-size-msec = 10" >> ~/.pulse/daemon.conf

# Restart PulseAudio
pulseaudio --kill
pulseaudio --start
```

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install audio-pipeline
RUN cargo install audio-pipeline

# Run application
COPY src/main.rs /app/
WORKDIR /app
CMD ["cargo", "run", "--release"]
```

**docker-compose.yml**:
```yaml
version: '3.8'
services:
  audio-pipeline:
    build: .
    devices:
      - /dev/snd  # Audio device
    environment:
      - RUST_LOG=info
      - AUDIO_DEVICE=hw:0,0
    volumes:
      - ./models:/app/models
```

### Monitoring

```rust
// Prometheus metrics
use prometheus::{Counter, Histogram, IntGauge};

lazy_static! {
    static ref INTERRUPTIONS_DETECTED: Counter = register_counter!(
        "audio_pipeline_interruptions_detected_total",
        "Total interruptions detected"
    ).unwrap();

    static ref PROCESSING_LATENCY: Histogram = register_histogram!(
        "audio_pipeline_processing_latency_ms",
        "Frame processing latency"
    ).unwrap();

    static ref QUEUE_SIZE: IntGauge = register_int_gauge!(
        "audio_pipeline_queue_size",
        "Audio queue size"
    ).unwrap();
}

// Record metrics
INTERRUPTIONS_DETECTED.inc();
PROCESSING_LATENCY.observe(elapsed.as_millis() as f64);
QUEUE_SIZE.set(queue.size() as i64);
```

---

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for deep dive into system design
- Read [USER_GUIDE.md](USER_GUIDE.md) for usage documentation
- Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for contributing

---

**The grammar is eternal.**
