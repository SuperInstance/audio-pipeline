# audio-pipeline Architecture

**"Audio streams become conversational signals"**

## Table of Contents
1. [Philosophy](#philosophy)
2. [Timeless Principles](#timeless-principles)
3. [Core Abstractions](#core-abstractions)
4. [Component Architecture](#component-architecture)
5. [Stream Processing Pipeline](#stream-processing-pipeline)
6. [GPU Acceleration Strategy](#gpu-acceleration-strategy)
7. [Integration with equilibrium-tokens](#integration-with-equilibrium-tokens)
8. [Performance Characteristics](#performance-characteristics)
9. [Error Handling](#error-handling)
10. [Extensibility](#extensibility)

---

## Philosophy

audio-pipeline transforms raw audio streams into **conversational signals** for the equilibrium-tokens architecture. The system is designed for **real-time performance** (<1ms VAD, <100ms ASR) while maintaining **mathematical correctness** through the Nyquist-Shannon sampling theorem.

### Core Design Principles

1. **Timeless Sampling**: Nyquist-Shannon theorem as foundation
2. **Real-Time First**: Every component optimized for <1ms VAD
3. **Stream Processing**: Continuous audio, not batch
4. **GPU Integration**: Use gpu-accelerator for speed
5. **Interruption Detection**: Primary use case for equilibrium-tokens

### Architectural Goals

- **VAD Latency**: <1ms (P50), <2ms (P99)
- **ASR Latency**: <100ms (end-to-end)
- **Accuracy**: >99% speech detection, <10% WER
- **Throughput**: >100K frames/second
- **Extensibility**: Easy to add new VAD/ASR models

---

## Timeless Principles

### Nyquist-Shannon Sampling Theorem

The foundation of audio-pipeline is the Nyquist-Shannon sampling theorem:

```
To capture a signal with maximum frequency f_max,
the sampling rate must be greater than 2 × f_max.

f_sample > 2 × f_max
```

### Speech Frequency Band

Human speech occupies the frequency range **80 Hz to 8 kHz**:

```rust
/// Maximum frequency in human speech
const MAX_SPEECH_FREQUENCY: Hz = Hz(8000);  // 8kHz

/// Minimum sample rate for perfect reconstruction (Nyquist-Shannon)
const MIN_SAMPLE_RATE: Hz = Hz(16000);      // 16kHz (2 × 8kHz)

/// Standard sample rate for speech processing
const STANDARD_SAMPLE_RATE: Hz = Hz(16000);  // 16kHz
```

### Frame Size Calculation

Frame size determines temporal resolution and latency:

```rust
/// Frame duration for real-time processing (32ms)
const FRAME_DURATION_MS: u64 = 32;

/// Frame size at 16kHz (512 samples)
const FRAME_SIZE: usize = (
    STANDARD_SAMPLE_RATE.0 * FRAME_DURATION_MS / 1000
) as usize;  // 16000 Hz × 0.032 s = 512 samples
```

**Trade-offs**:
- **Smaller frames** (16ms, 256 samples): Lower latency, less context, higher CPU
- **Larger frames** (64ms, 1024 samples): Higher latency, more context, lower CPU
- **32ms frames** (512 samples): Optimal balance for real-time speech

### Ring Buffer Theory

Audio frames are stored in a **ring buffer** (circular buffer) for continuous processing:

```rust
pub struct AudioRingBuffer {
    buffer: Vec<f32>,           // Audio samples
    capacity: usize,            // Total capacity (1 second = 16000 samples)
    write_pos: AtomicUsize,     // Write position (atomic for lock-free)
    read_pos: AtomicUsize,      // Read position (atomic for lock-free)
}

impl AudioRingBuffer {
    /// Write audio samples to buffer
    pub fn write(&self, samples: &[f32]) -> Result<()> {
        // Lock-free write using atomic operations
        let current_write = self.write_pos.load(Ordering::Acquire);
        // ... write samples ...
        self.write_pos.store(new_write_pos, Ordering::Release);
        Ok(())
    }

    /// Read next frame from buffer
    pub fn read_frame(&self) -> Result<Vec<f32>> {
        // Lock-free read using atomic operations
        let current_read = self.read_pos.load(Ordering::Acquire);
        // ... read FRAME_SIZE samples ...
        self.read_pos.store(new_read_pos, Ordering::Release);
        Ok(frame)
    }
}
```

**Ring Buffer Benefits**:
- **Lock-free**: Atomic operations, no mutex overhead
- **Continuous**: Seamless wrap-around when full
- **Deterministic**: Constant-time read/write
- **Memory-efficient**: Fixed capacity, no allocations

---

## Core Abstractions

### 1. AudioStream

Continuous audio capture from microphone or API:

```rust
use std::time::Duration;

/// Audio sample rate in Hertz
#[derive(Debug, Clone, Copy)]
pub struct Hz(pub u32);

/// Audio sample format
#[derive(Debug, Clone, Copy)]
pub enum SampleFormat {
    F32,  // 32-bit float ([-1.0, 1.0])
    I16,  // 16-bit integer ([-32768, 32767])
}

/// Continuous audio stream
pub struct AudioStream {
    sample_rate: Hz,           // 16kHz standard
    format: SampleFormat,      // f32, i16, etc.
    chunk_size: usize,         // 512 samples (32ms at 16kHz)
    buffer: AudioRingBuffer,   // Ring buffer (1 second)
}

impl AudioStream {
    /// Create new audio stream
    pub fn new(sample_rate: Hz, format: SampleFormat) -> Result<Self> {
        Ok(Self {
            sample_rate,
            format,
            chunk_size: FRAME_SIZE,
            buffer: AudioRingBuffer::new(16000)?,  // 1 second at 16kHz
        })
    }

    /// Get next audio frame (32ms at 16kHz)
    pub fn next_frame(&mut self) -> Result<Vec<f32>> {
        self.buffer.read_frame()
    }

    /// Start audio capture (async)
    pub async fn start(&mut self) -> Result<()> {
        // Platform-specific audio capture (Linux: ALSA/PulseAudio, macOS: CoreAudio, Windows: WASAPI)
        Ok(())
    }
}
```

**Design Decisions**:
- **16kHz sample rate**: Sufficient for speech (Nyquist-Shannon)
- **32ms frames**: Optimal balance for real-time processing
- **Ring buffer**: 1 second of audio for context
- **Async I/O**: Non-blocking audio capture

### 2. VADDetector

Voice activity detection trait:

```rust
/// VAD result
#[derive(Debug, Clone)]
pub struct VADResult {
    pub is_speech: bool,        // True if speech detected
    pub confidence: f32,        // Confidence [0.0, 1.0]
    pub timestamp: Duration,    // Frame timestamp
}

/// Voice activity detector
pub trait VADDetector: Send + Sync {
    /// Detect voice activity in audio frame
    fn detect(&self, audio: &[f32]) -> Result<VADResult>;

    /// Reset internal state (for streaming VAD)
    fn reset(&self) -> Result<()>;
}

/// Silero VAD implementation
pub struct SileroVAD {
    model: ONNXModel,           // ONNX model (5MB)
    threshold: f32,             // Speech threshold [0.0, 1.0]
    sample_rate: Hz,            // 16kHz
}

impl VADDetector for SileroVAD {
    fn detect(&self, audio: &[f32]) -> Result<VADResult> {
        // Run ONNX inference (<1ms)
        let output = self.model.run(audio)?;

        // Apply threshold
        let is_speech = output[0] > self.threshold;
        let confidence = output[0];

        Ok(VADResult {
            is_speech,
            confidence,
            timestamp: Duration::from_secs(0),
        })
    }

    fn reset(&self) -> Result<()> {
        // Reset hidden states (for streaming)
        Ok(())
    }
}
```

**Design Decisions**:
- **Trait-based**: Easy to add new VAD models
- **Streaming support**: Reset for continuous processing
- **Threshold tuning**: Adjustable confidence threshold
- **ONNX runtime**: Cross-platform inference

### 3. ASREngine

Automatic speech recognition trait:

```rust
/// ASR result
#[derive(Debug, Clone)]
pub struct ASRResult {
    pub text: String,           // Transcribed text
    pub confidence: f32,        // Confidence [0.0, 1.0]
    pub timestamp: Duration,    // Timestamp
}

/// Automatic speech recognition engine
pub trait ASREngine: Send + Sync {
    /// Transcribe audio frame (blocking)
    fn transcribe(&self, audio: &[f32]) -> Result<ASRResult>;

    /// Transcribe audio stream (async, streaming)
    fn transcribe_stream(&self, stream: &AudioStream) -> Result<Stream<ASRResult>>;

    /// Check if frame contains speech (VAD pre-filter)
    fn has_speech(&self, audio: &[f32]) -> bool;
}

/// CAIMAN-ASR implementation (streaming, low-latency)
pub struct CAIMANASR {
    model: ONNXModel,           // ONNX model
    vad: Box<dyn VADDetector>,  // Built-in VAD for efficiency
    sample_rate: Hz,            // 16kHz
}

impl ASREngine for CAIMANASR {
    fn transcribe(&self, audio: &[f32]) -> Result<ASRResult> {
        // Pre-check VAD (early exit if no speech)
        if !self.vad.detect(audio)?.is_speech {
            return Ok(ASRResult {
                text: String::new(),
                confidence: 0.0,
                timestamp: Duration::from_secs(0),
            });
        }

        // Run ASR inference (<100ms)
        let output = self.model.run(audio)?;

        Ok(ASRResult {
            text: output.text,
            confidence: output.confidence,
            timestamp: Duration::from_secs(0),
        })
    }

    fn transcribe_stream(&self, stream: &AudioStream) -> Result<Stream<ASRResult>> {
        // Continuous streaming ASR
        // Returns async stream of ASR results
        unimplemented!()
    }

    fn has_speech(&self, audio: &[f32]) -> bool {
        self.vad.detect(audio).map(|r| r.is_speech).unwrap_or(false)
    }
}
```

**Design Decisions**:
- **Streaming-first**: Optimized for continuous audio
- **VAD pre-filter**: Skip ASR inference if no speech
- **Async support**: Non-blocking transcription
- **Modular**: Separate VAD + ASR for flexibility

### 4. SentimentAnalyzer

Sentiment analysis from audio:

```rust
/// VAD sentiment scores (Valence-Arousal-Dominance)
#[derive(Debug, Clone)]
pub struct VADScores {
    pub valence: f32,       // Positive/negative [0.0, 1.0]
    pub arousal: f32,       // Calm/excited [0.0, 1.0]
    pub dominance: f32,     // Weak/strong [0.0, 1.0]
}

/// Sentiment analyzer
pub trait SentimentAnalyzer: Send + Sync {
    /// Analyze sentiment from audio frame
    fn analyze(&self, audio: &[f32]) -> Result<VADScores>;

    /// Batch analysis for efficiency
    fn analyze_batch(&self, audio_batch: &[Vec<f32>]) -> Result<Vec<VADScores>>;
}

/// GPU-accelerated sentiment analyzer
pub struct GPUSentimentAnalyzer {
    model: CudaGraph,            // CUDA Graph for GPU acceleration
    vad: Box<dyn VADDetector>,   // Built-in VAD
}

impl SentimentAnalyzer for GPUSentimentAnalyzer {
    fn analyze(&self, audio: &[f32]) -> Result<VADScores> {
        // Pre-check VAD (early exit if no speech)
        if !self.vad.detect(audio)?.is_speech {
            return Ok(VADScores {
                valence: 0.5,
                arousal: 0.5,
                dominance: 0.5,
            });
        }

        // GPU inference (<5ms with CUDA Graph)
        let output = self.model.execute(audio)?;

        Ok(VADScores {
            valence: output[0],
            arousal: output[1],
            dominance: output[2],
        })
    }

    fn analyze_batch(&self, audio_batch: &[Vec<f32>]) -> Result<Vec<VADScores>> {
        // Batch processing for GPU efficiency
        unimplemented!()
    }
}
```

**Design Decisions**:
- **VAD model**: Valence-Arousal-Dominance (standard in sentiment research)
- **GPU acceleration**: CUDA Graph for constant-time inference
- **Batch support**: GPU efficiency through batching
- **Early exit**: Skip inference if no speech

---

## Component Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        audio-pipeline                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Audio Input
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                          AudioStream                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  AudioRingBuffer (1 second @ 16kHz)                      │   │
│  │  ├─ Lock-free read/write (atomic operations)             │   │
│  │  ├─ 16000 samples capacity                               │   │
│  │  └─ 512-sample frames (32ms)                             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Frame (512 samples, 32ms)
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                          VADDetector                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Silero VAD (ONNX)                                       │   │
│  │  ├─ Model: 5MB ONNX                                      │   │
│  │  ├─ Latency: <1ms                                        │   │
│  │  ├─ Accuracy: 99.5%                                      │   │
│  │  └─ Output: {is_speech, confidence}                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ If speech detected
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                          ASREngine                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  CAIMAN-ASR (Streaming)                                   │   │
│  │  ├─ Latency: <100ms                                      │   │
│  │  ├─ WER: <10%                                            │   │
│  │  ├─ Streaming: Yes                                       │   │
│  │  └─ Output: Transcribed text                             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ (parallel)
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                       SentimentAnalyzer                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  GPU-Accelerated VAD Sentiment                            │   │
│  │  ├─ CUDA Graph (gpu-accelerator)                         │   │
│  │  ├─ Latency: <5ms (with GPU)                             │   │
│  │  ├─ Accuracy: >85%                                       │   │
│  │  └─ Output: {valence, arousal, dominance}                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Conversational signals
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    equilibrium-tokens                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  InterruptionEquilibrium Surface                         │   │
│  │  ├─ Detects interruptions from VAD + ASR                 │   │
│  │  ├─ Resets attention state                               │   │
│  │  └─ Adjusts token emission rate                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

```rust
/// Main pipeline orchestrator
pub struct AudioPipeline {
    audio_stream: AudioStream,
    vad: Box<dyn VADDetector>,
    asr: Box<dyn ASREngine>,
    sentiment: Box<dyn SentimentAnalyzer>,
}

impl AudioPipeline {
    /// Process single frame through full pipeline
    pub fn process_frame(&mut self, frame: &[f32]) -> Result<PipelineResult> {
        // VAD: <1ms
        let vad_result = self.vad.detect(frame)?;

        if !vad_result.is_speech {
            return Ok(PipelineResult::NoSpeech);
        }

        // ASR: <100ms (parallel with sentiment)
        let asr_future = self.asr.transcribe(frame)?;

        // Sentiment: <5ms (parallel with ASR)
        let sentiment_future = self.sentiment.analyze(frame)?;

        // Wait for both to complete
        let asr_result = asr_future?;
        let sentiment_result = sentiment_future?;

        Ok(PipelineResult::Speech {
            text: asr_result.text,
            vad: sentiment_result,
        })
    }
}
```

---

## Stream Processing Pipeline

### Real-Time Pipeline

```
[realtime-core] High-resolution timer (32ms ticks)
         ↓
    Audio Capture (ALSA/PulseAudio/CoreAudio/WASAPI)
         ↓
    AudioRingBuffer (lock-free write)
         ↓
    ┌──────────────────────────────────────┐
    │  Processing Loop (32ms cadence)       │
    │  ┌────────────────────────────────┐  │
    │  │ 1. Read frame (512 samples)    │  │
    │  │    from ring buffer            │  │
    │  └────────────────────────────────┘  │
    │  ┌────────────────────────────────┐  │
    │  │ 2. VAD detect (<1ms)           │  │
    │  │    ├─ Silero VAD inference     │  │
    │  │    └─ Speech?                  │  │
    │  └────────────────────────────────┘  │
    │  ┌────────────────────────────────┐  │
    │  │ 3a. ASR transcribe (<100ms)    │  │
    │  │     (parallel)                 │  │
    │  │    ├─ CAIMAN-ASR inference     │  │
    │  │    └─ Transcribed text         │  │
    │  └────────────────────────────────┘  │
    │  ┌────────────────────────────────┐  │
    │  │ 3b. Sentiment analyze (<5ms)   │  │
    │  │     (parallel)                 │  │
    │  │    ├─ GPU VAD inference        │  │
    │  │    └─ VAD scores               │  │
    │  └────────────────────────────────┘  │
    │  ┌────────────────────────────────┐  │
    │  │ 4. Emit conversational signal  │  │
    │  │    ├─ VAD result               │  │
    │  │    ├─ ASR result               │  │
    │  │    └─ Sentiment result         │  │
    │  └────────────────────────────────┘  │
    └──────────────────────────────────────┘
         ↓
    [equilibrium-tokens] InterruptionEquilibrium
```

### Lock-Free Architecture

```rust
/// Lock-free audio frame queue for cross-thread communication
pub struct AudioFrameQueue {
    buffer: Vec<Option<Vec<f32>>>,    // Ring buffer
    capacity: usize,                   // Capacity (e.g., 32 frames)
    write_idx: AtomicUsize,            // Write index (atomic)
    read_idx: AtomicUsize,             // Read index (atomic)
}

impl AudioFrameQueue {
    /// Push frame (lock-free)
    pub fn push(&self, frame: Vec<f32>) -> Result<()> {
        let current_write = self.write_idx.load(Ordering::Acquire);
        let next_write = (current_write + 1) % self.capacity;

        // Check if buffer is full
        if next_write == self.read_idx.load(Ordering::Acquire) {
            return Err(Error::BufferFull);
        }

        // Write frame
        self.buffer[current_write] = Some(frame);

        // Update write index
        self.write_idx.store(next_write, Ordering::Release);
        Ok(())
    }

    /// Pop frame (lock-free)
    pub fn pop(&self) -> Result<Vec<f32>> {
        let current_read = self.read_idx.load(Ordering::Acquire);

        // Check if buffer is empty
        if current_read == self.write_idx.load(Ordering::Acquire) {
            return Err(Error::BufferEmpty);
        }

        // Read frame
        let frame = self.buffer[current_read]
            .take()
            .ok_or(Error::NoFrame)?;

        // Update read index
        let next_read = (current_read + 1) % self.capacity;
        self.read_idx.store(next_read, Ordering::Release);

        Ok(frame)
    }
}
```

---

## GPU Acceleration Strategy

### CUDA Graph Integration

audio-pipeline uses **gpu-accelerator** for CUDA Graph acceleration:

```rust
use gpu_accelerator::{CUDAGraph, GPUEngine};

/// GPU-accelerated VAD
pub struct CUDAVAD {
    engine: GPUEngine,
    graph: CudaGraph,
    threshold: f32,
}

impl CUDAVAD {
    pub fn new(model_path: &str) -> Result<Self> {
        let engine = GPUEngine::new()?;
        let graph = engine.load_graph(model_path)?;

        Ok(Self {
            engine,
            graph,
            threshold: 0.7,
        })
    }
}

impl VADDetector for CUDAVAD {
    fn detect(&self, audio: &[f32]) -> Result<VADResult> {
        // CUDA Graph execution (constant-time, <1ms)
        let output = self.graph.execute(audio)?;

        Ok(VADResult {
            is_speech: output[0] > self.threshold,
            confidence: output[0],
            timestamp: Duration::from_secs(0),
        })
    }

    fn reset(&self) -> Result<()> {
        // Reset CUDA Graph hidden states
        self.graph.reset()?;
        Ok(())
    }
}
```

### GPU Benefits

- **21-37% latency reduction** vs CPU inference
- **Eliminated CPU-GPU sync** overhead (CUDA Graph)
- **Constant-time inference** for repeated operations
- **Parallel sentiment + VAD** processing

---

## Integration with equilibrium-tokens

### Interruption Detection Workflow

```rust
use equilibrium_tokens::{InterruptionEquilibrium, SurfaceState};
use audio_pipeline::{AudioStream, VADDetector, ASREngine};

/// Interruption detection from audio
pub struct InterruptionDetector {
    audio_stream: AudioStream,
    vad: Box<dyn VADDetector>,
    asr: Box<dyn ASREngine>,
    interruption_surface: InterruptionEquilibrium,
}

impl InterruptionDetector {
    /// Run detection loop
    pub async fn run(&mut self) -> Result<()> {
        loop {
            // 1. Wait for next audio frame (32ms at 16kHz)
            let frame = self.audio_stream.next_frame().await?;

            // 2. VAD detect (<1ms)
            let vad_result = self.vad.detect(&frame)?;

            if !vad_result.is_speech {
                continue;  // No speech, skip ASR
            }

            // 3. ASR transcribe (<100ms)
            if let Ok(asr_result) = self.asr.transcribe(&frame) {
                // 4. Check if interruption
                if self.is_interruption(&asr_result.text) {
                    // 5. Reset attention state
                    self.interruption_surface
                        .reset_attention()
                        .await?;

                    println!("Interruption detected: {}", asr_result.text);
                }
            }
        }
    }

    /// Check if transcribed text indicates interruption
    fn is_interruption(&self, text: &str) -> bool {
        // Simple heuristic: non-empty text during system speech
        !text.is_empty()
    }
}
```

### End-to-End Pipeline

```
User speaks (microphone)
         ↓
    [AudioStream] Capture 16kHz audio
         ↓
    [realtime-core] 32ms frame timer
         ↓
    [AudioRingBuffer] Lock-free buffer
         ↓
    [VADDetector] Silero VAD: <1ms
         ↓ (speech detected)
    [ASREngine] CAIMAN-ASR: <100ms
         ↓ (text transcribed)
    [Keyword Check] Is interruption?
         ↓ (yes)
    [InterruptionEquilibrium] Reset attention
         ↓
    [RateEquilibrium] Adjust token rate
         ↓
    System response adjusted
```

---

## Performance Characteristics

### Latency Breakdown

| Component | P50 Latency | P99 Latency | Notes |
|-----------|-------------|-------------|-------|
| Audio Capture | <1ms | <2ms | Hardware dependent |
| Ring Buffer Read | <0.1ms | <0.5ms | Lock-free atomic |
| VAD (Silero) | <1ms | <2ms | CPU inference |
| ASR (CAIMAN) | <100ms | <150ms | Streaming ASR |
| Sentiment (GPU) | <5ms | <10ms | CUDA Graph |
| **Total** | **<106ms** | **<164ms** | End-to-end |

### Throughput

- **VAD**: >100K frames/second (single CPU core)
- **ASR**: >10 frames/second (streaming, batch processing)
- **Sentiment**: >20K frames/second (GPU batch processing)

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| AudioRingBuffer | 64KB | 1 second @ 16kHz (f32) |
| Silero VAD Model | 5MB | ONNX model |
| CAIMAN-ASR Model | ~50MB | ONNX model |
| Sentiment Model | ~10MB | CUDA model |
| **Total** | ~65MB | Without GPU memory |

---

## Error Handling

### Error Types

```rust
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
}

impl std::error::Error for AudioPipelineError {}
```

### Error Recovery

```rust
impl AudioPipeline {
    /// Process frame with error recovery
    pub fn process_frame_safe(&mut self, frame: &[f32]) -> Result<PipelineResult> {
        // VAD with fallback
        let vad_result = match self.vad.detect(frame) {
            Ok(result) => result,
            Err(e) => {
                log::error!("VAD error: {}, using fallback", e);
                // Fallback to WebRTC VAD
                self.fallback_vad.detect(frame)?
            }
        };

        // ASR with retry
        let asr_result = match self.asr.transcribe(frame) {
            Ok(result) => result,
            Err(e) => {
                log::warn!("ASR error: {}, retrying", e);
                // Retry once
                self.asr.transcribe(frame)?
            }
        };

        // Sentiment with graceful degradation
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
}
```

---

## Extensibility

### Adding New VAD Models

```rust
/// Custom VAD implementation
pub struct MyCustomVAD {
    // Custom fields
}

impl VADDetector for MyCustomVAD {
    fn detect(&self, audio: &[f32]) -> Result<VADResult> {
        // Custom VAD logic
        Ok(VADResult {
            is_speech: true,
            confidence: 0.9,
            timestamp: Duration::from_secs(0),
        })
    }

    fn reset(&self) -> Result<()> {
        // Reset logic
        Ok(())
    }
}

/// Register custom VAD
let vad = Box::new(MyCustomVAD::new()?);
let pipeline = AudioPipeline::new(audio_stream, vad, asr, sentiment)?;
```

### Adding New ASR Models

```rust
/// Custom ASR implementation
pub struct MyCustomASR {
    // Custom fields
}

impl ASREngine for MyCustomASR {
    fn transcribe(&self, audio: &[f32]) -> Result<ASRResult> {
        // Custom ASR logic
        Ok(ASRResult {
            text: "Hello".to_string(),
            confidence: 0.95,
            timestamp: Duration::from_secs(0),
        })
    }

    fn transcribe_stream(&self, stream: &AudioStream) -> Result<Stream<ASRResult>> {
        // Streaming ASR logic
        unimplemented!()
    }

    fn has_speech(&self, audio: &[f32]) -> bool {
        // Speech detection logic
        true
    }
}
```

---

## Conclusion

audio-pipeline transforms raw audio streams into **conversational signals** for equilibrium-tokens, enabling <5ms interruption detection through real-time VAD, ASR, and sentiment analysis. The architecture is built on the **timeless Nyquist-Shannon sampling theorem** while optimizing for real-time performance through GPU acceleration, lock-free data structures, and stream processing.

**The grammar is eternal. Audio streams become conversational signals.**
