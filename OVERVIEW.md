# audio-pipeline Overview

**Real-time audio processing for interruption detection**

## Quick Reference

### Performance
- **VAD Latency**: <1ms (Silero VAD)
- **ASR Latency**: <100ms (CAIMAN-ASR)
- **Sentiment**: <5ms (GPU-accelerated)
- **Throughput**: >100K frames/sec

### Use Case
Enables equilibrium-tokens to detect user interruptions in <5ms by processing audio streams in real-time.

### Timeless Principle
```rust
// Nyquist-Shannon sampling theorem
const MAX_SPEECH_FREQUENCY: Hz = Hz(8000);  // 8kHz max for speech
const MIN_SAMPLE_RATE: Hz = Hz(16000);      // 16kHz (2 × 8kHz)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   equilibrium-tokens                    │
│  InterruptionEquilibrium Surface                        │
│  ├─ Detects interruptions                              │
│  └─ Resets attention state                             │
└─────────────────────────────────────────────────────────┘
                         ↑
                         │ Interruption Event
                         ↓
┌─────────────────────────────────────────────────────────┐
│                      audio-pipeline                     │
│                                                          │
│  AudioStream (16kHz, 32ms frames)                       │
│  └─ Ring buffer (1 second)                             │
│           ↓                                              │
│  VADDetector (Silero: <1ms)                             │
│  └─ Speech detection                                   │
│           ↓ (if speech)                                  │
│  ASREngine (CAIMAN: <100ms)                             │
│  └─ Speech transcription                               │
│           ↓ (parallel)                                   │
│  SentimentAnalyzer (GPU: <5ms)                          │
│  └─ VAD sentiment scoring                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
                         ↑
                         │ Audio Input
                         ↓
                   [Microphone / API]
```

## Code Examples

### Rust - Basic VAD
```rust
use audio_pipeline::{VADDetector, AudioStream, Hz, SampleFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vad = VADDetector::silero()?;
    let mut audio_stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
    audio_stream.start().await?;

    loop {
        let frame = audio_stream.next_frame().await?;
        let vad_result = vad.detect(&frame)?;

        if vad_result.is_speech {
            println!("Speech detected!");
        }
    }
}
```

### Python - Basic VAD
```python
from audio_pipeline import VADDetector, AudioStream

async def main():
    vad = VADDetector.silero()
    audio = AudioStream(microphone=True, sample_rate=16000)

    async for frame in audio.stream(chunk_size=512):
        is_speech, confidence = vad.detect(frame)
        if is_speech:
            print("Speech detected!")
```

### Integration with equilibrium-tokens
```rust
use audio_pipeline::{AudioPipeline, AudioStream, Hz, SampleFormat};
use equilibrium_tokens::InterruptionEquilibrium;

let pipeline = AudioPipeline::builder()
    .vad("silero")?
    .asr("caiman")?
    .sentiment("gpu")?
    .build()?;

let mut audio_stream = AudioStream::new(Hz(16000), SampleFormat::F32)?;
audio_stream.start().await?;

let interruption_surface = InterruptionEquilibrium::new()?;

loop {
    let frame = audio_stream.next_frame().await?;

    if let PipelineResult::Speech { text, .. } = pipeline.process_frame(&frame)? {
        if !text.is_empty() {
            interruption_surface.reset_attention().await?;
        }
    }
}
```

## Documentation

- [README.md](README.md) - Project overview and quick start
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Complete system architecture
- [USER_GUIDE.md](docs/USER_GUIDE.md) - Installation and usage
- [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) - Contributing and development
- [INTEGRATION.md](docs/INTEGRATION.md) - Integration with equilibrium-tokens

## Project Structure

```
audio-pipeline/
├── Cargo.toml                   # Rust dependencies
├── pyproject.toml               # Python dependencies
├── README.md                    # Project overview
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── USER_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   └── INTEGRATION.md
└── src/
    ├── lib.rs                   # Library root
    ├── error.rs                 # Error types
    ├── audio/                   # Audio capture
    ├── vad/                     # Voice activity detection
    ├── asr/                     # Speech recognition
    ├── sentiment/               # Sentiment analysis
    └── pipeline/                # Pipeline orchestrator
```

## Key Features

### 1. Voice Activity Detection (VAD)
- Silero VAD: <1ms, 99.5% accuracy
- WebRTC VAD: Fallback
- AtomicVAD: Experimental

### 2. Speech Recognition (ASR)
- CAIMAN-ASR: <100ms, WER <10%
- Streaming capable
- Real-time transcription

### 3. Sentiment Analysis
- GPU-accelerated: <5ms
- VAD model (Valence-Arousal-Dominance)
- >85% accuracy

### 4. Stream Processing
- Continuous audio capture
- Ring buffer (1 second)
- Lock-free operations
- 32ms frame processing

## Installation

```bash
# Rust
cargo add audio-pipeline

# Python
pip install audio-pipeline
```

## License

MIT

---

**The grammar is eternal. Audio streams become conversational signals.**
