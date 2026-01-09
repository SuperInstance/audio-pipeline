# audio-pipeline Developer Guide

Guide for contributing to and extending audio-pipeline.

## Table of Contents
1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Model Integration](#model-integration)
4. [Testing Strategies](#testing-strategies)
5. [Performance Profiling](#performance-profiling)
6. [Release Process](#release-process)
7. [Code Style](#code-style)
8. [Contributing](#contributing)

---

## Development Setup

### Prerequisites

**Required**:
- Rust 1.70+ (stable)
- Python 3.8+ (for Python bindings)
- Git
- Cargo

**Optional**:
- NVIDIA GPU with CUDA 11.x+ (for GPU development)
- Docker (for containerized testing)

### Clone Repository

```bash
git clone https://github.com/your-org/audio-pipeline.git
cd audio-pipeline
```

### Rust Development Setup

```bash
# Install Rust toolchain
rustup install stable
rustup default stable

# Add development components
rustup component add clippy rustfmt

# Install development dependencies
cargo install cargo-watch
cargo install cargo-edit
```

### Python Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements-dev.txt

# Install PyO3 for Rust-Python bindings
pip install maturin
```

### Build Project

```bash
# Build Rust library
cargo build --release

# Build Python bindings
maturin develop

# Run tests
cargo test

# Run Python tests
pytest tests/python/
```

### IDE Setup

**VS Code**:
```bash
# Install extensions
code --install-extension rust-lang.rust-analyzer
code --install-extension ms-python.python
code --install-extension tamasfe.even-better-toml
```

**Configuration (.vscode/settings.json)**:
```json
{
    "rust-analyzer.cargo.features": "all",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

---

## Project Structure

```
audio-pipeline/
├── Cargo.toml                    # Rust manifest
├── pyproject.toml                # Python manifest
├── README.md                     # Project overview
├── docs/
│   ├── ARCHITECTURE.md          # System architecture
│   ├── USER_GUIDE.md            # User documentation
│   ├── DEVELOPER_GUIDE.md       # This file
│   └── INTEGRATION.md           # Integration guide
├── src/
│   ├── lib.rs                   # Library root
│   ├── audio/
│   │   ├── mod.rs               # Audio module
│   │   ├── stream.rs            # AudioStream implementation
│   │   ├── buffer.rs            # Ring buffer
│   │   └── capture.rs           # Platform-specific capture
│   ├── vad/
│   │   ├── mod.rs               # VAD module
│   │   ├── trait.rs             # VADDetector trait
│   │   ├── silero.rs            # Silero VAD
│   │   ├── webrtc.rs            # WebRTC VAD
│   │   └── atomic.rs            # AtomicVAD
│   ├── asr/
│   │   ├── mod.rs               # ASR module
│   │   ├── trait.rs             # ASREngine trait
│   │   ├── caiman.rs            # CAIMAN-ASR
│   │   └── whisper.rs           # Whisper
│   ├── sentiment/
│   │   ├── mod.rs               # Sentiment module
│   │   ├── trait.rs             # SentimentAnalyzer trait
│   │   ├── gpu.rs               # GPU sentiment
│   │   └── cpu.rs               # CPU sentiment
│   ├── gpu/
│   │   ├── mod.rs               # GPU module
│   │   └── cuda.rs              # CUDA integration
│   ├── pipeline/
│   │   ├── mod.rs               # Pipeline module
│   │   └── orchestrator.rs      # AudioPipeline
│   └── python/
│       └── bindings.rs          # PyO3 bindings
├── models/
│   ├── silero_vad.onnx          # Silero VAD model (5MB)
│   ├── caiman_asr.onnx          # CAIMAN-ASR model
│   └── sentiment_vad.onnx       # Sentiment model
├── tests/
│   ├── common/                  # Test utilities
│   │   ├── mock_audio.rs        # Mock audio generation
│   │   └── assertions.rs        # Custom assertions
│   ├── unit/                    # Unit tests
│   │   ├── vad_tests.rs
│   │   ├── asr_tests.rs
│   │   └── sentiment_tests.rs
│   ├── integration/             # Integration tests
│   │   └── pipeline_tests.rs
│   └── benchmarks/              # Performance benchmarks
│       ├── vad_bench.rs
│       ├── asr_bench.rs
│       └── sentiment_bench.rs
├── python/
│   ├── audio_pipeline/
│   │   ├── __init__.py          # Python package
│   │   ├── vad.py               # Python VAD wrapper
│   │   ├── asr.py               # Python ASR wrapper
│   │   └── sentiment.py         # Python sentiment wrapper
│   └── tests/
│       └── test_bindings.py     # Python binding tests
├── examples/
│   ├── hello_audio.rs           # Basic VAD example
│   ├── asr_example.rs           # ASR example
│   ├── sentiment_example.rs     # Sentiment example
│   └── pipeline_example.rs      # Full pipeline example
├── scripts/
│   ├── download_models.sh       # Download pre-trained models
│   ├── benchmark.sh             # Run benchmarks
│   └── format.sh                # Format code
└── .github/
    └── workflows/
        ├── ci.yml               # CI/CD pipeline
        └── release.yml          # Release automation
```

---

## Model Integration

### Adding a New VAD Model

#### Step 1: Download and Convert Model

```bash
# Download model (example: PyTorch model)
wget https://example.com/my_vad.pt

# Convert to ONNX using torch.onnx
python scripts/convert_to_onnx.py my_vad.pt models/my_vad.onnx
```

#### Step 2: Implement VADDetector Trait

Create `src/vad/my_vad.rs`:

```rust
use crate::vad::{VADDetector, VADResult};
use crate::audio::Hz;
use std::time::Duration;

pub struct MyVAD {
    model: tract_onnx::OnnxModel,
    threshold: f32,
    sample_rate: Hz,
}

impl MyVAD {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            threshold: 0.7,
            sample_rate: Hz(16000),
        })
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
}

impl VADDetector for MyVAD {
    fn detect(&self, audio: &[f32]) -> Result<VADResult> {
        // Preprocess audio (if needed)
        let processed_audio = self.preprocess(audio)?;

        // Run ONNX inference
        let output = self.model.run(tvec!(processed_audio))?;

        // Postprocess output
        let confidence = output[0].to_scalar::<f32>()?;
        let is_speech = confidence > self.threshold;

        Ok(VADResult {
            is_speech,
            confidence,
            timestamp: Duration::from_secs(0),
        })
    }

    fn reset(&self) -> Result<()> {
        // Reset internal state (for streaming VAD)
        Ok(())
    }
}
```

#### Step 3: Register Model

Update `src/vad/mod.rs`:

```rust
pub mod my_vad;

pub use my_vad::MyVAD;

// Add factory function
impl VADDetectorBuilder {
    pub fn my_vad(&self) -> Result<Box<dyn VADDetector>> {
        let vad = MyVAD::new("models/my_vad.onnx")?
            .with_threshold(self.threshold.unwrap_or(0.7));
        Ok(Box::new(vad))
    }
}
```

#### Step 4: Add Tests

Create `tests/unit/my_vad_tests.rs`:

```rust
use audio_pipeline::{VADDetector, Hz};
use audio_pipeline::tests::common::mock_audio;

#[test]
fn test_my_vad_speech_detection() {
    let vad = VADDetector::my_vad().unwrap();

    // Generate mock speech audio
    let speech_audio = mock_audio::generate_speech(512, Hz(16000));

    let result = vad.detect(&speech_audio).unwrap();
    assert!(result.is_speech);
    assert!(result.confidence > 0.7);
}

#[test]
fn test_my_vad_silence_detection() {
    let vad = VADDetector::my_vad().unwrap();

    // Generate mock silence
    let silence_audio = mock_audio::generate_silence(512);

    let result = vad.detect(&silence_audio).unwrap();
    assert!(!result.is_speech);
    assert!(result.confidence < 0.3);
}
```

#### Step 5: Document Model

Update docs/USER_GUIDE.md:

```markdown
### MyVAD

My custom VAD model with <1ms latency.

```rust
let vad = VADDetector::my_vad()?;
```
```

---

### Adding a New ASR Model

Similar process to VAD, but implement `ASREngine` trait:

```rust
use crate::asr::{ASREngine, ASRResult};
use crate::vad::VADDetector;

pub struct MyASR {
    model: tract_onnx::OnnxModel,
    vad: Box<dyn VADDetector>,
    sample_rate: Hz,
}

impl ASREngine for MyASR {
    fn transcribe(&self, audio: &[f32]) -> Result<ASRResult> {
        // Pre-check VAD
        if !self.vad.detect(audio)?.is_speech {
            return Ok(ASRResult {
                text: String::new(),
                confidence: 0.0,
                timestamp: Duration::from_secs(0),
            });
        }

        // Run ASR inference
        let output = self.model.run(tvec!(audio))?;

        // Extract text and confidence
        Ok(ASRResult {
            text: output[0].to_string(),
            confidence: output[1].to_scalar::<f32>()?,
            timestamp: Duration::from_secs(0),
        })
    }

    fn transcribe_stream(&self, stream: &AudioStream) -> Result<Stream<ASRResult>> {
        // Streaming implementation
        unimplemented!()
    }

    fn has_speech(&self, audio: &[f32]) -> bool {
        self.vad.detect(audio).map(|r| r.is_speech).unwrap_or(false)
    }
}
```

---

## Testing Strategies

### Mock Audio Generation

Use `tests/common/mock_audio.rs` for generating test audio:

```rust
pub mod mock_audio {
    use crate::audio::Hz;

    /// Generate mock speech audio (sine wave at speech frequency)
    pub fn generate_speech(num_samples: usize, sample_rate: Hz) -> Vec<f32> {
        let mut audio = vec![0.0; num_samples];
        let frequency = 440.0;  // A4 note

        for (i, sample) in audio.iter_mut().enumerate() {
            let t = i as f32 / sample_rate.0 as f32;
            *sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
        }

        audio
    }

    /// Generate mock silence (zeros)
    pub fn generate_silence(num_samples: usize) -> Vec<f32> {
        vec![0.0; num_samples]
    }

    /// Generate mock noisy audio
    pub fn generate_noise(num_samples: usize, snr_db: f32) -> Vec<f32> {
        let signal = generate_speech(num_samples, Hz(16000));
        let noise_power = signal.iter().map(|x| x * x).sum::<f32>() / num_samples as f32;
        let signal_power = noise_power * f32::powf(10.0, snr_db / 10.0);

        signal.into_iter().map(|s| {
            let noise = (rand::random::<f32>() - 0.5) * 2.0 * signal_power.sqrt();
            s + noise
        }).collect()
    }
}
```

### Unit Tests

Test individual components in isolation:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_threshold() {
        let vad = VADDetector::builder()
            .threshold(0.8)
            .build()
            .unwrap();

        let audio = mock_audio::generate_speech(512, Hz(16000));
        let result = vad.detect(&audio).unwrap();

        // High confidence speech should exceed threshold
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_vad_reset() {
        let vad = VADDetector::silero().unwrap();

        // Process some audio
        let audio = mock_audio::generate_speech(512, Hz(16000));
        vad.detect(&audio).unwrap();

        // Reset should clear state
        vad.reset().unwrap();

        // Subsequent detection should work correctly
        let result = vad.detect(&audio).unwrap();
        assert!(result.is_speech);
    }
}
```

### Integration Tests

Test full pipeline with real audio:

```rust
#[tokio::test]
async fn test_full_pipeline() {
    // Initialize pipeline
    let pipeline = AudioPipeline::builder()
        .vad("silero").unwrap()
        .asr("caiman").unwrap()
        .sentiment("gpu").unwrap()
        .build().unwrap();

    // Load real audio file
    let audio = load_wav("tests/data/speech.wav").unwrap();

    // Process frame
    let result = pipeline.process_frame(&audio).unwrap();

    match result {
        PipelineResult::Speech { text, vad } => {
            assert!(!text.is_empty());
            assert!(vad.valence >= 0.0 && vad.valence <= 1.0);
        }
        PipelineResult::NoSpeech => {
            panic!("Expected speech detection");
        }
    }
}
```

### Real Audio Testing

Use real audio files for validation:

```bash
# Download test audio
wget https://example.com/test_speech.wav -O tests/data/speech.wav
wget https://example.com/test_silence.wav -O tests/data/silence.wav

# Run real audio tests
cargo test --test real_audio
```

### CI/CD Testing

Configure `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Install Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y portaudio19-dev
          pip install maturin pytest

      - name: Run Rust tests
        run: cargo test --verbose

      - name: Run Python tests
        run: pytest tests/python/

      - name: Run benchmarks
        run: cargo test --release --benches
```

---

## Performance Profiling

### Benchmarking

Use Criterion.rs for benchmarks:

**benches/vad_bench.rs**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use audio_pipeline::VADDetector;
use audio_pipeline::tests::common::mock_audio;

fn bench_vad(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad");

    for frame_size in [256, 512, 1024].iter() {
        let vad = VADDetector::silero().unwrap();
        let audio = mock_audio::generate_speech(*frame_size, Hz(16000));

        group.bench_with_input(
            BenchmarkId::new("silero", frame_size),
            frame_size,
            |b, &_size| {
                b.iter(|| {
                    black_box(vad.detect(black_box(&audio)).unwrap())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_vad);
criterion_main!(benches);
```

Run benchmarks:
```bash
cargo bench
```

### Flamegraph Profiling

Generate flamegraphs for hot spots:

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bin audio-pipeline -- example

# View flamegraph
firefox flamegraph.svg
```

### Memory Profiling

Use `valgrind` for memory profiling:

```bash
# Install valgrind
sudo apt-get install valgrind

# Run with valgrind
valgrind --tool=massif --massif-out-file=massif.out \
    cargo run --example hello_audio

# Analyze memory usage
ms_print massif.out
```

### CPU Profiling

Use `perf` for CPU profiling:

```bash
# Record CPU usage
sudo perf record -g cargo run --example hello_audio

# Report hot functions
sudo perf report

# Annotate source code
sudo perf annotate
```

---

## Release Process

### Versioning

Follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Pre-release Checklist

- [ ] All tests pass (`cargo test`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Release notes prepared
- [ ] Benchmark results documented
- [ ] Security audit passed (if applicable)

### Release Steps

1. **Bump version**:
```bash
# Update Cargo.toml
vim Cargo.toml
# Update version from "0.1.0" to "0.2.0"

# Update pyproject.toml
vim pyproject.toml
# Sync version
```

2. **Run full test suite**:
```bash
cargo test --all-features
pytest tests/python/
```

3. **Generate documentation**:
```bash
cargo doc --no-deps --open
```

4. **Create git tag**:
```bash
git add -A
git commit -m "Release v0.2.0"
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin main --tags
```

5. **Publish to crates.io**:
```bash
cargo publish --dry-run  # Check if publish will succeed
cargo publish
```

6. **Publish to PyPI**:
```bash
maturin build --release
twine upload target/wheels/audio_pipeline-0.2.0-*.whl
```

7. **Create GitHub release**:
```bash
gh release create v0.2.0 --notes "Release v0.2.0: ..."
```

---

## Code Style

### Rust Style Guide

Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/):

**Naming**:
- Types: `PascalCase` (e.g., `AudioStream`)
- Functions: `snake_case` (e.g., `detect_speech`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `MAX_SAMPLE_RATE`)

**Error Handling**:
```rust
// Use Result for recoverable errors
pub fn detect(&self, audio: &[f32]) -> Result<VADResult> {
    if audio.is_empty() {
        return Err(AudioPipelineError::InvalidInput("Empty audio".to_string()));
    }
    // ...
}

// Use Option for optional values
pub fn get_confidence(&self) -> Option<f32> {
    Some(self.confidence)
}
```

**Documentation**:
```rust
/// Detect voice activity in audio frame.
///
/// # Arguments
///
/// * `audio` - Audio samples (16kHz, f32 format)
///
/// # Returns
///
/// * `Result<VADResult>` - Detection result with confidence
///
/// # Errors
///
/// * `AudioPipelineError::InvalidInput` - If audio is empty
/// * `AudioPipelineError::InferenceFailed` - If ONNX inference fails
///
/// # Example
///
/// ```rust
/// use audio_pipeline::VADDetector;
///
/// let vad = VADDetector::silero()?;
/// let audio = vec![0.0; 512];
/// let result = vad.detect(&audio)?;
/// ```
pub fn detect(&self, audio: &[f32]) -> Result<VADResult> {
    // ...
}
```

### Python Style Guide

Follow [PEP 8](https://pep8.org/):

```python
class VADDetector:
    """Voice activity detector.

    Args:
        model: VAD model to use (default: "silero")

    Example:
        >>> vad = VADDetector.silero()
        >>> is_speech, confidence = vad.detect(audio)
    """

    def __init__(self, model: str = "silero"):
        self.model = model

    def detect(self, audio: np.ndarray) -> tuple[bool, float]:
        """Detect voice activity in audio.

        Args:
            audio: Audio samples (16kHz, f32 format)

        Returns:
            Tuple of (is_speech, confidence)
        """
        # ...
```

### Formatting

**Rust**:
```bash
cargo fmt
```

**Python**:
```bash
black python/
isort python/
```

### Linting

**Rust**:
```bash
cargo clippy -- -D warnings
```

**Python**:
```bash
pylint python/
flake8 python/
mypy python/
```

---

## Contributing

### Pull Request Process

1. Fork repository
2. Create feature branch (`git checkout -b feature/my-feature`)
3. Make changes
4. Run tests (`cargo test`)
5. Commit changes (`git commit -m "Add my feature"`)
6. Push to branch (`git push origin feature/my-feature`)
7. Create Pull Request

### Pull Request Checklist

- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Code formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Descriptive commit message
- [ ] PR description explains changes

### Code Review Guidelines

- Review for correctness and performance
- Check documentation clarity
- Verify error handling
- Ensure backwards compatibility (unless breaking change)
- Test on multiple platforms (Linux, macOS, Windows)

---

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- Read [USER_GUIDE.md](USER_GUIDE.md) for usage documentation
- Read [INTEGRATION.md](INTEGRATION.md) for integration guide

---

**The grammar is eternal.**
