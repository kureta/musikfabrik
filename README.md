# Musikfabrik

Audio feature extraction and composition tools for spectral music composition.

## Overview

This project provides modular tools for:
- **Audio Feature Extraction**: Extract partials, f0, and loudness from audio files
- **Dissonance Analysis**: Calculate dissonance curves and generate scales from consonant intervals
- **DDSP Composition**: Compose music using spectral stretching and DDSP synthesis

## Installation

```bash
# Install dependencies using uv
uv sync
```

## User-Facing Notebooks

The project includes three main Marimo notebooks for interactive audio analysis and composition:

### 1. Audio Feature Extraction (`notebooks/audio_feature_extraction.py`)

Extract audio features for further analysis or composition.

**Features:**
- Load audio files from disk
- Extract static partials using FFT (one set for entire file)
- Extract dynamic f0 and loudness using STFT (time-varying)
- Apply time-varying spectral stretching to audio
- Save extracted features to JSON or pickle files

**Usage:**
```bash
marimo edit notebooks/audio_feature_extraction.py
```

**Workflow:**
1. Select or enter an audio file path
2. Adjust partial extraction parameters (number of partials, peak height threshold, f0 estimate)
3. Extract dynamic features (f0 and loudness over time)
4. Apply spectral stretching with custom envelope (start, peak, end stretch factors)
5. Save features to JSON/pickle for later use

### 2. Dissonance Curves and Scale Generation (`notebooks/dissonance_and_scales.py`)

Generate scales based on sensory dissonance analysis.

**Features:**
- Define synthetic partials or load saved partials from JSON
- Calculate dissonance curves using Plomp-Levelt model
- Detect consonant intervals (local minima in dissonance curve)
- Generate valid scales from consonant intervals
- Save scales and intervals to JSON

**Usage:**
```bash
marimo edit notebooks/dissonance_and_scales.py
```

**Workflow:**
1. Choose partial source:
   - **Synthetic**: Define f0, number of partials, stretch factors, amplitude decay
   - **File**: Load previously extracted partials from JSON
2. Set dissonance curve parameters (start/end cents, resolution)
3. Find consonant intervals (adjust deviation and distance thresholds)
4. Generate scales from interval groups
5. Save results to JSON

### 3. Composition Workspace (`notebooks/composition_workspace.py`)

Compose music using DDSP synthesis and spectral stretching.

**Features:**
- Load pre-trained DDSP models (Drums, Violin, Cello, Flute)
- Audio-driven DDSP control (extract f0/loudness from audio)
- Manual phrase composition with multiple modes:
  - Simple sustained notes
  - Note sequences
  - Glissando (linear, exponential, logarithmic)
  - Vibrato (adjustable rate and depth)
  - Custom breakpoints for complex phrases
- Save generated audio to WAV files

**Usage:**
```bash
marimo edit notebooks/composition_workspace.py
```

**Workflow:**

**Audio-Driven Mode:**
1. Select a DDSP model
2. Load an audio file
3. Features (f0, loudness) are automatically extracted
4. Set spectral stretch envelope
5. Generate and save audio

**Manual Composition Mode:**
1. Select a DDSP model
2. Choose phrase type (note, sequence, glissando, vibrato, breakpoints)
3. Set parameters for the chosen phrase type
4. Generate and preview audio
5. Save to WAV file

## Python API

The notebooks use three modular Python libraries that can also be used directly:

### Audio Features (`musikfabrik.audio_features`)

```python
from musikfabrik.audio_features import (
    load_audio,
    extract_partials,
    get_dynamic_f0,
    get_dynamic_loudness,
    stretch_spectrum
)

# Load and extract partials
sample, sr = load_audio("audio.wav")
partials = extract_partials(sample, n_partials=8, f0_estimate=440.0)

# Extract time-varying features
f0 = get_dynamic_f0(sample)
loudness = get_dynamic_loudness(sample)

# Apply spectral stretching
from musikfabrik.audio_features import create_stretch_curve
stretch_curve = create_stretch_curve(len(f0), peak_stretch=1.05)
stretched = stretch_spectrum(sample, f0, stretch_curve)
```

### Dissonance Analysis (`musikfabrik.dissonance_analysis`)

```python
from musikfabrik.dissonance_analysis import (
    calculate_dissonance_curve,
    find_consonant_intervals,
    generate_scales_from_intervals
)

# Calculate dissonance curve
cents, roughness = calculate_dissonance_curve(
    partials["frequencies"],
    partials["amplitudes"],
    partials["frequencies"],
    partials["amplitudes"]
)

# Find consonant intervals
peaks = find_consonant_intervals(roughness, deviation=0.7)

# Generate scales
scales = generate_scales_from_intervals(interval_groups)
```

### DDSP Helpers (`musikfabrik.ddsp_helpers`)

```python
from musikfabrik.ddsp_helpers import (
    DDSPPhrase,
    adsr_envelope,
    glissando,
    vibrato,
    generate_audio_with_ddsp
)
from performer.models.ddsp_module import get_harmonic_stretching_model

# Load DDSP model
model = get_harmonic_stretching_model("path/to/checkpoint.ckpt")

# Create phrase
pitch_curve = glissando(duration=3.0, start_pitch=60, end_pitch=72)
f0_curve = librosa.midi_to_hz(pitch_curve)
loudness_curve = constant_curve(3.0, -40)
stretch_curve = constant_curve(3.0, 1.05)

# Generate audio
audio = generate_audio_with_ddsp(model, f0_curve, loudness_curve, stretch_curve)
```

## Project Structure

```
musikfabrik/
├── notebooks/                          # Marimo notebooks
│   ├── audio_feature_extraction.py    # Extract audio features
│   ├── dissonance_and_scales.py       # Generate scales from dissonance
│   ├── composition_workspace.py       # Compose with DDSP
│   └── fingering_charts.py            # Brass instrument fingering charts
├── src/
│   ├── musikfabrik/                    # Main library
│   │   ├── audio_features.py          # Audio feature extraction
│   │   ├── dissonance_analysis.py     # Dissonance and scale generation
│   │   ├── ddsp_helpers.py            # DDSP control utilities
│   │   └── seth.py                    # Dissonance calculation primitives
│   ├── composition/                    # Composition helpers
│   │   ├── common.py                  # Signal generation utilities
│   │   └── instrument.py              # Phrase and score classes
│   └── performer/                      # DDSP models and utilities
│       ├── models/                    # DDSP model implementations
│       └── utils/                     # Feature extraction and utilities
└── data/
    ├── samples/                       # Audio samples
    ├── checkpoints/                   # DDSP model checkpoints
    └── extracted_features/            # Saved features (JSON/pickle)
```

## License

[Add license information]

## Credits

[Add credits]
