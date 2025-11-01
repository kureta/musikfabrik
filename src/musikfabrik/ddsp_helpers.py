# pyright: basic
"""DDSP control and phrase generation utilities.

This module provides helper functions for controlling DDSP models including:
- Envelope generation (ADSR)
- Vibrato and glissando effects
- Phrase composition utilities
"""

import librosa
import numpy as np
import torch
from numpy.typing import NDArray

from composition.common import generate_audio

FloatArray = NDArray[np.float64]

# Constants
SAMPLE_RATE = 48000
FRAME_RATE = 250  # frames per second
PAD_START_DURATION = 2.0  # seconds of padding at the start
PAD_END_DURATION_MULTIPLIER = 4.0  # multiplier for end padding relative to start


def generate_audio_with_ddsp(
    model,
    f0: FloatArray,
    loudness: FloatArray,
    stretch: FloatArray,
) -> FloatArray:
    """Generate audio using a DDSP model.

    Args:
        model: DDSP model with controller, harmonics, noise, and reverb
        f0: Fundamental frequency over time
        loudness: Loudness in dB over time
        stretch: Spectral stretch factor over time

    Returns:
        Generated audio samples
    """
    return generate_audio(model, f0, loudness, stretch)


def time_to_frames(duration: float, fps: int = FRAME_RATE) -> int:
    """Convert duration in seconds to number of frames.

    Args:
        duration: Duration in seconds
        fps: Frames per second

    Returns:
        Number of frames
    """
    return int(duration * fps)


def get_time_array(duration: float, fps: int = FRAME_RATE) -> FloatArray:
    """Get time array for a given duration.

    Args:
        duration: Duration in seconds
        fps: Frames per second

    Returns:
        Array of time values
    """
    return np.linspace(0.0, duration, time_to_frames(duration, fps))


def constant_curve(duration: float, value: float = 0.0, fps: int = FRAME_RATE) -> FloatArray:
    """Create a constant value curve.

    Args:
        duration: Duration in seconds
        value: Constant value
        fps: Frames per second

    Returns:
        Array of constant values
    """
    return np.ones(time_to_frames(duration, fps), dtype="float32") * value


def linear_curve(
    duration: float,
    start: float = 0.0,
    end: float = 1.0,
    fps: int = FRAME_RATE,
) -> FloatArray:
    """Create a linear interpolation curve.

    Args:
        duration: Duration in seconds
        start: Starting value
        end: Ending value
        fps: Frames per second

    Returns:
        Array of linearly interpolated values
    """
    t = get_time_array(duration, fps)
    a = (end - start) / duration
    b = start

    return (a * t + b).astype("float32")


def adsr_envelope(
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    peak: float = 1.0,
    sustain_level: float = 0.7,
    fps: int = FRAME_RATE,
) -> FloatArray:
    """Create an ADSR envelope.

    Args:
        attack: Attack time in seconds
        decay: Decay time in seconds
        sustain: Sustain time in seconds
        release: Release time in seconds
        peak: Peak amplitude
        sustain_level: Sustain level (0-1)
        fps: Frames per second

    Returns:
        ADSR envelope array
    """
    attack_curve = linear_curve(attack, 0.0, peak, fps)
    decay_curve = linear_curve(decay, peak, sustain_level, fps)
    sustain_curve = constant_curve(sustain, sustain_level, fps)
    release_curve = linear_curve(release, sustain_level, 0.0, fps)

    return np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])


def random_adsr_envelope(
    duration: float,
    sustain_level: float = 0.5,
    fps: int = FRAME_RATE,
) -> FloatArray:
    """Create a random ADSR envelope.

    Args:
        duration: Total duration in seconds
        sustain_level: Sustain level (0-1)
        fps: Frames per second

    Returns:
        Random ADSR envelope array
    """
    # Generate random breakpoints
    break_points = sorted(np.random.uniform(0, duration, 3))
    durations = np.diff(np.array([0, *break_points, duration]))

    env = adsr_envelope(*durations, 1.0, sustain_level, fps)

    # Ensure correct length
    steps = time_to_frames(duration, fps)
    if len(env) < steps:
        env = np.pad(env, (0, steps - len(env)))

    return env


def breakpoint_curve(
    values: list[float],
    durations: list[float],
    interpolation: str = "linear",
    fps: int = FRAME_RATE,
) -> FloatArray:
    """Create a curve from breakpoints with interpolation.

    Args:
        values: Breakpoint values
        durations: Duration of each segment
        interpolation: 'linear' or 'floor' (step function)
        fps: Frames per second

    Returns:
        Interpolated curve
    """
    segments = []

    for idx, d in enumerate(durations):
        if interpolation == "linear":
            segment = linear_curve(d, values[idx], values[idx + 1], fps)
        elif interpolation == "floor":
            segment = constant_curve(d, values[idx], fps)
        else:
            raise ValueError(f"Unknown interpolation: {interpolation}")

        segments.append(segment)

    return np.concatenate(segments)


def vibrato(
    duration: float,
    center_pitch: float,
    rate: float = 5.0,
    depth: float = 0.5,
    fps: int = FRAME_RATE,
) -> FloatArray:
    """Create vibrato (pitch oscillation).

    Args:
        duration: Duration in seconds
        center_pitch: Center pitch in MIDI
        rate: Vibrato rate in Hz
        depth: Vibrato depth in semitones
        fps: Frames per second

    Returns:
        Pitch curve with vibrato
    """
    t = get_time_array(duration, fps)
    w = rate * 2 * np.pi
    pitch_midi = center_pitch + depth * np.sin(w * t)

    return pitch_midi.astype("float32")


def glissando(
    duration: float,
    start_pitch: float,
    end_pitch: float,
    curve_type: str = "linear",
    fps: int = FRAME_RATE,
) -> FloatArray:
    """Create glissando (pitch sweep).

    Args:
        duration: Duration in seconds
        start_pitch: Starting pitch in MIDI
        end_pitch: Ending pitch in MIDI
        curve_type: 'linear', 'exponential', or 'logarithmic'
        fps: Frames per second

    Returns:
        Pitch curve
    """
    if curve_type == "linear":
        return linear_curve(duration, start_pitch, end_pitch, fps)
    elif curve_type == "exponential":
        t = get_time_array(duration, fps)
        norm_t = t / duration
        pitch = start_pitch + (end_pitch - start_pitch) * (norm_t**2)
        return pitch.astype("float32")
    elif curve_type == "logarithmic":
        t = get_time_array(duration, fps)
        norm_t = t / duration
        pitch = start_pitch + (end_pitch - start_pitch) * np.sqrt(norm_t)
        return pitch.astype("float32")
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")


def pitch_sequence(
    pitches: list[float],
    durations: list[float],
    fps: int = FRAME_RATE,
) -> FloatArray:
    """Create a sequence of pitches with durations.

    Args:
        pitches: List of MIDI pitches
        durations: List of durations for each pitch
        fps: Frames per second

    Returns:
        Pitch curve
    """
    segments = [constant_curve(d, p, fps) for p, d in zip(pitches, durations)]
    return np.concatenate(segments)


def pad_curve(
    curve: FloatArray,
    pad_duration: float,
    pad_value: float | None = None,
    fps: int = FRAME_RATE,
    end_duration_multiplier: float = PAD_END_DURATION_MULTIPLIER,
) -> FloatArray:
    """Pad a curve with constant values at beginning and end.

    Args:
        curve: Input curve
        pad_duration: Duration of padding in seconds
        pad_value: Padding value (uses last value if None)
        fps: Frames per second
        end_duration_multiplier: Multiplier for end padding duration

    Returns:
        Padded curve
    """
    if pad_value is None:
        pad_value = curve[-1]

    start_pad = constant_curve(pad_duration, pad_value, fps)
    end_pad = constant_curve(pad_duration * end_duration_multiplier, pad_value, fps)

    return np.concatenate([start_pad, curve, end_pad])


def midi_to_hz_curve(midi_curve: FloatArray) -> FloatArray:
    """Convert MIDI pitch curve to frequency in Hz.

    Args:
        midi_curve: Pitch curve in MIDI note numbers

    Returns:
        Frequency curve in Hz
    """
    return librosa.midi_to_hz(midi_curve).astype("float32")


def hz_to_midi_curve(hz_curve: FloatArray) -> FloatArray:
    """Convert frequency curve to MIDI pitch.

    Args:
        hz_curve: Frequency curve in Hz

    Returns:
        Pitch curve in MIDI note numbers
    """
    return librosa.hz_to_midi(hz_curve).astype("float32")


def db_to_amplitude(db_curve: FloatArray) -> FloatArray:
    """Convert dB curve to linear amplitude.

    Args:
        db_curve: Loudness in dB

    Returns:
        Linear amplitude
    """
    return librosa.db_to_amplitude(db_curve).astype("float32")


def amplitude_to_db(amp_curve: FloatArray) -> FloatArray:
    """Convert amplitude curve to dB.

    Args:
        amp_curve: Linear amplitude

    Returns:
        Loudness in dB
    """
    return librosa.amplitude_to_db(amp_curve).astype("float32")


class DDSPPhrase:
    """Container for DDSP control signals (f0, loudness, stretch)."""

    def __init__(
        self,
        f0: FloatArray,
        loudness: FloatArray,
        stretch: FloatArray,
    ):
        """Initialize DDSP phrase.

        Args:
            f0: Fundamental frequency in Hz over time
            loudness: Loudness in dB over time
            stretch: Spectral stretch factor over time
        """
        assert len(f0) == len(loudness) == len(stretch)

        self.f0 = f0.astype("float32")
        self.loudness = loudness.astype("float32")
        self.stretch = stretch.astype("float32")

    def __len__(self) -> int:
        """Get phrase length in frames."""
        return len(self.f0)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "f0": self.f0.tolist(),
            "loudness": self.loudness.tolist(),
            "stretch": self.stretch.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DDSPPhrase":
        """Create phrase from dictionary."""
        return cls(
            np.array(data["f0"]),
            np.array(data["loudness"]),
            np.array(data["stretch"]),
        )

    def render(self, model) -> FloatArray:
        """Render phrase to audio using DDSP model.

        Args:
            model: DDSP model

        Returns:
            Audio samples
        """
        # Pad for smooth onset/offset
        padded_f0 = pad_curve(self.f0, 2.0)
        padded_loudness = pad_curve(self.loudness, 2.0, pad_value=-110)
        padded_stretch = pad_curve(self.stretch, 2.0)

        return generate_audio_with_ddsp(
            model,
            padded_f0,
            padded_loudness,
            padded_stretch,
        )
