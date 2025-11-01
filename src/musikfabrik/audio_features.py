# pyright: basic
"""Audio feature extraction utilities for musikfabrik project.

This module provides functions for extracting various audio features including:
- Static partials from audio files (FFT-based)
- Dynamic f0 (fundamental frequency) over time (STFT-based)
- Dynamic loudness over time
- Spectral stretching
"""

import librosa
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from performer.utils.features import Loudness

FloatArray = NDArray[np.float64]


def load_audio(
    file_path: str,
    sr: int = 48000,
    mono: bool = True,
    normalize: bool = True,
) -> tuple[FloatArray, int]:
    """Load an audio file.

    Args:
        file_path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono if True
        normalize: Normalize amplitude to [-1, 1] if True

    Returns:
        Tuple of (audio samples, sample rate)
    """
    sample, sample_rate = librosa.load(file_path, sr=sr, mono=mono)
    if normalize:
        max_val = np.abs(sample).max()
        if max_val > 0:
            sample = sample / max_val
    return sample, sample_rate


def get_static_spectrum(
    sample: FloatArray,
    sr: int = 48000,
    in_dbs: bool = True,
    a_weighted: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Extract frequency spectrum using FFT (static, entire audio file).

    Args:
        sample: Audio samples
        sr: Sample rate
        in_dbs: Return amplitudes in dB scale
        a_weighted: Apply A-weighting to amplitudes (only with in_dbs=True)

    Returns:
        Tuple of (frequencies, amplitudes)
    """
    if not in_dbs and a_weighted:
        raise ValueError("A-weighting only makes sense in dB scale.")

    # Remove DC offset by skipping index 0
    frequencies = (np.fft.rfftfreq(sample.shape[0]) * sr)[1:]
    amplitudes = np.abs(np.fft.rfft(sample, norm="forward"))[1:]

    if in_dbs:
        amplitudes = librosa.amplitude_to_db(amplitudes)
        if a_weighted:
            amplitudes += librosa.A_weighting(frequencies, min_db=-70)

    # Normalize to [0, 1]
    amplitudes -= amplitudes.min()
    max_val = amplitudes.max()
    if max_val > 0:
        amplitudes /= max_val

    return frequencies, amplitudes


def find_partial_peaks(
    freqs: FloatArray,
    amps: FloatArray,
    height: float = 0.9,
    distance: float = 25.0,
    min_f: float = 25.0,
    max_f: float = 24000.0,
    filter_range: float = 25.0,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Find peaks in frequency spectrum representing partials.

    Args:
        freqs: Frequency array
        amps: Amplitude array
        height: Minimum peak height (0-1)
        distance: Minimum distance between peaks in Hz
        min_f: Minimum frequency to consider
        max_f: Maximum frequency to consider
        filter_range: Median filter range in Hz for noise reduction

    Returns:
        Tuple of (peak indices, filtered amplitudes)
    """
    # One bin equals to this many Hz
    unit = freqs[1] - freqs[0]

    # Zero out frequencies outside range
    amps = amps.copy()
    idx = (freqs > max_f) | (freqs < min_f)
    amps[idx] = 0.0

    # Hz to bin
    filter_bins = int(np.round(filter_range / unit))
    distance_bins = int(np.round(distance / unit))

    # Reduce noise floor
    noise_floor = median_filter(amps, size=filter_bins)
    amps = amps - noise_floor
    amps /= amps.max()

    peaks, _ = find_peaks(amps, distance=distance_bins, height=height)

    return peaks, amps


def get_loudest_n_partials(
    amps: FloatArray,
    peaks: NDArray[np.int64],
    n: int = 8,
) -> list[int]:
    """Get the n loudest partials from detected peaks.

    Args:
        amps: Amplitude array
        peaks: Peak indices
        n: Number of partials to return

    Returns:
        List of peak indices sorted by original position
    """
    return sorted(peaks[np.flip(np.argsort(amps[peaks]))][:n])


def extract_partials(
    sample: FloatArray,
    sr: int = 48000,
    n_partials: int = 8,
    height: float = 0.025,
    f0_estimate: float | None = None,
    in_dbs: bool = False,
    a_weighted: bool = False,
) -> dict:
    """Extract partial frequencies and amplitudes from audio sample.

    Args:
        sample: Audio samples
        sr: Sample rate
        n_partials: Number of partials to extract
        height: Minimum peak height threshold
        f0_estimate: Estimate of fundamental frequency (auto-detected if None)
        in_dbs: Use dB scale for amplitudes
        a_weighted: Apply A-weighting

    Returns:
        Dictionary with keys:
        - 'f0': Fundamental frequency
        - 'frequencies': Partial frequencies
        - 'amplitudes': Partial amplitudes
        - 'ratios': Frequency ratios relative to f0
    """
    freqs, amps = get_static_spectrum(sample, sr, in_dbs, a_weighted)

    # Auto-detect f0 if not provided
    if f0_estimate is None:
        # Use first peak as f0 estimate
        initial_peaks, _ = find_partial_peaks(
            freqs, amps, height=height, distance=20.0, min_f=25.0
        )
        if len(initial_peaks) == 0:
            raise ValueError("No partials detected. Try adjusting parameters.")
        f0_estimate = freqs[initial_peaks[0]]

    # Find peaks with f0-based parameters
    peak_idx, filtered_amps = find_partial_peaks(
        freqs,
        amps,
        height=height,
        distance=f0_estimate / 4,
        min_f=f0_estimate * (8 / 9),
        max_f=24000.0,
        filter_range=f0_estimate * n_partials,
    )

    # Get first n partials
    peak_idx = peak_idx[:n_partials]

    partial_freqs = freqs[peak_idx]
    partial_amps = amps[peak_idx]
    f0 = partial_freqs[0]
    ratios = partial_freqs / f0

    return {
        "f0": float(f0),
        "frequencies": partial_freqs,
        "amplitudes": partial_amps,
        "ratios": ratios,
    }


def get_dynamic_f0(
    sample: FloatArray,
    sr: int = 48000,
    n_fft: int = 8192,
    hop_length: int = 512,
    fmin: float | None = None,
    fmax: float | None = None,
) -> FloatArray:
    """Extract time-varying fundamental frequency using YIN algorithm.

    Args:
        sample: Audio samples
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        fmin: Minimum f0 to detect (default: C2)
        fmax: Maximum f0 to detect (default: C7)

    Returns:
        Array of f0 values over time
    """
    if fmin is None:
        fmin = float(librosa.note_to_hz("C2"))
    if fmax is None:
        fmax = float(librosa.note_to_hz("C7"))

    f0 = librosa.yin(
        sample,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    return f0


def get_dynamic_loudness(
    sample: FloatArray,
    sr: int = 48000,
    n_fft: int = 8192,
    hop_length: int = 512,
) -> FloatArray:
    """Extract time-varying loudness using A-weighted STFT.

    Args:
        sample: Audio samples
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT

    Returns:
        Array of loudness values (in dB) over time
    """
    loudness_detector = Loudness()

    with torch.no_grad():
        loudness = (
            loudness_detector.get_amp(
                torch.from_numpy(sample).unsqueeze(0).unsqueeze(0)
            )
            .squeeze(0)
            .squeeze(0)
            .numpy()
        )

    return loudness


def stretch_spectrum(
    sample: FloatArray,
    f0: FloatArray,
    stretch_factors: FloatArray,
    sr: int = 48000,
    n_fft: int = 8192,
    hop_length: int = 512,
) -> FloatArray:
    """Apply time-varying spectral stretching to audio.

    Args:
        sample: Audio samples
        f0: Fundamental frequency over time
        stretch_factors: Stretch factor over time (1.0 = no stretch)
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT

    Returns:
        Stretched audio samples
    """
    D = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    warped_spectrogram = np.zeros_like(magnitude)
    warped_phase = np.zeros_like(phase)

    for t in range(magnitude.shape[1]):
        F0 = f0[t]
        # Skip frames with invalid F0
        if F0 <= 0:
            warped_spectrogram[:, t] = magnitude[:, t]
            warped_phase[:, t] = phase[:, t]
            continue
            
        freqs_warped = F0 * (freqs / F0) ** stretch_factors[t]

        interp_func = interp1d(
            freqs_warped,
            magnitude[:, t],
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        warped_spectrogram[:, t] = interp_func(freqs)

        interp_func_phase = interp1d(
            freqs_warped,
            phase[:, t],
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        warped_phase[:, t] = interp_func_phase(freqs)

    warped_D = warped_spectrogram * np.exp(1j * warped_phase)
    warped_signal = librosa.istft(warped_D, hop_length=hop_length)

    return warped_signal


def create_stretch_curve(
    length: int,
    start_stretch: float = 1.0,
    peak_stretch: float = 1.05,
    end_stretch: float = 1.0,
    attack_frames: int = 100,
    release_frames: int = 100,
) -> FloatArray:
    """Create a stretch factor curve over time.

    Args:
        length: Total number of frames
        start_stretch: Initial stretch factor
        peak_stretch: Peak stretch factor
        end_stretch: Final stretch factor
        attack_frames: Number of frames for attack
        release_frames: Number of frames for release

    Returns:
        Array of stretch factors over time
    """
    sustain_frames = max(0, length - attack_frames - release_frames)

    attack = np.linspace(start_stretch, peak_stretch, attack_frames)
    sustain = np.full(sustain_frames, peak_stretch)
    release = np.linspace(peak_stretch, end_stretch, release_frames)

    return np.concatenate([attack, sustain, release])
