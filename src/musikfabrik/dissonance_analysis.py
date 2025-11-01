# pyright: basic
"""Dissonance curve analysis and scale generation utilities.

This module provides functions for:
- Calculating dissonance curves from partials
- Finding consonant intervals
- Generating scales from consonant intervals
"""

import itertools

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from musikfabrik.seth import dissonance, generate_partial_amps, generate_partial_ratios, sweep_partials

FloatArray = NDArray[np.float64]


def calculate_dissonance_curve(
    fixed_partials: FloatArray,
    fixed_amplitudes: FloatArray,
    swept_partials: FloatArray,
    swept_amplitudes: FloatArray,
    start_delta_cents: float = -100,
    end_delta_cents: float = 1300,
    cents_per_bin: float = 0.25,
) -> tuple[FloatArray, FloatArray]:
    """Calculate a dissonance curve by sweeping one set of partials against another.

    Args:
        fixed_partials: Frequencies of fixed sound partials
        fixed_amplitudes: Amplitudes of fixed sound partials
        swept_partials: Frequencies of partials to sweep
        swept_amplitudes: Amplitudes of swept partials
        start_delta_cents: Starting interval in cents
        end_delta_cents: Ending interval in cents
        cents_per_bin: Resolution in cents

    Returns:
        Tuple of (cents array, roughness/dissonance values)
    """
    swept = sweep_partials(
        swept_partials,
        start_delta_cents,
        end_delta_cents,
        cents_per_bin,
    )

    roughness = dissonance(
        fixed_partials,
        fixed_amplitudes,
        swept,
        swept_amplitudes,
    )

    num_points = np.round(
        (end_delta_cents - start_delta_cents) / cents_per_bin
    ).astype("int")

    cents = np.linspace(start_delta_cents, end_delta_cents, num_points)

    return cents, roughness


def generate_synthetic_partials(
    f0: float = 440.0,
    n_partials: int = 8,
    stretch_factor: float = 1.0,
    amp_decay_factor: float = 0.9,
    in_dbs: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Generate synthetic partials with exponential amplitude decay.

    Args:
        f0: Fundamental frequency
        n_partials: Number of partials to generate
        stretch_factor: Inharmonicity factor (1.0 = harmonic)
        amp_decay_factor: Amplitude decay per partial
        in_dbs: Apply A-weighting in dB scale

    Returns:
        Tuple of (partial frequencies, partial amplitudes)
    """
    partials = f0 * generate_partial_ratios(n_partials, stretch_factor)
    amplitudes = generate_partial_amps(1.0, n_partials, amp_decay_factor)

    if in_dbs:
        amplitudes = librosa.amplitude_to_db(amplitudes)
        amplitudes += librosa.A_weighting(partials, min_db=-180)
        amplitudes -= amplitudes.min()
        amp_max = amplitudes.max()
        if amp_max > 0:
            amplitudes /= amp_max

    return partials, amplitudes


def calculate_synthetic_dissonance_curve(
    f0: float = 440.0,
    n_partials: int = 8,
    stretch_factor_1: float = 1.05,
    stretch_factor_2: float = 1.0,
    amp_decay_factor: float = 0.9,
    in_dbs: bool = True,
    start_delta_cents: float = -100,
    end_delta_cents: float = 1300,
    cents_per_bin: float = 0.25,
) -> tuple[FloatArray, FloatArray]:
    """Calculate dissonance curve for two synthetic sounds.

    Args:
        f0: Fundamental frequency
        n_partials: Number of partials
        stretch_factor_1: Stretch factor for fixed sound
        stretch_factor_2: Stretch factor for swept sound
        amp_decay_factor: Amplitude decay factor
        in_dbs: Use dB scale with A-weighting
        start_delta_cents: Starting interval in cents
        end_delta_cents: Ending interval in cents
        cents_per_bin: Resolution in cents

    Returns:
        Tuple of (cents array, roughness values)
    """
    fixed_partials, fixed_amplitudes = generate_synthetic_partials(
        f0, n_partials, stretch_factor_1, amp_decay_factor, in_dbs
    )
    swept_partials, swept_amplitudes = generate_synthetic_partials(
        f0, n_partials, stretch_factor_2, amp_decay_factor, in_dbs
    )

    return calculate_dissonance_curve(
        fixed_partials,
        fixed_amplitudes,
        swept_partials,
        swept_amplitudes,
        start_delta_cents,
        end_delta_cents,
        cents_per_bin,
    )


def normalize(x: FloatArray) -> FloatArray:
    """Normalize array to [0, 1] range.
    
    Args:
        x: Array to normalize
        
    Returns:
        Normalized array, or zeros if input has no range
    """
    x = x - x.min()
    x_max = x.max()
    if x_max > 0:
        x = x / x_max
    return x


def find_consonant_intervals(
    dissonance_curve: FloatArray,
    deviation: float = 0.7,
    distance: float = 20.0,
) -> NDArray[np.int64]:
    """Find consonant intervals (local minima) in dissonance curve.

    Args:
        dissonance_curve: Dissonance/roughness values
        deviation: Standard deviation threshold for peak detection
        distance: Minimum distance between peaks in bins

    Returns:
        Array of peak indices
    """
    # Use second derivative to find local minima
    d2 = np.gradient(np.gradient(dissonance_curve))

    # Combine second derivative with inverted dissonance
    measure = np.minimum(normalize(d2), (1 - normalize(dissonance_curve)))

    peaks, _ = find_peaks(
        measure,
        height=measure.mean() + measure.std() * deviation,
        distance=distance,
    )

    return peaks


def calculate_dissonance_from_partials_dict(
    partials_dict: dict,
    start_delta_cents: float = -100,
    end_delta_cents: float = 1300,
    cents_per_bin: float = 0.25,
) -> tuple[FloatArray, FloatArray]:
    """Calculate dissonance curve from a partials dictionary.

    Args:
        partials_dict: Dictionary with 'frequencies' and 'amplitudes' keys
        start_delta_cents: Starting interval in cents
        end_delta_cents: Ending interval in cents
        cents_per_bin: Resolution in cents

    Returns:
        Tuple of (cents array, roughness values)
    """
    freqs = partials_dict["frequencies"]
    amps = partials_dict["amplitudes"]

    return calculate_dissonance_curve(
        freqs,
        amps,
        freqs,
        amps,
        start_delta_cents,
        end_delta_cents,
        cents_per_bin,
    )


def _validate_scale(scale: list[int], intervals: list[list[int]]) -> bool:
    """Check if scale satisfies interval constraints.
    
    Args:
        scale: Scale to validate
        intervals: List of interval groups (must have 8 elements)
        
    Returns:
        True if scale is valid, False otherwise
        
    Raises:
        ValueError: If intervals list doesn't have 8 elements
    """
    if len(intervals) != 8:
        raise ValueError(f"Expected 8 interval groups, got {len(intervals)}")
    
    octave = intervals[7][0]
    # Create extended scale for interval checking (avoid modifying input)
    extended_scale = scale[:-1] + [scale[-1] + item for item in scale]

    for interval in range(1, 8):
        for idx in range(len(extended_scale) - interval):
            distance = (extended_scale[idx + interval] - extended_scale[idx]) % octave

            # Handle special cases for 4ths and 5ths
            if interval == 3:
                if len(intervals[interval - 1]) == 0 or len(intervals[interval + 1]) <= 1:
                    continue
                lower_limit = max(intervals[interval - 1])
                upper_limit = min(intervals[interval + 1][1:])
            elif interval == 4:
                if len(intervals[interval - 1]) <= 1 or len(intervals[interval + 1]) == 0:
                    continue
                lower_limit = max(intervals[interval - 1][:-1])
                upper_limit = min(intervals[interval + 1])
            elif interval == 7:
                if distance != 0:
                    return False
                return True
            else:
                if len(intervals[interval - 1]) == 0 or len(intervals[interval + 1]) == 0:
                    continue
                lower_limit = max(intervals[interval - 1])
                upper_limit = min(intervals[interval + 1])

            if (distance <= lower_limit) or (distance >= upper_limit):
                return False
    return True


def generate_scales_from_intervals(
    intervals: list[list[int]],
) -> list[list[int]]:
    """Generate valid scales from consonant interval groups.

    This is a simplified version from the existing generate_scales.py.
    For full scale generation with constraints, use that module.

    Args:
        intervals: List of interval groups (in cents)

    Returns:
        List of valid scales
    """
    all_scales = itertools.product(*[i for i in intervals])

    valid_scales = []
    for scale in all_scales:
        scale_list = list(scale)
        if _validate_scale(scale_list, intervals):
            valid_scales.append(scale_list)

    # Remove duplicates
    unique_scales = [
        list(t_scale) for t_scale in set(tuple(scale) for scale in valid_scales)
    ]

    return unique_scales


def rotate_scale(scale: list[int], n: int = 1) -> list[int]:
    """Rotate a scale to create a mode.

    Args:
        scale: Scale as list of intervals
        n: Number of positions to rotate

    Returns:
        Rotated scale
    """
    scale_arr = np.array(scale)
    deltas = scale_arr[1:] - scale_arr[:-1]
    rotated_deltas = np.concatenate([deltas[-n:], deltas[:-n]])
    rotated_scale = np.concatenate([[0], np.cumsum(rotated_deltas)])

    return list(rotated_scale)


def get_all_modes(scales: list[list[int]]) -> list[list[int]]:
    """Generate all modes (rotations) from a list of scales.

    Args:
        scales: List of scales

    Returns:
        List of unique modes
    """
    modes = []

    for scale in scales:
        modes.append(scale)
        for i in range(1, len(scale) - 1):
            modes.append(rotate_scale(scale, i))

    # Remove duplicates
    unique_modes = [list(t) for t in set(tuple(m) for m in modes)]

    return unique_modes
