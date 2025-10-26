# pyright: basic

import einx as ex
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

FloatArray = NDArray[np.float64]


def generate_partial_ratios(
    n_partials: int = 16, stretch_factor: float = 1.0, step_size=1
) -> FloatArray:
    ratios = (
        np.arange(1, (n_partials * step_size) + 1, step_size, dtype=np.float64)
        ** stretch_factor
    )

    return ratios


def generate_partial_amps(
    amp0: float, n_partials: int, decay_factor: float = 0.88
) -> FloatArray:
    factors = decay_factor ** np.arange(n_partials, dtype=np.float64)
    amps = amp0 * factors

    return amps


def sweep_partials(
    partial_freqs: FloatArray,
    start_delta_cents: float,
    end_delta_cents: float,
    resolution: float = 0.25,
) -> FloatArray:
    num_points = round((end_delta_cents - start_delta_cents) / resolution)
    sweep_range_cents = np.linspace(start_delta_cents, end_delta_cents, num_points)
    sweep_range_ratios = 2 ** (sweep_range_cents / 1200)
    swept_partials = ex.multiply("a, ... -> a ...", sweep_range_ratios, partial_freqs)

    return swept_partials


def align_for_broadcast(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    n = a.shape[-1]
    m = b.shape[-1]

    a_batch = a.shape[:-1]
    b_batch = b.shape[:-1]

    na, nb = len(a_batch), len(b_batch)

    a_new_shape = a_batch + (1,) * nb + (n,)
    b_new_shape = (1,) * na + b_batch + (m,)

    return a.reshape(a_new_shape), b.reshape(b_new_shape)


# WARNING: AI Generated code! Study and refactor if necessary


def _get_pairs(a: FloatArray, b: FloatArray) -> FloatArray:
    """
    For inputs:
        a: (..., n)
        b: (..., n)  # potentially with completely different batch shapes,
                      # but broadcast-compatible on the batch axes

    Returns:
        pairs: (broadcasted_batch_shape..., num_pairs, 2)

    Each pair is all unordered pairs (upper triangle, k=1) formed from
    concatenated vectors along the last axis.
    """
    n = a.shape[-1]
    m = b.shape[-1]
    N = n + m
    idx0, idx1 = np.triu_indices(N, k=1)

    # Compute broadcasted batch shape
    a_batch = a.shape[:-1]
    b_batch = b.shape[:-1]
    out_batch = np.broadcast_shapes(a_batch, b_batch)
    a_bc = np.broadcast_to(a, out_batch + (n,))
    b_bc = np.broadcast_to(b, out_batch + (m,))
    combined = np.concatenate([a_bc, b_bc], axis=-1)  # (..., 2n)
    # Now, for each batch, select all pair indices
    pairs0 = np.take(combined, idx0, axis=-1)  # (..., num_pairs)
    pairs1 = np.take(combined, idx1, axis=-1)  # (..., num_pairs)
    pairs = np.stack([pairs0, pairs1], axis=-1)  # (..., num_pairs, 2)
    return pairs


# ======= END OF AI Generated Code =======


def get_pairs(a: FloatArray, b: FloatArray) -> FloatArray:
    aligned_a, aligned_b = align_for_broadcast(a, b)
    pairs = _get_pairs(aligned_a, aligned_b)

    return pairs


def __f_dissonance(f_min: FloatArray, f_max: FloatArray) -> FloatArray:
    # Used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # Point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96

    C1 = 5
    C2 = -5

    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75

    S = Dstar / (S1 * f_min + S2)
    Fdif = f_max - f_min

    SFdif = S * Fdif
    D = C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)

    return D


def _f_dissonance(freq_pairs: FloatArray, axis: int = -1) -> FloatArray:
    f_min, f_max = np.min(freq_pairs, axis=axis), np.max(freq_pairs, axis=axis)

    return __f_dissonance(f_min, f_max)


def _dissonance(
    partial_pairs: FloatArray,
    amplitude_pairs: FloatArray,
    axis: int = -1,
) -> FloatArray:
    f = ex.reduce("a... 2 -> a...", partial_pairs, op=_f_dissonance)
    a = np.min(amplitude_pairs, axis=axis)
    aligned_f, aligned_a = align_for_broadcast(f, a)
    D = np.sum(aligned_a * aligned_f, axis=axis)

    return D


# TODO: We shouldn't include frequencies above Nyquist
def dissonance(
    partials: FloatArray,
    amplitudes: FloatArray,
    other_partials: FloatArray,
    other_amplitudes: FloatArray,
    axis: int = -1,
) -> FloatArray:
    partial_pairs = get_pairs(partials, other_partials)
    amplitude_pairs = get_pairs(amplitudes, other_amplitudes)

    result = _dissonance(partial_pairs, amplitude_pairs, axis)

    return result


# TODO: Use a combination of the 2nd derivative and dissonance value.
# Higher octaves are more consonant than lower octaves but they are
# not selected because their 2nd derivaties are relatively smaller
# but their dissonance values are way lower.
def get_peaks(x_axis: FloatArray, curve: FloatArray, height: float = 0.2):
    second_derivative = np.gradient(np.gradient(curve, x_axis), x_axis)
    # second_derivative /= curve
    second_derivative -= second_derivative.min()
    second_derivative /= second_derivative.max()
    # second_derivative = np.log(second_derivative) + 3.1
    # second_derivative /= 3.1

    dpeaks, _ = find_peaks(second_derivative, height=height)

    return dpeaks, second_derivative
