# pyright: standard

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")

with app.setup:
    from dataclasses import dataclass

    import librosa
    import marimo as mo
    import numpy as np
    from matplotlib.figure import Figure
    from scipy.interpolate import interp1d
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    import torch

    from musikfabrik.seth import (
        dissonance,
        generate_partial_amps,
        generate_partial_ratios,
        sweep_partials,
    )
    from performer.utils.constants import (
        FRAME_RATE,
        HOP_LENGTH,
        N_FFT,
        SAMPLE_RATE,
    )
    from performer.utils.features import Loudness

    # TODO: disable backprop when using this
    loudness_detector = Loudness()


@app.function
def load_sample(file_path, sr=SAMPLE_RATE, is_mono=True, is_normalized=True):
    sample, _ = librosa.load(file_path, sr=sr, mono=is_mono)
    if is_normalized:
        sample /= np.abs(sample).max()
    return sample


@app.function
def get_dynamic_pitch(
    sample, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
):
    f0 = librosa.yin(
        sample,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    return f0


@app.function
def get_dynamic_loudness(sample):
    return (
        loudness_detector.get_amp(
            torch.from_numpy(sample).unsqueeze(0).unsqueeze(0)
        )
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )


@app.function
def get_spectrum(
    sample,
    sr=SAMPLE_RATE,
    in_dbs=True,
    A_weighted=True,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
):
    if not in_dbs and A_weighted:
        raise ValueError("A-weighting only makes sense in dB scale.")

    frequencies = (np.fft.rfftfreq(sample.shape[0]) * sr)[1:]
    amplitudes = np.abs(np.fft.rfft(sample, norm="forward"))[1:]
    # set DC offset to 0
    # amplitudes[0] = 0.0

    if in_dbs:
        amplitudes = librosa.amplitude_to_db(amplitudes)
        if A_weighted:
            amplitudes += librosa.A_weighting(frequencies, min_db=-70)

    amplitudes -= amplitudes.min()
    amplitudes /= amplitudes.max()

    return frequencies, amplitudes


@app.function
def get_partial_indices(
    freqs,
    amps,
    height=0.9,
    distance=25.0,
    min_f=25.0,
    max_f=24000.0,
    filter_range=25.0,
):
    # One bin equals to this many Hz
    unit = freqs[1] - freqs[0]
    idx = (freqs > max_f) | (freqs < min_f)
    amps[idx] = 0.0

    # Hz to bin
    filter_bins = int(np.round(filter_range / unit))
    distance_bins = int(np.round(distance / unit))

    # THIS HELPS
    noise_floor = median_filter(amps, size=filter_bins)
    amps = amps - noise_floor
    amps /= amps.max()

    peaks, _ = find_peaks(amps, distance=distance_bins, height=height)

    # We convert peaks into original indices to see the filtered out parts of the spectrum
    return peaks, amps


@app.function
def get_loudest_n(amps, peaks, n=8):
    return sorted(peaks[np.flip(np.argsort(amps[peaks]))][:n])


@app.function
def draw_overtone_curve(freqs, amps, peaks):
    fig = Figure(figsize=(12, 5), dpi=300)
    ax = fig.add_axes((0.05, 0.15, 0.9, 0.8))

    ax.set_xlim(freqs[min(peaks)] - 50, freqs[max(peaks)] + 50)
    ax.scatter(freqs[peaks], amps[peaks], color="red")
    ax.plot(freqs, amps)

    for xii in freqs[peaks]:
        ax.axvline(x=xii, color="b", linestyle="-", alpha=0.3)

    ax.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

    ax.set_xlabel("overtone ratio")
    ax.set_ylabel("amplitude")

    ax.set_xticks(
        freqs[peaks],
        [f"{t / freqs[peaks[0]]:.2f}" for t in freqs[peaks]],
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    return fig


@app.cell(hide_code=True)
def _():
    # choose one of these samples
    SAMPLE_PATHS = [
        "data/samples/instruments/brass/Horn/ordinario/Hn-ord-C5-ff-N-N.wav",
        "data/samples/instruments/brass/Horn/ordinario/Hn-ord-C2-ff-N-N.wav",
        "data/samples/instruments/brass/Horn/flatterzunge/Hn-flatt-F2-ff-N-N.wav",
        "data/samples/bells/42095__fauxpress__bell-meditation.mp3",
        "data/samples/bells/62964__ladycailin__deep-bell.wav",
        "data/samples/bells/124755__tec_studio__mono_bell_0_d_16sec.wav",
    ]
    sample_selection = mo.ui.dropdown(
        SAMPLE_PATHS,
        value=SAMPLE_PATHS[1],
        allow_select_none=False,
        label="Select Sample: ",
    )
    in_dbs = mo.ui.checkbox(True, label="In dB Scale")
    A_weighted = mo.ui.checkbox(True, label="A-weighted Spectrum")
    height = mo.ui.slider(
        0.0,
        1.0,
        value=0.1,
        step=0.01,
        show_value=True,
        label="Peak Height Threshold",
    )
    distance = mo.ui.slider(
        steps=np.logspace(np.log2(4.0), np.log2(1024.0), num=100, base=2.0),
        value=np.logspace(np.log2(4.0), np.log2(1024.0), num=100, base=2.0)[0],
        show_value=True,
        label="Peak Minimum Distance (Hz)",
    )
    min_f = mo.ui.slider(
        steps=np.logspace(np.log2(25.0), np.log2(1024.0), num=100, base=2.0),
        value=np.logspace(np.log2(25.0), np.log2(1024.0), num=100, base=2.0)[0],
        show_value=True,
        label="Minimum Frequency (Hz)",
    )
    max_f = mo.ui.slider(
        steps=np.logspace(np.log2(6000.0), np.log2(24000.0), num=100, base=2.0),
        value=np.logspace(np.log2(6000.0), np.log2(24000.0), num=100, base=2.0)[
            -1
        ],
        show_value=True,
        label="Maximum Frequency (Hz)",
    )
    filter_range = mo.ui.slider(
        steps=np.logspace(np.log2(4.0), np.log2(1024.0), num=100, base=2.0),
        value=np.logspace(np.log2(4.0), np.log2(1024.0), num=100, base=2.0)[0],
        show_value=True,
        label="Median Filter Range (Hz)",
    )
    n_partials = mo.ui.slider(
        1,
        20,
        value=8,
        step=1,
        show_value=True,
        label="Number of Partials to Extract",
    )
    f_zero = mo.ui.slider(
        steps=np.logspace(np.log2(4.0), np.log2(1024.0), num=100, base=2.0),
        value=np.logspace(np.log2(4.0), np.log2(1024.0), num=100, base=2.0)[0],
        show_value=True,
        include_input=True,
        label="F0 (Hz)",
    )
    mo.vstack(
        [
            sample_selection,
            mo.hstack(
                [
                    in_dbs,
                    A_weighted,
                ]
            ),
            mo.hstack(
                [
                    height,
                    distance,
                ]
            ),
            mo.hstack(
                [
                    min_f,
                    max_f,
                ]
            ),
            mo.hstack(
                [
                    filter_range,
                    n_partials,
                ]
            ),
            f_zero,
        ]
    )
    return (
        SAMPLE_PATHS,
        f_zero,
        filter_range,
        max_f,
        n_partials,
        sample_selection,
    )


@app.cell
def _(f_zero, filter_range, max_f, n_partials, sample_selection):
    # WARNING: Almost all slider values are currently overriden because I'm trying to minimize
    # the number of manually tuned parameters.
    # We need to add and additive synthesis section next to compare the results.
    # Amplitude works better than dB.
    # just setting min_f works well with the following calculations.
    # Look at the lowest frequency partial and set min_f to a bit lower than that.
    # Take first n partials rather than the loudest n partials.
    # Also plot the entire amplitude spectrum. Get the frequency of the first partial we are interested in.
    # Set f0 to a bit lower than that.
    # Create a form, where submit button saves f0, amplitudes, overtone ratios, and sample path to a json file.

    static_sample = load_sample(sample_selection.value)
    freqs, amps_ = get_spectrum(static_sample, in_dbs=False, A_weighted=False)
    filter_range
    peak_idx, amps = get_partial_indices(
        freqs,
        amps_,
        height=0.025,  # height.value,
        distance=f_zero.value / 4,  # distance.value,
        min_f=f_zero.value * (8 / 9),  # min_f.value,
        max_f=max_f.value,
        filter_range=f_zero.value * n_partials.value,  # filter_range.value,
    )
    # peak_idx = get_loudest_n(amps, peak_idx_, n=n_partials.value)
    fig = draw_overtone_curve(freqs, amps, peak_idx[: n_partials.value])

    time = np.linspace(0.0, 5.0, num=SAMPLE_RATE * 5)
    synthesis = np.zeros_like(time)

    for f, a in zip(freqs[peak_idx], amps_[peak_idx]):
        synthesis += a * np.sin(2 * np.pi * f * time)
    synthesis /= np.abs(synthesis).max()

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(
                        f"{freqs[peak_idx[0]]:.2f} ({librosa.hz_to_note(freqs[peak_idx[0]])})"
                    ),
                    mo.vstack(
                        [
                            mo.audio(static_sample, SAMPLE_RATE),
                            mo.audio(synthesis, SAMPLE_RATE),
                        ]
                    ),
                ]
            ),
            fig,
        ]
    )
    return (static_sample,)


@app.function
def f0_loudness_plot(f0, loudness):
    # create the figure and twin axes
    fig = Figure(figsize=(12, 5), dpi=300)
    ax1 = fig.add_axes((0.05, 0.15, 0.9, 0.8))
    ax2 = ax1.twinx()

    # prepare x-axis
    t = np.linspace(0, len(f0) * HOP_LENGTH / SAMPLE_RATE, num=len(f0))
    f0 = librosa.hz_to_midi(f0)

    # keep track of plots
    plots = []

    # plot on the left y-axis
    plots.append(ax1.plot(t, f0, "b-", label="f0 data"))
    ax1.set_xlabel("time (seconds)")
    ax1.set_ylabel("f0 (MIDI #)", color="b")
    ax1.tick_params(axis="y", colors="b")
    ax1.set_ylim(f0.mean() - 2, f0.mean() + 2)

    # plot on the left y-axis
    plots.append(ax2.plot(t, loudness, "r-", label="loudness data"))
    ax2.set_ylabel("Loudness (dB)", color="r")
    ax2.tick_params(axis="y", colors="r")

    # put labels
    labels = [l[0].get_label() for l in plots]
    ax1.legend([p[0] for p in plots], labels, loc="lower left")

    return fig


@app.cell
def _(SAMPLE_PATHS, static_sample):
    dynamic_sample = load_sample(
        SAMPLE_PATHS[0], is_mono=True, is_normalized=False
    )
    f0 = get_dynamic_pitch(static_sample)
    loudness = get_dynamic_loudness(static_sample)
    return f0, loudness


@app.cell
def _(f0, loudness):
    f0_loudness_plot(f0, loudness)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
