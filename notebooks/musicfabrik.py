# pyright: basic

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")

with app.setup:
    import librosa
    import marimo as mo
    import numpy as np
    from matplotlib.figure import Figure
    from scipy.interpolate import interp1d
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    from musikfabrik.seth import (
        dissonance,
        generate_partial_amps,
        generate_partial_ratios,
        sweep_partials,
    )

    FPS = 250
    SAMPLE_RATE = 48000


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Tools for Musikfabrik Piece and Thesis Idea""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Section for harmonic stretching any monophonic audio""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Load audio file and extract features""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// attention | TODO:

    Also get loudness in db so we can use it to generate DDSP sounds from audio files
    ///
    """
    )
    return


@app.function
def get_pitch(sample, sr=SAMPLE_RATE, n_fft=8192, hop_length=512):
    f0 = librosa.yin(
        sample,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    return f0


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Define stretch in time""")
    return


@app.function
def get_stretch_curve(length, strch=1.05):
    stretches = np.concatenate(
        [
            np.linspace(1.0, strch, 100),
            np.linspace(strch, strch, length - 200),
            np.linspace(strch, 1.0, 100),
        ]
    )

    return stretches


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Stretch the spectrum and convert back to audio""")
    return


@app.function
def stretch_sample(
    sample, f0, stretches, sr=SAMPLE_RATE, n_fft=8192, hop_length=512
):
    D = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    warped_spectrogram = np.zeros_like(magnitude)
    warped_phase = np.zeros_like(phase)
    for t in range(magnitude.shape[1]):
        F0 = f0[t]
        freqs_warped = F0 * (freqs / F0) ** stretches[t]
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


@app.cell
def _():
    def play_spectrum_stretch_results():
        SAMPLE_PATH = (
            "data/samples/instruments/brass/Horn/ordinario/Hn-ord-C2-ff-N-N.wav"
        )
        sample, _ = librosa.load(SAMPLE_PATH, sr=SAMPLE_RATE)
        f0 = get_pitch(sample)
        stretch_curve = get_stretch_curve(len(f0), 1.05)
        stretched_sample = stretch_sample(sample, f0, stretch_curve)

        return mo.vstack(
            [
                mo.md("### Spectrum stretching results"),
                mo.audio(src=stretched_sample, rate=SAMPLE_RATE),
                mo.audio(src=sample, rate=SAMPLE_RATE),
            ]
        )


    play_spectrum_stretch_results()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Getting consonant intervals from synthetic sounds""")
    return


@app.function
def get_dissonance_curve(
    fixed_partials,
    fixed_amplitudes,
    partials_to_be_swept,
    amplitudes_of_swept,
    start_delta_cents=-100,
    end_delta_cents=1300,
    cents_per_bin=0.25,
):
    swept_partials = sweep_partials(
        partials_to_be_swept,
        start_delta_cents,
        end_delta_cents,
        cents_per_bin,
    )

    roughness = dissonance(
        fixed_partials, fixed_amplitudes, swept_partials, amplitudes_of_swept
    )

    num_points = np.round(
        (end_delta_cents - start_delta_cents) / cents_per_bin
    ).astype("int")

    cents = np.linspace(start_delta_cents, end_delta_cents, num_points)

    return cents, roughness


@app.function
def get_synthetic_dissonance_curve(
    f0=440.0,
    n_partials=8,
    stretch_factor_1=1.05,
    stretch_factor_2=1.0,
    amp_decay_factor=0.9,
    in_dbs=True,
    start_delta_cents=-100,
    end_delta_cents=1300,
    cents_per_bin=0.25,
):
    fixed_partials = f0 * generate_partial_ratios(n_partials, stretch_factor_1)
    partials_to_be_swept = f0 * generate_partial_ratios(
        n_partials, stretch_factor_2
    )

    fixed_amplitudes = generate_partial_amps(1.0, n_partials, amp_decay_factor)
    amplitudes_of_swept = generate_partial_amps(
        1.0, n_partials, amp_decay_factor
    )

    if in_dbs:
        fixed_amplitudes = librosa.amplitude_to_db(fixed_amplitudes)
        fixed_amplitudes += librosa.A_weighting(fixed_partials, min_db=-180)
        fixed_amplitudes -= fixed_amplitudes.min()
        fixed_amplitudes /= fixed_amplitudes.max()

        amplitudes_of_swept = librosa.amplitude_to_db(amplitudes_of_swept)
        amplitudes_of_swept += librosa.A_weighting(
            partials_to_be_swept, min_db=-180
        )
        amplitudes_of_swept -= amplitudes_of_swept.min()
        amplitudes_of_swept /= amplitudes_of_swept.max()

    cents, roughness = get_dissonance_curve(
        fixed_partials,
        fixed_amplitudes,
        partials_to_be_swept,
        fixed_amplitudes,
        start_delta_cents,
        end_delta_cents,
        cents_per_bin,
    )

    return cents, roughness


@app.function
def normalize(x):
    x -= x.min()
    x /= x.max()
    return x


@app.function
def get_consonant_intervals(dissonance_curve, deviation=0.7, distance=20.0):
    d2 = np.gradient(np.gradient(dissonance_curve))
    measure = np.minimum(normalize(d2), (1 - normalize(dissonance_curve)))
    peaks, _ = find_peaks(
        measure,
        height=measure.mean() + measure.std() * deviation,
        distance=distance,
    )

    return peaks


@app.function
def draw_dissonance_curve(cents_axis, roughness, peaks):
    fig = Figure(figsize=(12, 5), dpi=300)
    ax = fig.add_axes((0.05, 0.15, 0.9, 0.8))
    ax.scatter(cents_axis[peaks], roughness[peaks], color="red")
    ax.plot(cents_axis, roughness)

    for xii in cents_axis[peaks]:
        ax.axvline(x=xii, color="b", linestyle="-", alpha=0.3)

    ax.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

    ax.set_xlabel("interval in cents")
    ax.set_ylabel("sensory dissonance")

    ax.set_xticks(
        cents_axis[peaks],
        [f"{int(np.round(t))}" for t in cents_axis[peaks]],
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    return fig


@app.cell(hide_code=True)
def _():
    start_slider = mo.ui.slider(
        start=-300,
        stop=300,
        step=1,
        value=-100,
        show_value=True,
        include_input=True,
        label="Start",
    )
    end_slider = mo.ui.slider(
        start=-2400,
        stop=2400,
        step=1,
        value=1300,
        show_value=True,
        include_input=True,
        label="End",
    )

    deviation_slider = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.01,
        value=1.0,
        show_value=True,
        include_input=True,
        label="Deviation",
    )
    distance_slider = mo.ui.slider(
        start=0.0,
        stop=200.0,
        step=0.25,
        value=20.0,
        show_value=True,
        include_input=True,
        label="Distance",
    )

    mo.hstack(
        [
            mo.vstack(
                [
                    start_slider,
                    end_slider,
                ]
            ),
            mo.vstack(
                [
                    deviation_slider,
                    distance_slider,
                ]
            ),
        ]
    )
    return deviation_slider, distance_slider, end_slider, start_slider


@app.cell(hide_code=True)
def _(deviation_slider, distance_slider, end_slider, start_slider):
    def generate_example_curve():
        cents_per_bin = 0.25

        cents_axis, roughness = get_synthetic_dissonance_curve(
            librosa.note_to_hz("C4"),
            in_dbs=True,
            start_delta_cents=start_slider.value,
            end_delta_cents=end_slider.value,
            cents_per_bin=cents_per_bin,
        )
        peaks = get_consonant_intervals(
            roughness,
            deviation=deviation_slider.value,
            distance=distance_slider.value,
        )

        return cents_axis, roughness, peaks


    mo.vstack(
        [
            mo.md("### Dissonance curve for two synthetic, harmonic sounds"),
            draw_dissonance_curve(*generate_example_curve()),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Getting consonant intervals from real sounds""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// attention | Note:

    Multiply `np.abs(np.fft.rfft(signal,norm="forward")` output with `2` to get the right overtone amplitude value.
    ///
    """
    )
    return


@app.function
def get_spectrum(
    sample, sr=SAMPLE_RATE, in_dbs=True, n_fft=8192, hop_length=512
):
    frequencies = np.fft.rfftfreq(sample.shape[0]) * sr
    amplitudes = np.abs(np.fft.rfft(sample, norm="forward"))
    if in_dbs:
        amplitudes = librosa.amplitude_to_db(amplitudes)
        amplitudes += librosa.A_weighting(frequencies, min_db=-70)
        amplitudes -= amplitudes.min()
        amplitudes /= amplitudes.max()

    return frequencies, amplitudes


@app.function
def auto_partial_peaks(
    freqs, amps, height=0.9, distance=25.0, min_f=25.0, max_f=24000.0
):
    # TODO: use min/max f, min distance, min height
    # cut below audible range and above some arbitrary frequency
    unit = freqs[1] if freqs[0] == 0.0 else freqs[0]
    idx = (freqs <= max_f) & (freqs >= min_f)
    freqs = freqs[idx]
    amps = amps[idx]

    # THIS HELPS
    noise_floor = median_filter(amps, size=int((16.0 * distance) / unit))
    filtered_amps = amps - noise_floor

    min_distance = int(distance / unit)

    peaks, _ = find_peaks(filtered_amps, distance=min_distance, height=height)

    # We convert peaks into original indices to see the filtered out parts of the spectrum
    return np.where(idx)[0][peaks]


@app.function
def get_loudest_n_partials(n, amps, peaks):
    return sorted(peaks[np.flip(np.argsort(amps[peaks]))][:n])


@app.function
def get_overtones(
    sample,
    is_loudest=True,
    in_dbs=True,
    height=0.9,
    distance=25.0,
    min_f=25.0,
    max_f=24000.0,
):
    freqs, amps = get_spectrum(sample, SAMPLE_RATE, in_dbs)

    peaks = auto_partial_peaks(
        freqs, amps, height=height, distance=distance, min_f=min_f, max_f=max_f
    )

    # we take the frequency of the first partial as f0
    f0 = freqs[peaks[0]]
    overtones = freqs / f0

    # But if that is too silent, we can exclude it from our calculations
    if is_loudest:
        peaks = get_loudest_n_partials(8, amps, peaks)
    else:
        peaks = peaks[:8]

    return f0, overtones, amps, peaks


@app.function
def draw_overtone_curve(freqs, amps, peaks):
    freqs = freqs[: max(peaks) + 100]
    amps = amps[: max(peaks) + 100]
    fig = Figure(figsize=(12, 5), dpi=300)
    ax = fig.add_axes((0.05, 0.15, 0.9, 0.8))
    ax.scatter(freqs[peaks], amps[peaks], color="red")
    ax.plot(freqs, amps)

    for xii in freqs[peaks]:
        ax.axvline(x=xii, color="b", linestyle="-", alpha=0.3)

    ax.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

    ax.set_xlabel("overtone ratio")
    ax.set_ylabel("amplitude")

    ax.set_xticks(
        freqs[peaks],
        [f"{t:.2f}" for t in freqs[peaks]],
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    return fig


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// warning
    There are problems with distance and hight in auto peak detection. When height limit is 0.5, even though there are many peaks above 0.6, it returns no peaks. drawing scale and calculation scale might be different. Also, take the time to determine a unit for distances between peaks.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _():
    height_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.001,
        value=0.9,
        show_value=True,
        include_input=True,
        label="Percentile",
    )
    min_distance_slider = mo.ui.slider(
        start=0.0,
        stop=6000.0,
        step=1.0,
        value=19.0,
        show_value=True,
        include_input=True,
        label="Distance",
    )
    min_f_slider = mo.ui.slider(
        start=19.0,
        stop=60.0,
        step=0.25,
        value=19.0,
        show_value=True,
        include_input=True,
        label="F_min (in MIDI)",
    )
    is_loudest = mo.ui.checkbox(
        value=True, label="Get loudest partials (not first)"
    )

    mo.hstack(
        [
            mo.vstack(
                [
                    height_slider,
                    min_distance_slider,
                    min_f_slider,
                ]
            ),
            is_loudest,
        ]
    )
    return height_slider, is_loudest, min_distance_slider, min_f_slider


@app.cell
def _(height_slider, is_loudest, min_distance_slider, min_f_slider):
    SAMPLE_PATH = (
        # "data/samples/instruments/brass/Horn/ordinario/Hn-ord-C5-ff-N-N.wav"
        "data/samples/instruments/brass/Horn/ordinario/Hn-ord-C2-ff-N-N.wav"
        # "data/samples/instruments/brass/Horn/flatterzunge/Hn-flatt-F2-ff-N-N.wav"
        # "data/samples/bells/42095__fauxpress__bell-meditation.mp3"
        # "/home/kureta/Documents/repos/musikfabrik/data/samples/bells/62964__ladycailin__deep-bell.wav"
        # "/home/kureta/Documents/repos/musikfabrik/data/samples/bells/124755__tec_studio__mono_bell_0_d_16sec.wav"
    )

    sample, _ = librosa.load(SAMPLE_PATH, sr=SAMPLE_RATE)
    f0, overtones, amps, peaks = get_overtones(
        sample,
        is_loudest=is_loudest.value,
        in_dbs=True,
        height=height_slider.value,
        distance=min_distance_slider.value,
        min_f=librosa.midi_to_hz(min_f_slider.value),
    )

    mo.vstack(
        [
            mo.audio(sample, SAMPLE_RATE),
            mo.hstack(
                [
                    mo.md(
                        f"**`F0 = {f0}`** ({librosa.hz_to_note(f0)}) - Found {len(peaks)} partials. `min_f={librosa.midi_to_note(min_f_slider.value)}`"
                    ),
                ]
            ),
            draw_overtone_curve(overtones, amps, peaks),
        ]
    )
    return amps, f0, overtones, peaks


@app.cell
def _(amps, f0, overtones, peaks):
    cents, roughness = get_dissonance_curve(
        f0 * overtones[peaks],
        amps[peaks],
        f0 * overtones[peaks],
        amps[peaks],
        # f0 * overtones[peaks],
        # amps[peaks],
        # f0 * overtones[peaks],
        # np.ones_like(overtones[peaks]),
        end_delta_cents=1600,
    )
    cons_peaks = get_consonant_intervals(roughness)
    draw_dissonance_curve(cents, roughness, cons_peaks)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## TODO: Bring the following in from `my-tools`""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    - [x] Dissonance curve calculation
    - [ ] Generate audio from a list of partials so we can check our results by ear
    - [ ] Do the same for samples
    - [x] Partials peak finder: Build a UI to select the overtones I want from a sample, using sliders and stuff.
    - [ ] Save those overtones with file names/paths
    - [x] Consonance peak finder: Same as above. Calculate at at least 4 cents per pixel.
    - [ ] Scale generator:
        - [ ] when both instruments are stretched at the same ratio they keep the unstretched consonances and just move them around. In this case we can use the consonances directly, add consonances of the most consonant interval below, and add consonances of the most consonant interval above. This generates an extended family of scales and modes (rotations of scales).
        - [ ] when they are stretched by different amounts, the most consonant intervals begin to bifurcate. In this case, there are too many consonances to apply the same procedure as above. There are 8 unisons, 4 octaves, 2 fifths, and 2 fourths when we take first 8 harmonics for dissonance calculation. We can use these without further modification to create **atmospheres** instead of playing scales.
    - [ ] Try to fit a N^1.x exponential onto the stretched overtones (such as the ones of bells) for stretching an instrument to the same inharmonicity of another instrument or maybe using the value to unstretch the bell if the effect sounds interesting.
    - [ ] Compare overtones of some different techniques, registers, dynamics, mutes on horn and trumpet to see if they significantly deviate from harmonic overtones. If not, just assume they are always harmonic.
    - [ ] THEN MOVE AS MUCH STUFF AS POSSIBLE INTO MODULES (normal python files).
    - [ ] After doing all of the above, maybe look into changing stretch factors of instruments in a way that will keep a specific interval consonant, or other stuff on the reverse side of this, i.e. not from partials to intervals but from intervals to partials.

    ### Focus on bells, DDSP for **power**
    ### Horn and trumpet will mostly play over or under inharmonic sounds
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
