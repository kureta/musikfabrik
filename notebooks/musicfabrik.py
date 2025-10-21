import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")

with app.setup:
    # pyright: basic

    import marimo as mo
    from matplotlib.figure import Figure
    import librosa
    import numpy as np
    import torch
    from torch.nn import functional as F
    import torch.nn as nn
    from scipy.interpolate import interp1d
    from scipy.signal import find_peaks

    from performer.models.ddsp_module import DDSP
    from musikfabrik.seth import (
        dissonance,
        sweep_partials,
        generate_partial_freqs,
        generate_partial_amps,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Musikfabrik (First piece based on the thesis)""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Section for reusing old DDSP checkpoints""")
    return


@app.cell
def _():
    DRM_CKPT = "data/checkpoints/drums_baseline.ckpt"
    VLN_CKPT = "data/checkpoints/violin_longrun.ckpt"
    VLC_CKPT = "data/checkpoints/cello_longrun.ckpt"
    FLT_CKPT = "data/checkpoints/flute_longrun.ckpt"
    FPS = 250
    SAMPLE_RATE = 48000
    return SAMPLE_RATE, VLC_CKPT


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// attention | Cleanup.

    Move `HarmonicOscillator` and `get_model()` into a module.
    ///
    """
    )
    return


@app.class_definition(hide_code=True)
class HarmonicOscillator(nn.Module):
    def __init__(
        self, n_harmonics: int = 64, n_channels: int = 1, sr: int = 48000
    ):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.n_channels = n_channels
        self.sr = sr

        next_phase = torch.zeros(self.n_harmonics)
        self.register_buffer("next_phase", next_phase, persistent=False)

    def forward(
        self,
        f0: torch.Tensor,
        master_amplitude: torch.Tensor,
        overtone_amplitudes: torch.Tensor,
        stretch: torch.Tensor,
    ):
        # f0.shape = [batch, n_channels, time]
        # master_amplitude.shape = [batch, n_channels, time]
        # overtone_amplitudes = [batch, n_channels, n_harmonics, time]

        # Convert f0 from Hz to radians / sample
        # This is faster and does not explode freq values when using 16-bit precision.
        f0 = f0 / self.sr
        f0 = f0 * 2 * np.pi

        harmonics = torch.arange(1, self.n_harmonics + 1, step=1) ** stretch
        # Calculate overtone frequencies
        overtone_fs = torch.einsum("bct,to->bcot", f0, harmonics**stretch)

        # set amplitudes of overtones above Nyquist to 0.0
        overtone_amplitudes[overtone_fs > np.pi] = 0.0
        # normalize harmonic_distribution so it always sums to one
        overtone_amplitudes /= torch.sum(
            overtone_amplitudes, dim=2, keepdim=True
        )
        # scale individual overtone amplitudes by the master amplitude
        overtone_amplitudes = torch.einsum(
            "bcot,bct->bcot", overtone_amplitudes, master_amplitude
        )

        # stretch controls by hop_size
        # refactor stretch into a function or a method
        # overtone_fs = self.pre_stretch(overtone_fs)
        # NOTE: 192 was the hop_size during training
        overtone_fs = F.interpolate(
            overtone_fs,
            size=(overtone_fs.shape[-2], (f0.shape[-1] - 1) * 192),
            mode="bilinear",
            align_corners=True,
        )
        # overtone_fs = self.post_stretch(overtone_fs)
        # overtone_amplitudes = self.pre_stretch(overtone_amplitudes)
        overtone_amplitudes = F.interpolate(
            overtone_amplitudes,
            size=(overtone_amplitudes.shape[-2], (f0.shape[-1] - 1) * 192),
            mode="bilinear",
            align_corners=True,
        )
        # overtone_amplitudes = self.post_stretch(overtone_amplitudes)

        # calculate phases and sinusoids
        # TODO: randomizing phases. Is it necessary?
        # overtone_fs[:, :, :, 0] = np.pi * (
        #     torch.rand(*overtone_fs.shape[:-1], device=overtone_fs.device) * 2 - 1
        # )
        # overtone_fs[:, :, 0] += faz
        phases = torch.cumsum(overtone_fs, dim=-1)
        sinusoids = torch.sin(phases)
        # faz = phases[:, :, -1]
        # print(faz.shape)

        # scale sinusoids by their corresponding amplitudes and sum them to get the final signal
        sinusoids = torch.einsum(
            "bcot,bcot->bcot", sinusoids, overtone_amplitudes
        )
        signal = torch.sum(sinusoids, dim=2)

        return signal


@app.function(hide_code=True)
def get_model(ckpt):
    with torch.inference_mode():
        model = DDSP.load_from_checkpoint(ckpt, map_location="cpu")
        model.eval()
    # swapping out with modified `harmonics` to be able to control overtone stretching
    model.harmonics = HarmonicOscillator(
        model.harmonics.n_harmonics, model.harmonics.n_channels
    )

    return model


@app.cell
def _(VLC_CKPT):
    model = get_model(VLC_CKPT)
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// attention | Use the `composition` module

    There are lots of helpers to control the DDSP. For now, we use simple lines.
    ///
    """
    )
    return


@app.function
def t(end, start=0.0, fps=250):
    duration = end - start
    return np.linspace(
        start, end, int(np.round(duration * fps)), dtype="float32"
    )


@app.function
def line(x1, x2, duration):
    return t(duration) * (x2 - x1) / duration + x1


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// attention | Cleanup.

    `generate_audio` is used in `composition` module. Update function signature everywhere, and add `stretch` ability to all composition tools.
    ///
    """
    )
    return


@app.function
def generate_audio(instrument, f0, db, stretch):
    with torch.inference_mode():
        (f0, master, overtones), noise_ctrl = instrument.controller(
            torch.from_numpy(f0[None, None, :]),  # .cuda()
            torch.from_numpy(db[None, None, :]),  # .cuda()
        )

        # harm = forward(f0, master, overtones, stretch[:, None])
        harm = instrument.harmonics(f0, master, overtones, stretch[:, None])
        noise = instrument.noise(noise_ctrl)
        dry = harm + noise
        wet = instrument.reverb(dry)
        out = dry * 0.1 + wet * 0.9

    return out


@app.function
def get_phrase(base_pitch, stretch_factor):
    dbs = []
    # freqs = []
    pitches = []
    stretches = []

    # make segment
    dur = 5.0
    db = line(-70, -30, dur)
    pitch = base_pitch * np.ones_like(db)
    stretch = line(1.0, 1 + (stretch_factor - 1) / 2, dur)

    # append to list of segments
    dbs.append(db)
    pitches.append(pitch)
    stretches.append(stretch)

    # make another segment
    attack = line(-90, -30, 0.01)
    decay = line(-30, -50, 3.0)
    db = np.concatenate((attack, decay))
    pitch = base_pitch * np.ones_like(db)
    stretch = line(1 + (stretch_factor - 1) / 2, stretch_factor, 3.01)
    dbs.append(db)
    pitches.append(pitch)
    stretches.append(stretch)

    # append to list of segments
    intervals = [
        0.0,
        2.43,
        2.80,
        3.31,
        4.06,
        5.23,
        6.12,
        7.37,
        8.54,
        9.29,
        10.17,
        12.60,
    ]
    for i in intervals:
        dbs.append(db)
        pitches.append(pitch + i)
        stretches.append(stretch_factor * np.ones_like(db))

    db = line(-40, -60, dur)
    pitch = pitch[-1] * np.ones_like(db)
    stretch = line(stretch_factor, 1.0, dur)
    dbs.append(db)
    pitches.append(pitch)
    stretches.append(stretch)

    # concat all segments into one signal
    db = np.concatenate(dbs)
    f0 = librosa.midi_to_hz(np.concatenate(pitches))
    stretch = np.concatenate(stretches)

    return f0, db, stretch


@app.cell
def _(SAMPLE_RATE, model):
    stretched_features = get_phrase(36.0, 1.05)
    stretched_ddsp = generate_audio(model, *stretched_features)

    constant_features = get_phrase(36.0, 1.0)
    constant_ddsp = generate_audio(
        model, np.ones_like(constant_features[0]) * 36.0, *constant_features[1:]
    )

    [
        mo.audio(src=stretched_ddsp.cpu().squeeze().numpy(), rate=SAMPLE_RATE),
        mo.audio(src=constant_ddsp.cpu().squeeze().numpy(), rate=SAMPLE_RATE),
    ]
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
def get_pitch(sample, sr=48000, n_fft=8192, hop_length=512):
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
    sample, f0, stretches, sr=48000, n_fft=8192, hop_length=512
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
def _(SAMPLE_RATE):
    SAMPLE_PATH = (
        "data/samples/instruments/brass/Horn/ordinario/Hn-ord-C2-ff-N-N.wav"
    )
    sample, _ = librosa.load(SAMPLE_PATH, sr=SAMPLE_RATE)
    f0 = get_pitch(sample)
    stretch_curve = get_stretch_curve(len(f0), 1.05)
    stretched_sample = stretch_sample(sample, f0, stretch_curve)

    [
        mo.audio(src=stretched_sample, rate=SAMPLE_RATE),
        mo.audio(src=sample, rate=SAMPLE_RATE),
    ]
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
    stretch_factor_1=1.0,
    stretch_factor_2=1.0,
    amp_decay_factor=0.9,
    start_delta_cents=-100,
    end_delta_cents=1300,
    cents_per_bin=0.25,
):
    fixed_partials = generate_partial_freqs(f0, n_partials, stretch_factor_1)
    fixed_amplitudes = generate_partial_amps(1.0, n_partials, amp_decay_factor)
    partials_to_be_swept = generate_partial_freqs(
        f0, n_partials, stretch_factor_2
    )

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
def _(deviation_slider, distance_slider, end_slider, start_slider):
    def generate_example_curve():
        cents_per_bin = 0.25

        cents_axis, roughness = get_synthetic_dissonance_curve(
            librosa.note_to_hz("C4"),
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
    return (generate_example_curve,)


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
        start=-900,
        stop=1500,
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


@app.cell
def _(generate_example_curve):
    draw_dissonance_curve(*generate_example_curve())
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


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## TODO: Bring the following in from `my-tools`

    - [x] Dissonance curve calculation
    - [ ] Partials peak finder: Build a UI to select the overtones I want from a sample, using sliders and stuff. Save those overtones with file names/paths
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
