import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")

with app.setup:
    # pyright: basic

    import marimo as mo
    import librosa
    import numpy as np
    import torch
    from torch.nn import functional as F
    import torch.nn as nn
    from scipy.interpolate import interp1d

    from performer.models.ddsp_module import DDSP


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
    ### Move the cell below into a module
    `get_model(CKPT_PATH)`
    """
    )
    return


@app.class_definition
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


@app.function
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
    ### These will be unnecessary

    There are lots of tools in `composition` module
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
    ### Convert this into a function

    `generate_audio(model, pitch, loudness_db, stretch)` is used in `composition` module. Update function signature everywhere, and add `stretch` ability to all composition tools.
    """
    )
    return


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
    mo.md(
        r"""
    ### Load audio file and extract features

    #### TODO: also get loudness in db so we can use it to generate DDSP sounds from audio files
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
    mo.md(
        r"""
    ## TODO: Bring the following in from `my-tools`

    - Dissonance curve calculation
    - Partials peak finder: Build a UI to select the overtones I want from a sample, using sliders and stuff. Save those overtones with file names/paths
    - Consonance peak finder: Same as above. Calculate at at least 4 cents per pixel.
    - Scale generator:
        - when both instruments are stretched the same they keep the unstretched consonances, and just move them around. In this case we can use the consonances directly, add consonances of the most consonant interval below, and add consonances of the most consonant interval below. This generates an extended family of scales and modes (rotations of scales).
        - when they are stretched by different amounts, the most consonant intervals begin to bifurcate. In this case, there are too many consonances to apply the same procedure as above. There are 8 unisons, 4 octaves, 2 fifths, and 2 fourths when we take first 8 harmonics for dissonance calculation. We can use these without further modification to create **atmospheres** instead of playing scales.
    - Try to fit a N^1.x exponential onto the stretched overtones (such as the ones of bells) for maybe unstretching later if the effect sounds interesting.
    - Compare overtones of some different techniques, registers, dynamics, mutes on horn and trumpet to see if they significantly deviate from harmonic overtones. If not, just assume they are always harmonic.
    - THEN MOVE AS MUCH STUFF AS POSSIBLE INTO MODULES (normal python files).
    - After doing all of the above, maybe look into changing stretch factors of instruments in a way that will keep a specific interval consonant, or other stuff on the reverse side of this, i.e. not from partials to intervals but from intervals to partials.

    ### Focus on bells, DDSP for **power**
    ### Horn and trumpet will mostly play over or under inharmonic sounds
    """
    )
    return


if __name__ == "__main__":
    app.run()
