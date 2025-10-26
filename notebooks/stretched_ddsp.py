# pyright: basic

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")

with app.setup:
    import librosa
    import marimo as mo
    import numpy as np
    import torch

    import composition.common as cc
    from performer.models.ddsp_module import get_harmonic_stretching_model
    from performer.utils import constants as pc
    from performer.utils.features import Loudness

    FPS = 250
    SAMPLE_RATE = 48000


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Reusing Old DDSP Checkpoints""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Notes:
    /// attention | Use the `composition` module

    There are lots of helpers to control the DDSP. For now, we use simple lines.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    #### Example usage of the `composition` module:

    ```
    import composition.common as cc
    import composition.instrument as ci

    def get_phrase(base_pitch, stretch_factor):
        intervals = np.array([
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
            ], dtype='float32')
        pitches = intervals + base_pitch
        frequencies = librosa.midi_to_hz(pitches)
        freq_bpc = cc.bpc_floor(frequencies, [3.0] * len(frequencies))
        amp_bpc = np.concatenate(
            [cc.random_adsr(3.0) for _ in range(len(frequencies))]
        ) * 120. - 140.
        stretch_values = [1.0] + (len(frequencies) - 2) * [stretch_factor] + [1.0]
        stretch_bpc = cc.bpc_floor(stretch_values, [3.0] * len(frequencies))

        phrase = ci.Phrase(freq_bpc, amp_bpc, stretch_bpc)
        phrase2 = ci.Phrase(cc.constant(3.0 * len(frequencies), base_pitch), amp_bpc, cc.constant(3.0 * len(frequencies), 1.0))

        return phrase, phrase2
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # Utilities for generating an example phrase


    def t(end, start=0.0, fps=FPS):
        duration = end - start
        return np.linspace(
            start, end, int(np.round(duration * fps)), dtype="float32"
        )


    def line(x1, x2, duration):
        return t(duration) * (x2 - x1) / duration + x1


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
    return (get_phrase,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## DDSP Model Selection and Inference""")
    return


@app.cell(hide_code=True)
def _():
    checkpoints = {
        "Drums": "data/checkpoints/drums_baseline.ckpt",
        "Violin": "data/checkpoints/violin_longrun.ckpt",
        "Cello": "data/checkpoints/cello_longrun.ckpt",
        "Flute": "data/checkpoints/flute_longrun.ckpt",
    }

    selected_model = mo.ui.dropdown(
        options=checkpoints.keys(),
        value="Cello",
        allow_select_none=False,
        searchable=False,
        label="Instrument: ",
    )

    selected_model
    return checkpoints, selected_model


@app.cell(hide_code=True)
def _(checkpoints, selected_model):
    model = get_harmonic_stretching_model(checkpoints[selected_model.value])
    return (model,)


@app.cell(hide_code=True)
def _(get_phrase, model):
    def play_ddsp_results():
        with mo.status.spinner(title="Running DDSP Inference...") as _spinner:
            _spinner.update(subtitle="Generating stretched sample...")
            stretched_features = get_phrase(36.0, 1.05)
            stretched_ddsp = cc.generate_audio(model, *stretched_features)

            _spinner.update(subtitle="Generating harmonic sample...")
            constant_features = get_phrase(36.0, 1.0)
            constant_ddsp = cc.generate_audio(
                model,
                np.ones_like(constant_features[0]) * 36.0,
                *constant_features[1:],
            )
            _spinner.update(subtitle="Done.")

        # phrase, phrase2 = get_phrase(36.0, 1.05)
        # signal = phrase.audio(model)
        # signal2 = phrase2.audio(model)

        return mo.vstack(
            [
                mo.md("### DDSP Results"),
                mo.audio(src=stretched_ddsp, rate=SAMPLE_RATE),
                mo.audio(src=constant_ddsp, rate=SAMPLE_RATE),
            ]
        )


    play_ddsp_results()
    return


@app.cell
def _():
    sample, _ = librosa.load(
        "data/samples/instruments/brass/Horn/ordinario/Hn-ord-C2-ff-N-N.wav",
        sr=SAMPLE_RATE,
    )
    return (sample,)


@app.cell
def _(sample):
    f0 = librosa.yin(
        sample,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=pc.SAMPLE_RATE,
        frame_length=pc.N_FFT,
        hop_length=pc.HOP_LENGTH,
        # center=False
    ).astype("float32")

    loudness = Loudness()
    ld = (
        loudness.get_amp(torch.from_numpy(sample).unsqueeze(0).unsqueeze(0))
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )

    stretch = np.ones_like(f0) * 1.05
    return f0, ld


@app.cell
def _(f0, ld, model):
    controlled_stretched = cc.generate_audio(
        model, f0, ld, np.ones_like(f0) * 1.05
    )
    return (controlled_stretched,)


@app.cell
def _(f0, ld, model):
    dubling = cc.generate_audio(
        model,
        librosa.midi_to_hz(librosa.hz_to_midi(f0) + 8.57),
        ld,
        np.ones_like(f0),
    )
    return


@app.cell
def _(f0, ld, model):
    tripling = cc.generate_audio(
        model,
        librosa.midi_to_hz(librosa.hz_to_midi(f0) + 11.37),
        ld,
        np.ones_like(f0),
    )
    return


@app.cell
def _(controlled_stretched, sample):
    [
        mo.audio(controlled_stretched, 48000),
        mo.audio(
            librosa.effects.pitch_shift(
                sample, sr=48000, n_steps=857, bins_per_octave=1200
            ),
            48000,
        ),
        mo.audio(
            librosa.effects.pitch_shift(
                sample, sr=48000, n_steps=1137, bins_per_octave=1200
            ),
            48000,
        ),
    ]
    return


@app.cell
def _():
    mo.md("""### mo""")
    return


if __name__ == "__main__":
    app.run()
