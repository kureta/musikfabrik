# pyright: basic
"""Composition Workspace Notebook

This Marimo notebook provides a UI for:
1. Spectral stretching of audio samples with time-varying stretch factors
2. Controlling DDSP with audio-derived f0/loudness
3. Manually composing DDSP phrases with f0, loudness, and stretch factor
4. Saving generated audio files
"""

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    import pickle
    from datetime import datetime
    from pathlib import Path

    import librosa
    import marimo as mo
    import numpy as np
    import soundfile as sf
    import torch
    from matplotlib.figure import Figure

    from musikfabrik.audio_features import (
        get_dynamic_f0,
        get_dynamic_loudness,
        load_audio,
        stretch_spectrum,
    )
    from musikfabrik.ddsp_helpers import (
        DDSPPhrase,
        adsr_envelope,
        breakpoint_curve,
        constant_curve,
        generate_audio_with_ddsp,
        glissando,
        linear_curve,
        midi_to_hz_curve,
        pitch_sequence,
        vibrato,
    )
    from performer.models.ddsp_module import get_harmonic_stretching_model

    SAMPLE_RATE = 48000
    FRAME_RATE = 250
    return (
        DDSPPhrase,
        Figure,
        FRAME_RATE,
        Path,
        adsr_envelope,
        breakpoint_curve,
        constant_curve,
        datetime,
        generate_audio_with_ddsp,
        get_dynamic_f0,
        get_dynamic_loudness,
        get_harmonic_stretching_model,
        glissando,
        librosa,
        linear_curve,
        load_audio,
        midi_to_hz_curve,
        mo,
        np,
        pickle,
        pitch_sequence,
        sf,
        stretch_spectrum,
        torch,
        vibrato,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Composition Workspace""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 1. Load DDSP Model

    Select a pre-trained DDSP model for synthesis.
    """
    )
    return


@app.cell
def __(Path, mo):
    # Find available checkpoints
    checkpoints_dir = Path("data/checkpoints")
    if checkpoints_dir.exists():
        checkpoint_files = {
            p.stem: str(p) for p in checkpoints_dir.glob("*.ckpt")
        }
    else:
        checkpoint_files = {}

    if not checkpoint_files:
        checkpoint_files = {
            "Drums": "data/checkpoints/drums_baseline.ckpt",
            "Violin": "data/checkpoints/violin_longrun.ckpt",
            "Cello": "data/checkpoints/cello_longrun.ckpt",
            "Flute": "data/checkpoints/flute_longrun.ckpt",
        }

    model_selector = mo.ui.dropdown(
        options=list(checkpoint_files.keys()),
        value=list(checkpoint_files.keys())[0] if checkpoint_files else None,
        label="Select DDSP Model:",
        allow_select_none=False,
    )

    model_selector
    return checkpoint_files, checkpoints_dir, model_selector


@app.cell
def __(checkpoint_files, get_harmonic_stretching_model, mo, model_selector):
    # Load the selected model
    if model_selector.value:
        try:
            with mo.status.spinner(
                title=f"Loading {model_selector.value} model..."
            ):
                ddsp_model = get_harmonic_stretching_model(
                    checkpoint_files[model_selector.value]
                )
            mo.md(f"✓ Loaded **{model_selector.value}** model")
        except Exception as e:
            ddsp_model = None
            mo.md(f"✗ Error loading model: {e}")
    else:
        ddsp_model = None
        mo.md("_Select a model_")
    return (ddsp_model,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 2. Audio-Driven DDSP Control

    Load an audio file to extract f0 and loudness for DDSP control.
    """
    )
    return


@app.cell
def __(Path, mo):
    # File selection for audio-driven control
    samples_dir = Path("data/samples")
    if samples_dir.exists():
        audio_files_ddsp = sorted(
            [str(p.relative_to(".")) for p in samples_dir.rglob("*.wav")]
        )
    else:
        audio_files_ddsp = []

    audio_file_input = mo.ui.text(
        value="data/samples/instruments/brass/Horn/ordinario/Hn-ord-C2-ff-N-N.wav",
        label="Audio file path:",
        full_width=True,
    )

    audio_file_selector = mo.ui.dropdown(
        options=audio_files_ddsp if audio_files_ddsp else ["No audio files found"],
        value=audio_files_ddsp[0] if audio_files_ddsp else None,
        label="Or select from data/samples:",
        allow_select_none=True,
    )

    mo.vstack([audio_file_input, audio_file_selector])
    return audio_file_input, audio_file_selector, audio_files_ddsp, samples_dir


@app.cell
def __(
    audio_file_input,
    audio_file_selector,
    get_dynamic_f0,
    get_dynamic_loudness,
    load_audio,
    mo,
):
    # Load audio and extract features for DDSP
    selected_audio_file = (
        audio_file_selector.value
        if audio_file_selector.value
        else audio_file_input.value
    )

    if selected_audio_file and selected_audio_file != "No audio files found":
        try:
            audio_for_ddsp, _ = load_audio(selected_audio_file)
            f0_from_audio = get_dynamic_f0(audio_for_ddsp)
            loudness_from_audio = get_dynamic_loudness(audio_for_ddsp)

            mo.vstack(
                [
                    mo.md(f"**Loaded:** `{selected_audio_file}`"),
                    mo.audio(src=audio_for_ddsp, rate=48000),
                    mo.md(f"**Extracted {len(f0_from_audio)} frames**"),
                ]
            )
        except Exception as e:
            audio_for_ddsp = None
            f0_from_audio = None
            loudness_from_audio = None
            mo.md(f"**Error loading audio:** {e}")
    else:
        audio_for_ddsp = None
        f0_from_audio = None
        loudness_from_audio = None
        mo.md("**Select or enter an audio file**")
    return (
        audio_for_ddsp,
        f0_from_audio,
        loudness_from_audio,
        selected_audio_file,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Spectral Stretch Control for Audio-Driven Mode""")
    return


@app.cell
def __(mo):
    # Stretch factor control for audio-driven mode
    audio_stretch_start = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.0,
        step=0.01,
        show_value=True,
        label="Start stretch:",
    )

    audio_stretch_peak = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.05,
        step=0.01,
        show_value=True,
        label="Peak stretch:",
    )

    audio_stretch_end = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.0,
        step=0.01,
        show_value=True,
        label="End stretch:",
    )

    mo.hstack([audio_stretch_start, audio_stretch_peak, audio_stretch_end])
    return audio_stretch_end, audio_stretch_peak, audio_stretch_start


@app.cell
def __(
    audio_stretch_end,
    audio_stretch_peak,
    audio_stretch_start,
    constant_curve,
    ddsp_model,
    f0_from_audio,
    generate_audio_with_ddsp,
    linear_curve,
    loudness_from_audio,
    mo,
    np,
):
    # Generate audio from loaded features
    if ddsp_model and f0_from_audio is not None and loudness_from_audio is not None:
        try:
            # Create stretch curve for audio-driven mode
            n_frames_audio = len(f0_from_audio)
            attack_frames_audio = min(100, n_frames_audio // 10)
            release_frames_audio = min(100, n_frames_audio // 10)
            sustain_frames_audio = max(
                0, n_frames_audio - attack_frames_audio - release_frames_audio
            )

            stretch_curve_audio = np.concatenate(
                [
                    np.linspace(
                        audio_stretch_start.value,
                        audio_stretch_peak.value,
                        attack_frames_audio,
                    ),
                    np.full(sustain_frames_audio, audio_stretch_peak.value),
                    np.linspace(
                        audio_stretch_peak.value,
                        audio_stretch_end.value,
                        release_frames_audio,
                    ),
                ]
            )

            with mo.status.spinner(title="Generating audio with DDSP..."):
                audio_driven_output = generate_audio_with_ddsp(
                    ddsp_model,
                    f0_from_audio,
                    loudness_from_audio,
                    stretch_curve_audio,
                )

            mo.vstack(
                [
                    mo.md("**Audio-driven DDSP output:**"),
                    mo.audio(src=audio_driven_output, rate=48000),
                ]
            )
        except Exception as e:
            audio_driven_output = None
            mo.md(f"**Error generating audio:** {e}")
    else:
        audio_driven_output = None
        mo.md("**Load model and audio file first**")
    return (
        attack_frames_audio,
        audio_driven_output,
        n_frames_audio,
        release_frames_audio,
        stretch_curve_audio,
        sustain_frames_audio,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 3. Manual DDSP Phrase Composition

    Create phrases by manually defining f0, loudness, and stretch factor curves.
    """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Phrase Type Selection""")
    return


@app.cell
def __(mo):
    phrase_type = mo.ui.radio(
        options=[
            "Simple Note",
            "Note Sequence",
            "Glissando",
            "Vibrato",
            "Custom Breakpoints",
        ],
        value="Simple Note",
        label="Phrase type:",
    )
    phrase_type
    return (phrase_type,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Simple Note Parameters""")
    return


@app.cell
def __(librosa, mo, np):
    # Simple note controls
    note_pitch = mo.ui.slider(
        steps=np.arange(36, 84, 0.01).tolist(),
        value=60.0,
        show_value=True,
        label="Pitch (MIDI):",
    )

    note_duration = mo.ui.slider(
        start=0.1,
        stop=10.0,
        value=3.0,
        step=0.1,
        show_value=True,
        label="Duration (s):",
    )

    note_loudness = mo.ui.slider(
        start=-80,
        stop=-20,
        value=-40,
        step=1,
        show_value=True,
        label="Loudness (dB):",
    )

    note_stretch = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.0,
        step=0.01,
        show_value=True,
        label="Stretch factor:",
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    note_pitch,
                    mo.md(f"({librosa.midi_to_note(int(note_pitch.value))})"),
                ]
            ),
            note_duration,
            note_loudness,
            note_stretch,
        ]
    )
    return note_duration, note_loudness, note_pitch, note_stretch


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Note Sequence Parameters""")
    return


@app.cell
def __(mo):
    # Note sequence controls
    sequence_pitches = mo.ui.text(
        value="60, 62, 64, 65, 67, 69, 71, 72",
        label="Pitches (MIDI, comma-separated):",
        full_width=True,
    )

    sequence_durations = mo.ui.text(
        value="0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0",
        label="Durations (seconds, comma-separated):",
        full_width=True,
    )

    sequence_loudness = mo.ui.slider(
        start=-80,
        stop=-20,
        value=-40,
        step=1,
        show_value=True,
        label="Loudness (dB):",
    )

    sequence_stretch = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.0,
        step=0.01,
        show_value=True,
        label="Stretch factor:",
    )

    mo.vstack(
        [
            sequence_pitches,
            sequence_durations,
            sequence_loudness,
            sequence_stretch,
        ]
    )
    return (
        sequence_durations,
        sequence_loudness,
        sequence_pitches,
        sequence_stretch,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Glissando Parameters""")
    return


@app.cell
def __(mo, np):
    # Glissando controls
    gliss_start_pitch = mo.ui.slider(
        steps=np.arange(36, 84, 0.01).tolist(),
        value=48.0,
        show_value=True,
        label="Start pitch (MIDI):",
    )

    gliss_end_pitch = mo.ui.slider(
        steps=np.arange(36, 84, 0.01).tolist(),
        value=72.0,
        show_value=True,
        label="End pitch (MIDI):",
    )

    gliss_duration = mo.ui.slider(
        start=0.1,
        stop=10.0,
        value=3.0,
        step=0.1,
        show_value=True,
        label="Duration (s):",
    )

    gliss_curve = mo.ui.radio(
        options=["linear", "exponential", "logarithmic"],
        value="linear",
        label="Curve type:",
    )

    gliss_loudness = mo.ui.slider(
        start=-80,
        stop=-20,
        value=-40,
        step=1,
        show_value=True,
        label="Loudness (dB):",
    )

    gliss_stretch = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.05,
        step=0.01,
        show_value=True,
        label="Stretch factor:",
    )

    mo.vstack(
        [
            mo.hstack([gliss_start_pitch, gliss_end_pitch]),
            gliss_duration,
            gliss_curve,
            gliss_loudness,
            gliss_stretch,
        ]
    )
    return (
        gliss_curve,
        gliss_duration,
        gliss_end_pitch,
        gliss_loudness,
        gliss_start_pitch,
        gliss_stretch,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Vibrato Parameters""")
    return


@app.cell
def __(mo, np):
    # Vibrato controls
    vibr_center_pitch = mo.ui.slider(
        steps=np.arange(36, 84, 0.01).tolist(),
        value=60.0,
        show_value=True,
        label="Center pitch (MIDI):",
    )

    vibr_rate = mo.ui.slider(
        start=1.0,
        stop=10.0,
        value=5.0,
        step=0.1,
        show_value=True,
        label="Rate (Hz):",
    )

    vibr_depth = mo.ui.slider(
        start=0.0,
        stop=2.0,
        value=0.5,
        step=0.01,
        show_value=True,
        label="Depth (semitones):",
    )

    vibr_duration = mo.ui.slider(
        start=0.1,
        stop=10.0,
        value=3.0,
        step=0.1,
        show_value=True,
        label="Duration (s):",
    )

    vibr_loudness = mo.ui.slider(
        start=-80,
        stop=-20,
        value=-40,
        step=1,
        show_value=True,
        label="Loudness (dB):",
    )

    vibr_stretch = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.0,
        step=0.01,
        show_value=True,
        label="Stretch factor:",
    )

    mo.vstack(
        [
            vibr_center_pitch,
            mo.hstack([vibr_rate, vibr_depth]),
            vibr_duration,
            vibr_loudness,
            vibr_stretch,
        ]
    )
    return (
        vibr_center_pitch,
        vibr_depth,
        vibr_duration,
        vibr_loudness,
        vibr_rate,
        vibr_stretch,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Custom Breakpoints""")
    return


@app.cell
def __(mo):
    # Custom breakpoint controls
    custom_pitches = mo.ui.text(
        value="60, 64, 62, 67, 65, 60",
        label="Pitch breakpoints (MIDI, comma-separated):",
        full_width=True,
    )

    custom_pitch_durations = mo.ui.text(
        value="0.5, 0.5, 0.5, 0.5, 0.5",
        label="Pitch segment durations (s, comma-separated):",
        full_width=True,
    )

    custom_loudness_values = mo.ui.text(
        value="-70, -30, -40, -35, -40, -80",
        label="Loudness breakpoints (dB, comma-separated):",
        full_width=True,
    )

    custom_stretch_values = mo.ui.text(
        value="1.0, 1.05, 1.05, 1.0",
        label="Stretch breakpoints (comma-separated):",
        full_width=True,
    )

    custom_stretch_durations = mo.ui.text(
        value="0.5, 1.0, 0.5",
        label="Stretch segment durations (s, comma-separated):",
        full_width=True,
    )

    mo.vstack(
        [
            custom_pitches,
            custom_pitch_durations,
            custom_loudness_values,
            custom_stretch_values,
            custom_stretch_durations,
        ]
    )
    return (
        custom_loudness_values,
        custom_pitch_durations,
        custom_pitches,
        custom_stretch_durations,
        custom_stretch_values,
    )


@app.cell
def __(
    FRAME_RATE,
    breakpoint_curve,
    constant_curve,
    custom_loudness_values,
    custom_pitch_durations,
    custom_pitches,
    custom_stretch_durations,
    custom_stretch_values,
    gliss_curve,
    gliss_duration,
    gliss_end_pitch,
    gliss_loudness,
    gliss_start_pitch,
    gliss_stretch,
    glissando,
    midi_to_hz_curve,
    mo,
    note_duration,
    note_loudness,
    note_pitch,
    note_stretch,
    np,
    phrase_type,
    pitch_sequence,
    sequence_durations,
    sequence_loudness,
    sequence_pitches,
    sequence_stretch,
    vibr_center_pitch,
    vibr_depth,
    vibr_duration,
    vibr_loudness,
    vibr_rate,
    vibr_stretch,
    vibrato,
):
    # Generate phrase based on type
    try:
        if phrase_type.value == "Simple Note":
            # Simple sustained note
            pitch_midi = constant_curve(note_duration.value, note_pitch.value, FRAME_RATE)
            f0_curve = midi_to_hz_curve(pitch_midi)
            loudness_curve = constant_curve(
                note_duration.value, note_loudness.value, FRAME_RATE
            )
            stretch_curve_manual = constant_curve(
                note_duration.value, note_stretch.value, FRAME_RATE
            )

        elif phrase_type.value == "Note Sequence":
            # Parse sequences
            pitches = [float(p.strip()) for p in sequence_pitches.value.split(",")]
            durations = [
                float(d.strip()) for d in sequence_durations.value.split(",")
            ]

            pitch_midi = pitch_sequence(pitches, durations, FRAME_RATE)
            f0_curve = midi_to_hz_curve(pitch_midi)
            total_dur = sum(durations)
            loudness_curve = constant_curve(
                total_dur, sequence_loudness.value, FRAME_RATE
            )
            stretch_curve_manual = constant_curve(
                total_dur, sequence_stretch.value, FRAME_RATE
            )

        elif phrase_type.value == "Glissando":
            pitch_midi = glissando(
                gliss_duration.value,
                gliss_start_pitch.value,
                gliss_end_pitch.value,
                gliss_curve.value,
                FRAME_RATE,
            )
            f0_curve = midi_to_hz_curve(pitch_midi)
            loudness_curve = constant_curve(
                gliss_duration.value, gliss_loudness.value, FRAME_RATE
            )
            stretch_curve_manual = constant_curve(
                gliss_duration.value, gliss_stretch.value, FRAME_RATE
            )

        elif phrase_type.value == "Vibrato":
            pitch_midi = vibrato(
                vibr_duration.value,
                vibr_center_pitch.value,
                vibr_rate.value,
                vibr_depth.value,
                FRAME_RATE,
            )
            f0_curve = midi_to_hz_curve(pitch_midi)
            loudness_curve = constant_curve(
                vibr_duration.value, vibr_loudness.value, FRAME_RATE
            )
            stretch_curve_manual = constant_curve(
                vibr_duration.value, vibr_stretch.value, FRAME_RATE
            )

        elif phrase_type.value == "Custom Breakpoints":
            # Parse custom breakpoints
            pitch_bps = [float(p.strip()) for p in custom_pitches.value.split(",")]
            pitch_durs = [
                float(d.strip()) for d in custom_pitch_durations.value.split(",")
            ]
            loudness_bps = [
                float(l.strip()) for l in custom_loudness_values.value.split(",")
            ]
            stretch_bps = [
                float(s.strip()) for s in custom_stretch_values.value.split(",")
            ]
            stretch_durs = [
                float(d.strip()) for d in custom_stretch_durations.value.split(",")
            ]

            pitch_midi = breakpoint_curve(pitch_bps, pitch_durs, "linear", FRAME_RATE)
            f0_curve = midi_to_hz_curve(pitch_midi)
            loudness_curve = breakpoint_curve(
                loudness_bps, pitch_durs, "linear", FRAME_RATE
            )
            stretch_curve_manual = breakpoint_curve(
                stretch_bps, stretch_durs, "linear", FRAME_RATE
            )

        else:
            raise ValueError(f"Unknown phrase type: {phrase_type.value}")

        # Ensure all curves have the same length
        min_len = min(len(f0_curve), len(loudness_curve), len(stretch_curve_manual))
        f0_curve = f0_curve[:min_len]
        loudness_curve = loudness_curve[:min_len]
        stretch_curve_manual = stretch_curve_manual[:min_len]

        phrase_created = True
        mo.md(f"✓ Created **{phrase_type.value}** phrase ({min_len} frames)")

    except Exception as e:
        f0_curve = None
        loudness_curve = None
        stretch_curve_manual = None
        phrase_created = False
        mo.md(f"✗ Error creating phrase: {e}")
    return (
        d,
        durations,
        f0_curve,
        l,
        loudness_bps,
        loudness_curve,
        min_len,
        p,
        pitch_bps,
        pitch_durs,
        pitch_midi,
        pitches,
        phrase_created,
        stretch_bps,
        stretch_curve_manual,
        stretch_durs,
        total_dur,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Generate Audio from Phrase""")
    return


@app.cell
def __(
    ddsp_model,
    f0_curve,
    generate_audio_with_ddsp,
    loudness_curve,
    mo,
    phrase_created,
    stretch_curve_manual,
):
    # Generate audio from manual phrase
    if ddsp_model and phrase_created:
        try:
            with mo.status.spinner(title="Generating phrase with DDSP..."):
                manual_phrase_output = generate_audio_with_ddsp(
                    ddsp_model,
                    f0_curve,
                    loudness_curve,
                    stretch_curve_manual,
                )

            mo.vstack(
                [
                    mo.md("**Manual phrase DDSP output:**"),
                    mo.audio(src=manual_phrase_output, rate=48000),
                ]
            )
        except Exception as e:
            manual_phrase_output = None
            mo.md(f"**Error generating phrase:** {e}")
    else:
        manual_phrase_output = None
        if not ddsp_model:
            mo.md("**Load a DDSP model first**")
        else:
            mo.md("**Create a phrase first**")
    return (manual_phrase_output,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 4. Save Generated Audio

    Save the generated audio to WAV files.
    """
    )
    return


@app.cell
def __(mo):
    # Save controls
    output_dir = mo.ui.text(
        value="data/generated_audio",
        label="Output directory:",
        full_width=True,
    )

    save_audio_driven_btn = mo.ui.button(label="Save Audio-Driven Output")
    save_manual_phrase_btn = mo.ui.button(label="Save Manual Phrase Output")

    mo.vstack([output_dir, mo.hstack([save_audio_driven_btn, save_manual_phrase_btn])])
    return output_dir, save_audio_driven_btn, save_manual_phrase_btn


@app.cell
def __(
    Path,
    audio_driven_output,
    datetime,
    manual_phrase_output,
    mo,
    output_dir,
    save_audio_driven_btn,
    save_manual_phrase_btn,
    sf,
):
    # Save functionality
    output_messages = []

    if save_audio_driven_btn.value and audio_driven_output is not None:
        try:
            output_path = Path(output_dir.value)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = output_path / f"audio_driven_{timestamp}.wav"

            sf.write(audio_path, audio_driven_output, 48000)
            output_messages.append(f"✓ Saved audio-driven output to {audio_path}")
        except Exception as e:
            output_messages.append(f"✗ Error saving audio-driven output: {e}")

    if save_manual_phrase_btn.value and manual_phrase_output is not None:
        try:
            output_path = Path(output_dir.value)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            phrase_path = output_path / f"manual_phrase_{timestamp}.wav"

            sf.write(phrase_path, manual_phrase_output, 48000)
            output_messages.append(f"✓ Saved manual phrase output to {phrase_path}")
        except Exception as e:
            output_messages.append(f"✗ Error saving manual phrase: {e}")

    if output_messages:
        mo.vstack([mo.md(msg) for msg in output_messages])
    else:
        mo.md("_Click buttons above to save audio_")
    return audio_path, output_messages, output_path, phrase_path, timestamp


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
