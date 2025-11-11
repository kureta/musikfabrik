# pyright: basic
"""Audio Feature Extraction Notebook

This Marimo notebook provides a UI for:
1. Loading audio files
2. Extracting static partials (FFT-based)
3. Extracting dynamic f0 and loudness (STFT-based)
4. Spectral stretching with time-varying stretch factors
5. Saving extracted features to JSON/pickle
"""

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json
    import pickle
    from pathlib import Path

    import librosa
    import marimo as mo
    import numpy as np
    import soundfile as sf
    from matplotlib.figure import Figure

    from musikfabrik.audio_features import (
        create_stretch_curve,
        extract_partials,
        get_dynamic_f0,
        get_dynamic_loudness,
        load_audio,
        stretch_spectrum,
    )

    SAMPLE_RATE = 48000
    return (
        Figure,
        Path,
        SAMPLE_RATE,
        create_stretch_curve,
        extract_partials,
        get_dynamic_f0,
        get_dynamic_loudness,
        json,
        librosa,
        load_audio,
        mo,
        np,
        pickle,
        stretch_spectrum,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Audio Feature Extraction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Audio File

    Select an audio file to extract features from.
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    # File browser for selecting audio files
    data_path = Path("data/samples")
    if data_path.exists():
        audio_files = sorted(
            [str(p.relative_to(".")) for p in data_path.rglob("*.wav")]
            + [str(p.relative_to(".")) for p in data_path.rglob("*.mp3")]
        )
    else:
        audio_files = []

    file_input = mo.ui.text(
        value="data/samples/instruments/brass/Horn/ordinario/Hn-ord-C2-ff-N-N.wav"
        if not audio_files
        else "",
        label="Audio file path:",
        full_width=True,
    )

    file_selector = mo.ui.dropdown(
        options=audio_files if audio_files else ["No audio files found"],
        value=audio_files[0] if audio_files else None,
        label="Or select from data/samples:",
        allow_select_none=True,
    )

    mo.vstack([file_input, file_selector])
    return file_input, file_selector


@app.cell(hide_code=True)
def _(SAMPLE_RATE, file_input, file_selector, load_audio, mo):
    # Load the selected audio file
    selected_file = (
        file_selector.value if file_selector.value else file_input.value
    )

    if selected_file and selected_file != "No audio files found":
        try:
            audio_sample, _ = load_audio(selected_file, sr=SAMPLE_RATE)
            audio_load_display = mo.vstack(
                [
                    mo.md(f"**Loaded:** `{selected_file}`"),
                    mo.audio(src=audio_sample, rate=SAMPLE_RATE),
                ]
            )
        except Exception as e:
            audio_sample = None
            audio_load_display = mo.md(f"**Error loading file:** {e}")
    else:
        audio_sample = None
        audio_load_display = mo.md("**Please select or enter an audio file path**")

    audio_load_display
    return audio_sample, selected_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Extract Static Partials (FFT)

    Extract partial frequencies and amplitudes from the entire audio file.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    # UI controls for partial extraction
    n_partials_slider = mo.ui.slider(
        start=1,
        stop=20,
        value=8,
        step=1,
        show_value=True,
        label="Number of partials:",
    )

    height_slider = mo.ui.slider(
        start=0.001,
        stop=0.5,
        value=0.025,
        step=0.001,
        show_value=True,
        label="Peak height threshold:",
    )

    f0_estimate_slider = mo.ui.slider(
        steps=np.logspace(
            np.log2(40.0), np.log2(1000.0), num=100, base=2.0
        ).tolist(),
        value=np.logspace(np.log2(40.0), np.log2(1000.0), num=100, base=2.0)[0],
        show_value=True,
        label="F0 estimate (Hz):",
    )

    auto_detect_f0 = mo.ui.checkbox(value=True, label="Auto-detect F0")

    mo.vstack(
        [
            n_partials_slider,
            height_slider,
            mo.hstack([f0_estimate_slider, auto_detect_f0]),
        ]
    )
    return auto_detect_f0, f0_estimate_slider, height_slider, n_partials_slider


@app.cell(hide_code=True)
def _(
    Figure,
    SAMPLE_RATE,
    audio_sample,
    auto_detect_f0,
    extract_partials,
    f0_estimate_slider,
    height_slider,
    librosa,
    mo,
    n_partials_slider,
    np,
):
    # Extract partials
    if audio_sample is not None:
        try:
            f0_est = None if auto_detect_f0.value else f0_estimate_slider.value

            partials_data = extract_partials(
                audio_sample,
                sr=SAMPLE_RATE,
                n_partials=n_partials_slider.value,
                height=height_slider.value,
                f0_estimate=f0_est,
                in_dbs=False,
                a_weighted=False,
            )

            # Create visualization
            fig = Figure(figsize=(12, 4), dpi=150)
            ax = fig.add_subplot(111)

            freqs = partials_data["frequencies"]
            amps = partials_data["amplitudes"]
            ratios = partials_data["ratios"]

            ax.stem(ratios, amps, basefmt=" ")
            ax.set_xlabel("Partial Ratio (relative to F0)")
            ax.set_ylabel("Amplitude")
            ax.set_title(
                f"Extracted Partials (F0 = {partials_data['f0']:.2f} Hz, "
                f"{librosa.hz_to_note(partials_data['f0'])})"
            )
            ax.grid(True, alpha=0.3)

            # Add frequency labels
            for i, (r, a, freq_val) in enumerate(zip(ratios, amps, freqs)):
                ax.text(
                    r,
                    a,
                    f"{freq_val:.1f}Hz",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )

            # Create additive synthesis for comparison
            time = np.linspace(0.0, 30.0, num=SAMPLE_RATE * 30)
            synthesis = np.zeros_like(time)
            for freq, amp in zip(freqs, amps):
                synthesis += amp * np.sin(2 * np.pi * freq * time)
            synthesis /= np.abs(synthesis).max()

            cons = [
        -0.90,
        0.00,
        0.90,
        1.21,
        1.80,
        2.20,
        2.70,
        3.06,
        3.51,
        3.82,
        4.08,
        4.43,
        5.30,
        6.02,
        6.28,
        7.59,
        8.25,
        9.19,
        9.79,
        10.10,
        10.60,
        11.31,
        11.79,
        13.41
      ]
            dur = 1.0
            time = np.linspace(
                0.0, dur * len(freqs), num=int(SAMPLE_RATE  * len(cons) * dur)
            )
            synthesis2 = np.zeros_like(time)
            for idx, c in enumerate(cons):
                for freq, amp in zip(freqs, amps):
                    ff = librosa.midi_to_hz(c + librosa.hz_to_midi(freq)) * 2
                    synthesis2[int(idx*SAMPLE_RATE * dur):int((idx+1)*SAMPLE_RATE * dur)] += amp * np.sin(2 * np.pi * ff * time[int(idx*SAMPLE_RATE * dur):int((idx+1)*SAMPLE_RATE * dur)])
            synthesis2 /= np.abs(synthesis2).max()

            partials_display = mo.vstack(
                [
                    mo.md(
                        f"**F0:** {partials_data['f0']:.2f} Hz ({librosa.hz_to_note(partials_data['f0'])})"
                    ),
                    mo.md(f"**Partials found:** {len(freqs)}"),
                    fig,
                    mo.md("**Additive synthesis (to verify partials):**"),
                    mo.audio(src=synthesis, rate=SAMPLE_RATE),
                    mo.audio(src=synthesis2, rate=SAMPLE_RATE),
                ]
            )
        except Exception as e:
            partials_data = None
            partials_display = mo.md(f"**Error extracting partials:** {e}")
    else:
        partials_data = None
        partials_display = mo.md("**Load an audio file first**")

    partials_display
    return freqs, partials_data


@app.cell
def _(freqs, librosa):
    librosa.hz_to_midi(freqs) - 29  # librosa.hz_to_midi(freqs)[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Extract Dynamic Features (STFT)

    Extract time-varying f0 and loudness.
    """)
    return


@app.cell(hide_code=True)
def _(
    Figure,
    SAMPLE_RATE,
    audio_sample,
    get_dynamic_f0,
    get_dynamic_loudness,
    librosa,
    mo,
    np,
):
    from performer.utils.constants import HOP_LENGTH as DDSP_HOP_LENGTH

    # Extract dynamic features
    if audio_sample is not None:
        try:
            dynamic_f0 = get_dynamic_f0(audio_sample)
            dynamic_loudness = get_dynamic_loudness(audio_sample)

            # Visualize
            fig_dynamic = Figure(figsize=(12, 6), dpi=150)
            ax1 = fig_dynamic.add_subplot(211)
            ax2 = fig_dynamic.add_subplot(212, sharex=ax1)

            t = np.arange(len(dynamic_f0)) * DDSP_HOP_LENGTH / SAMPLE_RATE

            # F0 plot
            f0_midi = librosa.hz_to_midi(dynamic_f0)
            ax1.plot(t, f0_midi, linewidth=1)
            ax1.set_ylabel("F0 (MIDI)")
            ax1.set_title("Dynamic F0")
            ax1.grid(True, alpha=0.3)

            # Loudness plot
            ax2.plot(t, dynamic_loudness, linewidth=1)
            ax2.fill_between(
                t, dynamic_loudness.min(), dynamic_loudness, alpha=0.3
            )
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Loudness (dB)")
            ax2.set_title("Dynamic Loudness")
            ax2.grid(True, alpha=0.3)

            fig_dynamic.tight_layout()

            dynamic_features = {
                "f0": dynamic_f0,
                "loudness": dynamic_loudness,
                "sample_rate": SAMPLE_RATE,
                "hop_length": DDSP_HOP_LENGTH,
            }

            dynamic_display = mo.vstack(
                [
                    mo.md(
                        f"**Extracted {len(dynamic_f0)} frames** "
                        f"({len(dynamic_f0) * DDSP_HOP_LENGTH / SAMPLE_RATE:.2f} seconds)"
                    ),
                    fig_dynamic,
                ]
            )
        except Exception as e:
            dynamic_features = None
            dynamic_display = mo.md(f"**Error extracting dynamic features:** {e}")
    else:
        dynamic_features = None
        dynamic_display = mo.md("**Load an audio file first**")

    dynamic_display
    return (dynamic_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Spectral Stretching

    Apply time-varying spectral stretching to the audio.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # UI controls for stretching
    stretch_start = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.0,
        step=0.01,
        show_value=True,
        label="Start stretch:",
    )

    stretch_peak = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.05,
        step=0.01,
        show_value=True,
        label="Peak stretch:",
    )

    stretch_end = mo.ui.slider(
        start=0.8,
        stop=1.5,
        value=1.0,
        step=0.01,
        show_value=True,
        label="End stretch:",
    )

    attack_frames = mo.ui.slider(
        start=10,
        stop=500,
        value=100,
        step=10,
        show_value=True,
        label="Attack frames:",
    )

    release_frames = mo.ui.slider(
        start=10,
        stop=500,
        value=100,
        step=10,
        show_value=True,
        label="Release frames:",
    )

    mo.vstack(
        [
            mo.hstack([stretch_start, stretch_peak, stretch_end]),
            mo.hstack([attack_frames, release_frames]),
        ]
    )
    return (
        attack_frames,
        release_frames,
        stretch_end,
        stretch_peak,
        stretch_start,
    )


@app.cell(hide_code=True)
def _(
    SAMPLE_RATE,
    attack_frames,
    audio_sample,
    create_stretch_curve,
    dynamic_features,
    mo,
    release_frames,
    stretch_end,
    stretch_peak,
    stretch_spectrum,
    stretch_start,
):
    # Apply stretching
    if audio_sample is not None and dynamic_features is not None:
        try:
            # Use pre-computed f0 or recompute
            f0_for_stretch = dynamic_features["f0"]

            # Create stretch curve
            stretch_curve = create_stretch_curve(
                length=len(f0_for_stretch),
                start_stretch=stretch_start.value,
                peak_stretch=stretch_peak.value,
                end_stretch=stretch_end.value,
                attack_frames=attack_frames.value,
                release_frames=release_frames.value,
            )

            # Apply stretching
            stretched_audio = stretch_spectrum(
                audio_sample,
                f0_for_stretch,
                stretch_curve,
                sr=SAMPLE_RATE,
            )

            stretch_display = mo.vstack(
                [
                    mo.md("**Original vs. Stretched:**"),
                    mo.audio(src=audio_sample, rate=SAMPLE_RATE),
                    mo.audio(src=stretched_audio, rate=SAMPLE_RATE),
                ]
            )
        except Exception as e:
            stretched_audio = None
            stretch_display = mo.md(f"**Error stretching audio:** {e}")
    else:
        stretched_audio = None
        stretch_display = mo.md("**Load an audio file and extract dynamic features first**")

    stretch_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Save Extracted Features

    Save the extracted features to JSON and/or pickle files.
    """)
    return


@app.cell(hide_code=True)
def _(Path, file_selector, freqs, mo):
    # UI for saving
    save_dir = mo.ui.text(
        value="data/extracted_features",
        label="Save directory:",
        full_width=True,
    )

    save_name = mo.ui.text(
        value=f"{Path(file_selector.value).resolve().stem}-np{len(freqs)}",
        label="Filename (without extension):",
        full_width=True,
    )

    save_partials_btn = mo.ui.run_button(label="Save Partials (JSON)")
    save_dynamic_btn = mo.ui.run_button(label="Save Dynamic Features (Pickle)")

    mo.vstack(
        [
            save_dir,
            save_name,
            mo.hstack([save_partials_btn, save_dynamic_btn]),
        ]
    )
    return save_dir, save_dynamic_btn, save_name, save_partials_btn


@app.cell(hide_code=True)
def _(
    Path,
    dynamic_features,
    json,
    mo,
    partials_data,
    pickle,
    save_dir,
    save_dynamic_btn,
    save_name,
    save_partials_btn,
    selected_file,
):
    # Save functionality
    messages = []

    if save_partials_btn.value and partials_data is not None:
        try:
            save_path = Path(save_dir.value)
            save_path.mkdir(parents=True, exist_ok=True)

            # Convert numpy arrays to lists for JSON
            partials_json = {
                "source_file": selected_file,
                "f0": float(partials_data["f0"]),
                "frequencies": partials_data["frequencies"].tolist(),
                "amplitudes": partials_data["amplitudes"].tolist(),
                "ratios": partials_data["ratios"].tolist(),
            }

            json_path = save_path / f"{save_name.value}_partials.json"
            with open(json_path, "w") as json_file:
                json.dump(partials_json, json_file, indent=2)

            messages.append(f"✓ Saved partials to {json_path}")
        except Exception as e:
            messages.append(f"✗ Error saving partials: {e}")

    if save_dynamic_btn.value and dynamic_features is not None:
        try:
            save_path = Path(save_dir.value)
            save_path.mkdir(parents=True, exist_ok=True)

            dynamic_dict = {
                "source_file": selected_file,
                **dynamic_features,
            }

            pickle_path = save_path / f"{save_name.value}_dynamic.pkl"
            with open(pickle_path, "wb") as file_handle:
                pickle.dump(dynamic_dict, file_handle)

            messages.append(f"✓ Saved dynamic features to {pickle_path}")
        except Exception as e:
            messages.append(f"✗ Error saving dynamic features: {e}")

    if messages:
        save_display = mo.vstack([mo.md(msg) for msg in messages])
    else:
        save_display = mo.md("_Click buttons above to save features_")

    save_display
    return


if __name__ == "__main__":
    app.run()
