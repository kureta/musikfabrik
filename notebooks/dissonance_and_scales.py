# pyright: basic
"""Dissonance Curves and Scale Generation Notebook

This Marimo notebook provides a UI for:
1. Loading saved partials or defining synthetic partials
2. Generating dissonance curves
3. Detecting consonant intervals
4. Generating scales from intervals
5. Saving scales and intervals to JSON
"""

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import librosa
    import marimo as mo
    import numpy as np
    from matplotlib.figure import Figure

    from musikfabrik.audio_features import (
        synthesize_audio_from_partials,
        load_audio,
    )
    from musikfabrik.dissonance_analysis import (
        calculate_dissonance_curve,
        calculate_dissonance_from_partials_dict,
        calculate_synthetic_dissonance_curve,
        find_consonant_intervals,
        generate_scales_from_intervals,
        generate_synthetic_partials,
        get_all_modes,
    )
    return (
        Figure,
        Path,
        calculate_dissonance_curve,
        find_consonant_intervals,
        generate_scales_from_intervals,
        generate_synthetic_partials,
        get_all_modes,
        json,
        librosa,
        load_audio,
        mo,
        np,
        synthesize_audio_from_partials,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Dissonance Curves and Scale Generation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Define Partials

    Define partials for both the **base (fixed) sound** and the **swept sound** independently.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Base (Fixed) Sound""")
    return


@app.cell
def _(mo):
    fixed_source = mo.ui.radio(
        options=["Synthetic", "Load from file"],
        value="Synthetic",
        label="Base sound source:",
    )
    fixed_source
    return (fixed_source,)


@app.cell
def _(Path, mo):
    # File loader for fixed partials
    features_dir = Path("data/extracted_features")
    if features_dir.exists():
        partial_files = sorted(
            [str(p.relative_to(".")) for p in features_dir.glob("*_partials.json")]
        )
    else:
        partial_files = []

    fixed_file_selector = mo.ui.dropdown(
        options=partial_files if partial_files else ["No files found"],
        value=partial_files[0] if partial_files else None,
        label="Select partials file for base sound:",
        allow_select_none=True,
    )

    fixed_file_selector
    return fixed_file_selector, partial_files


@app.cell
def _(librosa, mo, np):
    # Synthetic partials controls for fixed sound
    fixed_f0 = mo.ui.slider(
        steps=np.logspace(
            np.log2(55.0), np.log2(880.0), num=100, base=2.0
        ).tolist(),
        value=float(librosa.note_to_hz("C4")),
        show_value=True,
        label="F0 (Hz):",
    )

    fixed_n_partials = mo.ui.slider(
        start=2,
        stop=16,
        value=8,
        step=1,
        show_value=True,
        label="Number of partials:",
    )

    fixed_stretch = mo.ui.slider(
        start=0.8,
        stop=1.3,
        value=1.0,
        step=0.01,
        show_value=True,
        label="Stretch factor:",
    )

    fixed_decay = mo.ui.slider(
        start=0.5,
        stop=0.99,
        value=0.9,
        step=0.01,
        show_value=True,
        label="Amplitude decay:",
    )

    mo.vstack(
        [
            fixed_f0,
            fixed_n_partials,
            fixed_stretch,
            fixed_decay,
        ]
    )
    return fixed_decay, fixed_f0, fixed_n_partials, fixed_stretch


@app.cell
def _(
    fixed_decay,
    fixed_f0,
    fixed_file_selector,
    fixed_n_partials,
    fixed_source,
    fixed_stretch,
    generate_synthetic_partials,
    json,
    mo,
    np,
):
    # Generate or load fixed partials
    if fixed_source.value == "Synthetic":
        fixed_partials, fixed_amps = generate_synthetic_partials(
            f0=fixed_f0.value,
            n_partials=fixed_n_partials.value,
            stretch_factor=fixed_stretch.value,
            amp_decay_factor=fixed_decay.value,
        )
        fixed_status = mo.md(
            f"**Generated synthetic partials** (F0={fixed_f0.value:.1f} Hz)"
        )
    elif fixed_source.value == "Load from file" and fixed_file_selector.value:
        try:
            with open(fixed_file_selector.value, "r") as fixed_file_handle:
                fixed_loaded = json.load(fixed_file_handle)
            fixed_partials = np.array(fixed_loaded["frequencies"])
            fixed_amps = np.array(fixed_loaded["amplitudes"])
            fixed_status = mo.md(
                f"**Loaded from:** `{fixed_file_selector.value}`  \n"
                f"**F0:** {fixed_loaded['f0']:.2f} Hz, **Partials:** {len(fixed_partials)}"
            )
        except Exception as e:
            fixed_partials = None
            fixed_amps = None
            fixed_status = mo.md(f"**Error loading file:** {e}")
    else:
        fixed_partials = None
        fixed_amps = None
        fixed_status = mo.md("_Select a file to load base sound partials_")

    fixed_status
    return fixed_amps, fixed_loaded, fixed_partials


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Swept Sound""")
    return


@app.cell
def _(mo):
    swept_source = mo.ui.radio(
        options=["Synthetic", "Load from file"],
        value="Synthetic",
        label="Swept sound source:",
    )
    swept_source
    return (swept_source,)


@app.cell
def _(mo, partial_files):
    swept_file_selector = mo.ui.dropdown(
        options=partial_files if partial_files else ["No files found"],
        value=partial_files[0] if partial_files else None,
        label="Select partials file for swept sound:",
        allow_select_none=True,
    )

    swept_file_selector
    return (swept_file_selector,)


@app.cell
def _(mo):
    # Synthetic partials controls for swept sound
    # Note: swept sound uses the same F0 as the fixed (base) sound

    swept_n_partials = mo.ui.slider(
        start=2,
        stop=16,
        value=8,
        step=1,
        show_value=True,
        label="Number of partials:",
    )

    swept_stretch = mo.ui.slider(
        start=0.8,
        stop=1.3,
        value=1.05,
        step=0.01,
        show_value=True,
        label="Stretch factor:",
    )

    swept_decay = mo.ui.slider(
        start=0.5,
        stop=0.99,
        value=0.9,
        step=0.01,
        show_value=True,
        label="Amplitude decay:",
    )

    mo.vstack(
        [
            mo.md("_Swept sound will use the same F0 as the base sound_"),
            swept_n_partials,
            swept_stretch,
            swept_decay,
        ]
    )
    return swept_decay, swept_n_partials, swept_stretch


@app.cell
def _(
    fixed_partials,
    generate_synthetic_partials,
    json,
    mo,
    np,
    swept_decay,
    swept_file_selector,
    swept_n_partials,
    swept_source,
    swept_stretch,
):
    # Generate or load swept partials
    # Swept partials must have the same F0 as fixed partials for dissonance calculation

    if fixed_partials is not None and len(fixed_partials) > 0:
        base_f0 = fixed_partials[0]  # F0 is the first partial

        if swept_source.value == "Synthetic":
            swept_partials, swept_amps = generate_synthetic_partials(
                f0=base_f0,
                n_partials=swept_n_partials.value,
                stretch_factor=swept_stretch.value,
                amp_decay_factor=swept_decay.value,
            )
            swept_status = mo.md(
                f"**Generated synthetic partials** (F0={base_f0:.1f} Hz, matching base sound)"
            )
        elif swept_source.value == "Load from file" and swept_file_selector.value:
            try:
                with open(swept_file_selector.value, "r") as swept_file_handle:
                    swept_loaded = json.load(swept_file_handle)

                # Load original partials
                swept_partials_orig = np.array(swept_loaded["frequencies"])
                swept_amps = np.array(swept_loaded["amplitudes"])
                swept_f0_orig = swept_loaded["f0"]

                # Transpose to match base_f0
                transpose_ratio = base_f0 / swept_f0_orig
                swept_partials = swept_partials_orig * transpose_ratio

                swept_status = mo.md(
                    f"**Loaded from:** `{swept_file_selector.value}`  \n"
                    f"**Original F0:** {swept_f0_orig:.2f} Hz → **Transposed to:** {base_f0:.2f} Hz  \n"
                    f"**Partials:** {len(swept_partials)}"
                )
            except Exception as e:
                swept_partials = None
                swept_amps = None
                swept_status = mo.md(f"**Error loading file:** {e}")
        else:
            swept_partials = None
            swept_amps = None
            swept_status = mo.md("_Select a file to load swept sound partials_")
    else:
        swept_partials = None
        swept_amps = None
        swept_status = mo.md("_Define base (fixed) sound first_")

    swept_status
    return base_f0, swept_amps, swept_f0_orig, swept_loaded, swept_partials


@app.cell
def _(fixed_amps, fixed_partials, swept_amps, swept_partials):
    # Check if partials are defined
    partials_defined = (
        fixed_partials is not None
        and fixed_amps is not None
        and swept_partials is not None
        and swept_amps is not None
    )
    return (partials_defined,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Calculate Dissonance Curve

    Generate the dissonance curve by sweeping one sound against another.
    """
    )
    return


@app.cell
def _(mo):
    # Dissonance curve parameters
    start_cents = mo.ui.slider(
        start=-800,
        stop=300,
        value=-800,
        step=1,
        show_value=True,
        label="Start (cents):",
    )

    end_cents = mo.ui.slider(
        start=300,
        stop=2400,
        value=800,
        step=1,
        show_value=True,
        label="End (cents):",
    )

    cents_resolution = mo.ui.slider(
        start=0.1,
        stop=2.0,
        value=0.25,
        step=0.05,
        show_value=True,
        label="Resolution (cents/bin):",
    )

    mo.vstack([mo.hstack([start_cents, end_cents]), cents_resolution])
    return cents_resolution, end_cents, start_cents


@app.cell
def _(
    calculate_dissonance_curve,
    cents_resolution,
    end_cents,
    fixed_amps,
    fixed_partials,
    mo,
    partials_defined,
    start_cents,
    swept_amps,
    swept_partials,
):
    # Calculate dissonance curve
    if partials_defined:
        try:
            cents_axis, roughness = calculate_dissonance_curve(
                fixed_partials,
                fixed_amps,
                swept_partials,
                swept_amps,
                start_delta_cents=start_cents.value,
                end_delta_cents=end_cents.value,
                cents_per_bin=cents_resolution.value,
            )

            dissonance_calculated = True
            dissonance_status = mo.md(
                f"**Dissonance curve calculated** with {len(cents_axis)} points "
                f"({start_cents.value} to {end_cents.value} cents)"
            )
        except Exception as e:
            cents_axis = None
            roughness = None
            dissonance_calculated = False
            dissonance_status = mo.md(f"**Error calculating dissonance:** {e}")
    else:
        cents_axis = None
        roughness = None
        dissonance_calculated = False
        dissonance_status = mo.md("**Define partials first**")

    dissonance_status
    return cents_axis, dissonance_calculated, roughness


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Find Consonant Intervals

    Detect local minima in the dissonance curve that represent consonant intervals.
    """
    )
    return


@app.cell
def _(mo):
    # Consonant interval detection parameters
    deviation_slider = mo.ui.slider(
        start=0.0,
        stop=2.0,
        value=0.7,
        step=0.01,
        show_value=True,
        label="Deviation threshold:",
    )

    distance_slider = mo.ui.slider(
        start=1.0,
        stop=200.0,
        value=20.0,
        step=1.0,
        show_value=True,
        label="Minimum distance (bins):",
    )

    mo.hstack([deviation_slider, distance_slider])
    return deviation_slider, distance_slider


@app.cell
def _(
    Figure,
    cents_axis,
    deviation_slider,
    dissonance_calculated,
    distance_slider,
    find_consonant_intervals,
    mo,
    np,
    roughness,
):
    # Find and visualize consonant intervals
    if dissonance_calculated:
        try:
            consonant_peaks = find_consonant_intervals(
                roughness,
                deviation=deviation_slider.value,
                distance=distance_slider.value,
            )

            # Create visualization
            fig_diss = Figure(figsize=(14, 6), dpi=150)
            ax_diss = fig_diss.add_subplot(111)

            ax_diss.plot(cents_axis, roughness, linewidth=1, label="Dissonance")
            ax_diss.scatter(
                cents_axis[consonant_peaks],
                roughness[consonant_peaks],
                color="red",
                s=50,
                zorder=5,
                label="Consonant intervals",
            )

            # Draw vertical lines at consonant intervals
            for peak in consonant_peaks:
                ax_diss.axvline(
                    x=cents_axis[peak], color="blue", linestyle="--", alpha=0.3
                )

            ax_diss.set_xlabel("Interval (cents)")
            ax_diss.set_ylabel("Dissonance")
            ax_diss.set_title("Dissonance Curve with Consonant Intervals")
            ax_diss.grid(True, alpha=0.3)
            ax_diss.legend()

            # Set x-axis ticks at consonant intervals
            if len(consonant_peaks) > 0:
                ax_diss.set_xticks(
                    cents_axis[consonant_peaks],
                    [f"{int(np.round(c))}" for c in cents_axis[consonant_peaks]],
                    rotation=45,
                    fontsize=8,
                )

            fig_diss.tight_layout()

            consonant_cents = cents_axis[consonant_peaks]

            consonant_display = mo.vstack(
                [
                    mo.md(
                        f"**Found {len(consonant_peaks)} consonant intervals:**"
                    ),
                    mo.md(
                        ", ".join(
                            [f"{int(np.round(c))} cents" for c in consonant_cents]
                        )
                    ),
                    fig_diss,
                ]
            )
        except Exception as e:
            consonant_peaks = None
            consonant_cents = None
            consonant_display = mo.md(
                f"**Error finding consonant intervals:** {e}"
            )
    else:
        consonant_peaks = None
        consonant_cents = None
        consonant_display = mo.md("**Calculate dissonance curve first**")

    consonant_display
    return (consonant_cents,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3.5. Listen to Consonant Intervals

    Generate audio to hear the consonant intervals over the base (fixed) sound.
    """
    )
    return


@app.cell
def _(mo):
    # Controls for audio generation
    audio_duration_slider = mo.ui.slider(
        start=0.5,
        stop=5.0,
        value=2.0,
        step=0.5,
        show_value=True,
        label="Audio duration (seconds):",
    )

    interval_gap_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.3,
        step=0.1,
        show_value=True,
        label="Gap between intervals (seconds):",
    )

    mo.vstack(
        [
            mo.md("**Audio Generation Settings:**"),
            audio_duration_slider,
            interval_gap_slider,
        ]
    )
    return audio_duration_slider, interval_gap_slider


@app.cell
def _(mo):
    generate_audio_button = mo.ui.run_button(
        label="Generate Audio for Consonant Intervals",
        kind="success",
    )
    generate_audio_button
    return (generate_audio_button,)


@app.cell
def _(
    audio_duration_slider,
    base_f0,
    consonant_cents,
    fixed_amps,
    fixed_loaded,
    fixed_partials,
    fixed_source,
    generate_audio_button,
    interval_gap_slider,
    librosa,
    load_audio,
    mo,
    np,
    swept_amps,
    swept_f0_orig,
    swept_loaded,
    swept_partials,
    swept_source,
    synthesize_audio_from_partials,
):
    # Generate audio for each consonant interval
    if (
        generate_audio_button.value
        and consonant_cents is not None
        and len(consonant_cents) > 0
        and fixed_partials is not None
        and swept_partials is not None
    ):
        try:
            duration = audio_duration_slider.value
            gap = interval_gap_slider.value
            sr = 48000

            audio_segments = []
            # Generate base sound
            if fixed_source.value == "Synthetic":
                base_audio = synthesize_audio_from_partials(
                    fixed_partials,
                    fixed_amps,
                    duration=duration,
                    sr=sr,
                )
            else:
                base_audio = load_audio(fixed_loaded["source_file"], sr)[0][
                    : int(duration * sr)
                ]

            for interval_cents in consonant_cents:
                # Transpose swept sound to the consonant interval
                if swept_source.value == "Synthetic":
                    # For synthetic, just transpose the frequencies
                    ratio = 2 ** (interval_cents / 1200.0)
                    swept_freq_transposed = swept_partials * ratio
                    swept_audio = synthesize_audio_from_partials(
                        swept_freq_transposed,
                        swept_amps,
                        duration=duration,
                        sr=sr,
                    )
                else:
                    # For loaded file, use librosa pitch shift
                    # First synthesize at base f0, then pitch shift
                    swept_audio_base = load_audio(swept_loaded["source_file"], sr)[
                        0
                    ][: int(duration * sr)]
                    swept_audio_transposed = librosa.effects.pitch_shift(
                        swept_audio_base,
                        sr=sr,
                        n_steps=librosa.hz_to_midi(base_f0)
                        - librosa.hz_to_midi(swept_f0_orig),
                    )
                    # Pitch shift by interval_cents
                    n_steps = interval_cents / 100.0  # Convert cents to semitones
                    swept_audio = librosa.effects.pitch_shift(
                        swept_audio_transposed,
                        sr=sr,
                        n_steps=n_steps,
                    )

                # Mix base and swept sounds
                mixed_audio = base_audio + swept_audio

                # Normalize
                max_val = np.abs(mixed_audio).max()
                if max_val > 0:
                    mixed_audio = mixed_audio / max_val * 0.95

                audio_segments.append(mixed_audio)

                # Add gap
                if gap > 0:
                    gap_samples = int(gap * sr)
                    audio_segments.append(np.zeros(gap_samples))

            # Concatenate all segments
            consonant_audio = np.concatenate(audio_segments)

            # Create display with all audio previews
            audio_displays = [
                mo.md(
                    f"**Consonant Intervals Audio** ({len(consonant_cents)} intervals):"
                ),
                mo.md(
                    ", ".join(
                        [f"{int(np.round(c))} cents" for c in consonant_cents]
                    )
                ),
                mo.audio(src=consonant_audio, rate=sr),
            ]

            consonant_audio_display = mo.vstack(audio_displays)

        except Exception as e:
            consonant_audio = None
            consonant_audio_display = mo.md(f"**Error generating audio:** {e}")
    else:
        consonant_audio = None
        if not generate_audio_button.value:
            consonant_audio_display = mo.md(
                "_Click button above to generate audio_"
            )
        elif consonant_cents is None or len(consonant_cents) == 0:
            consonant_audio_display = mo.md("**Find consonant intervals first**")
        else:
            consonant_audio_display = mo.md(
                "**Define both base and swept sounds first**"
            )

    consonant_audio_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Generate Scales

    Generate scales from consonant intervals using constraint-based algorithm.
    """
    )
    return


@app.cell
def _(consonant_cents, generate_scales_from_intervals, get_all_modes, mo, np):
    # Generate scales from consonant intervals
    if consonant_cents is not None and len(consonant_cents) > 0:
        try:
            # Group intervals by octave position (simplified)
            # This is a basic grouping - the original has more sophisticated logic
            octave_approx = 1200  # cents

            # Create interval groups (unison, 2nds, 3rds, etc.)
            # Simplified: group by proximity to scale degrees
            intervals_grouped = [
                [0],  # unison
                list(
                    consonant_cents[
                        (consonant_cents > 0) & (consonant_cents < 250)
                    ]
                ),
                list(
                    consonant_cents[
                        (consonant_cents >= 250) & (consonant_cents < 400)
                    ]
                ),
                list(
                    consonant_cents[
                        (consonant_cents >= 400) & (consonant_cents < 600)
                    ]
                ),
                list(
                    consonant_cents[
                        (consonant_cents >= 600) & (consonant_cents < 800)
                    ]
                ),
                list(
                    consonant_cents[
                        (consonant_cents >= 800) & (consonant_cents < 1000)
                    ]
                ),
                list(
                    consonant_cents[
                        (consonant_cents >= 1000) & (consonant_cents < 1200)
                    ]
                ),
                [int(np.round(octave_approx))],  # octave
            ]

            # Convert to integers
            intervals_grouped = [
                [int(np.round(c)) for c in group] if group else [0]
                for group in intervals_grouped
            ]

            # Generate scales (if we have enough intervals)
            if all(len(group) > 0 for group in intervals_grouped):
                scales = generate_scales_from_intervals(intervals_grouped)
                modes = get_all_modes(scales)

                scales_display = mo.vstack(
                    [
                        mo.md(f"**Generated {len(scales)} scales**"),
                        mo.md(
                            f"**Generated {len(modes)} modes (including rotations)**"
                        ),
                        mo.md("**Example scales:**"),
                        mo.md("\n\n".join([str(s) for s in scales[:5]])),
                    ]
                )
            else:
                scales = None
                modes = None
                scales_display = mo.md(
                    "**Not enough intervals to generate scales.** "
                    "Try adjusting consonant interval detection parameters."
                )
        except Exception as e:
            scales = None
            modes = None
            intervals_grouped = None
            scales_display = mo.md(f"**Error generating scales:** {e}")
    else:
        scales = None
        modes = None
        intervals_grouped = None
        scales_display = mo.md("**Find consonant intervals first**")

    scales_display
    return intervals_grouped, modes, scales


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Save Results

    Save the consonant intervals and generated scales to JSON.
    """
    )
    return


@app.cell
def _(mo):
    # Save controls
    save_dir_scales = mo.ui.text(
        value="data/scales",
        label="Save directory:",
        full_width=True,
    )

    save_name_scales = mo.ui.text(
        value="scale_data",
        label="Filename (without extension):",
        full_width=True,
    )

    save_scales_btn = mo.ui.button(label="Save Scales and Intervals (JSON)")

    mo.vstack([save_dir_scales, save_name_scales, save_scales_btn])
    return save_dir_scales, save_name_scales, save_scales_btn


@app.cell
def _(
    Path,
    consonant_cents,
    intervals_grouped,
    json,
    mo,
    modes,
    save_dir_scales,
    save_name_scales,
    save_scales_btn,
    scales,
):
    # Save functionality
    save_messages = []

    if save_scales_btn.value:
        try:
            save_path_scales = Path(save_dir_scales.value)
            save_path_scales.mkdir(parents=True, exist_ok=True)

            # Prepare data for saving
            data_to_save = {
                "consonant_intervals_cents": (
                    consonant_cents.tolist() if consonant_cents is not None else []
                ),
                "interval_groups": (
                    intervals_grouped if intervals_grouped is not None else []
                ),
                "scales": scales if scales is not None else [],
                "modes": modes if modes is not None else [],
            }

            # Save to JSON
            json_path_scales = (
                save_path_scales / f"{save_name_scales.value}_scales.json"
            )
            with open(json_path_scales, "w") as f:
                json.dump(data_to_save, f, indent=2)

            save_messages.append(f"✓ Saved scales to {json_path_scales}")
        except Exception as e:
            save_messages.append(f"✗ Error saving scales: {e}")

    if save_messages:
        save_scales_display = mo.vstack([mo.md(msg) for msg in save_messages])
    else:
        save_scales_display = mo.md("_Click button above to save scales_")

    save_scales_display
    return


if __name__ == "__main__":
    app.run()
