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
def __():
    import json
    from pathlib import Path

    import librosa
    import marimo as mo
    import numpy as np
    from matplotlib.figure import Figure

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
        calculate_dissonance_from_partials_dict,
        calculate_synthetic_dissonance_curve,
        find_consonant_intervals,
        generate_scales_from_intervals,
        generate_synthetic_partials,
        get_all_modes,
        json,
        librosa,
        mo,
        np,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Dissonance Curves and Scale Generation""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 1. Define Partials

    Choose between loading saved partials or defining synthetic partials.
    """
    )
    return


@app.cell
def __(mo):
    partial_source = mo.ui.radio(
        options=["Synthetic", "Load from file"],
        value="Synthetic",
        label="Partial source:",
    )
    partial_source
    return (partial_source,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Synthetic Partials Parameters""")
    return


@app.cell
def __(librosa, mo, np):
    # Synthetic partials controls
    f0_slider = mo.ui.slider(
        steps=np.logspace(np.log2(55.0), np.log2(880.0), num=100, base=2.0).tolist(),
        value=float(librosa.note_to_hz("C4")),
        show_value=True,
        label="F0 (Hz):",
    )

    n_partials_synth = mo.ui.slider(
        start=2,
        stop=16,
        value=8,
        step=1,
        show_value=True,
        label="Number of partials:",
    )

    stretch_1 = mo.ui.slider(
        start=0.8,
        stop=1.3,
        value=1.0,
        step=0.01,
        show_value=True,
        label="Stretch factor (fixed sound):",
    )

    stretch_2 = mo.ui.slider(
        start=0.8,
        stop=1.3,
        value=1.05,
        step=0.01,
        show_value=True,
        label="Stretch factor (swept sound):",
    )

    decay_factor = mo.ui.slider(
        start=0.5,
        stop=0.99,
        value=0.9,
        step=0.01,
        show_value=True,
        label="Amplitude decay factor:",
    )

    use_dbs = mo.ui.checkbox(value=True, label="Use dB scale with A-weighting")

    mo.vstack(
        [
            f0_slider,
            n_partials_synth,
            mo.hstack([stretch_1, stretch_2]),
            decay_factor,
            use_dbs,
        ]
    )
    return (
        decay_factor,
        f0_slider,
        n_partials_synth,
        stretch_1,
        stretch_2,
        use_dbs,
    )


@app.cell
def __(
    decay_factor,
    f0_slider,
    generate_synthetic_partials,
    n_partials_synth,
    partial_source,
    stretch_1,
    stretch_2,
    use_dbs,
):
    # Generate synthetic partials
    if partial_source.value == "Synthetic":
        fixed_partials, fixed_amps = generate_synthetic_partials(
            f0=f0_slider.value,
            n_partials=n_partials_synth.value,
            stretch_factor=stretch_1.value,
            amp_decay_factor=decay_factor.value,
            in_dbs=use_dbs.value,
        )

        swept_partials, swept_amps = generate_synthetic_partials(
            f0=f0_slider.value,
            n_partials=n_partials_synth.value,
            stretch_factor=stretch_2.value,
            amp_decay_factor=decay_factor.value,
            in_dbs=use_dbs.value,
        )

        partials_defined = True
    else:
        fixed_partials = None
        fixed_amps = None
        swept_partials = None
        swept_amps = None
        partials_defined = False
    return (
        fixed_amps,
        fixed_partials,
        partials_defined,
        swept_amps,
        swept_partials,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Load Partials from File""")
    return


@app.cell
def __(Path, mo):
    # File loader for partials
    features_dir = Path("data/extracted_features")
    if features_dir.exists():
        partial_files = sorted(
            [str(p.relative_to(".")) for p in features_dir.glob("*_partials.json")]
        )
    else:
        partial_files = []

    partials_file_selector = mo.ui.dropdown(
        options=partial_files if partial_files else ["No files found"],
        value=partial_files[0] if partial_files else None,
        label="Select partials file:",
        allow_select_none=True,
    )

    partials_file_selector
    return features_dir, partial_files, partials_file_selector


@app.cell
def __(
    fixed_amps,
    fixed_partials,
    json,
    mo,
    np,
    partial_source,
    partials_file_selector,
    swept_amps,
    swept_partials,
):
    # Load partials from file
    if partial_source.value == "Load from file" and partials_file_selector.value:
        try:
            with open(partials_file_selector.value, "r") as f:
                loaded_partials = json.load(f)

            # Use loaded partials for both fixed and swept
            fixed_partials = np.array(loaded_partials["frequencies"])
            fixed_amps = np.array(loaded_partials["amplitudes"])
            swept_partials = fixed_partials.copy()
            swept_amps = fixed_amps.copy()
            partials_defined = True

            mo.md(
                f"**Loaded partials from:** `{partials_file_selector.value}`\n\n"
                f"**F0:** {loaded_partials['f0']:.2f} Hz, "
                f"**Partials:** {len(fixed_partials)}"
            )
        except Exception as e:
            mo.md(f"**Error loading partials:** {e}")
    else:
        loaded_partials = None
        mo.md("_Select a file to load partials_")
    return (loaded_partials,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 2. Calculate Dissonance Curve

    Generate the dissonance curve by sweeping one sound against another.
    """
    )
    return


@app.cell
def __(mo):
    # Dissonance curve parameters
    start_cents = mo.ui.slider(
        start=-300,
        stop=300,
        value=-100,
        step=1,
        show_value=True,
        label="Start (cents):",
    )

    end_cents = mo.ui.slider(
        start=300,
        stop=2400,
        value=1300,
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
def __(
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
            mo.md(
                f"**Dissonance curve calculated** with {len(cents_axis)} points "
                f"({start_cents.value} to {end_cents.value} cents)"
            )
        except Exception as e:
            cents_axis = None
            roughness = None
            dissonance_calculated = False
            mo.md(f"**Error calculating dissonance:** {e}")
    else:
        cents_axis = None
        roughness = None
        dissonance_calculated = False
        mo.md("**Define partials first**")
    return cents_axis, dissonance_calculated, roughness


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 3. Find Consonant Intervals

    Detect local minima in the dissonance curve that represent consonant intervals.
    """
    )
    return


@app.cell
def __(mo):
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
def __(
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

            mo.vstack(
                [
                    mo.md(f"**Found {len(consonant_peaks)} consonant intervals:**"),
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
            mo.md(f"**Error finding consonant intervals:** {e}")
    else:
        consonant_peaks = None
        consonant_cents = None
        mo.md("**Calculate dissonance curve first**")
    return ax_diss, consonant_cents, consonant_peaks, fig_diss


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 4. Generate Scales

    Generate scales from consonant intervals using constraint-based algorithm.
    """
    )
    return


@app.cell
def __(consonant_cents, generate_scales_from_intervals, get_all_modes, mo, np):
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
                list(consonant_cents[(consonant_cents > 0) & (consonant_cents < 250)]),
                list(
                    consonant_cents[(consonant_cents >= 250) & (consonant_cents < 400)]
                ),
                list(
                    consonant_cents[(consonant_cents >= 400) & (consonant_cents < 600)]
                ),
                list(
                    consonant_cents[(consonant_cents >= 600) & (consonant_cents < 800)]
                ),
                list(
                    consonant_cents[(consonant_cents >= 800) & (consonant_cents < 1000)]
                ),
                list(
                    consonant_cents[(consonant_cents >= 1000) & (consonant_cents < 1200)]
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

                mo.vstack(
                    [
                        mo.md(f"**Generated {len(scales)} scales**"),
                        mo.md(f"**Generated {len(modes)} modes (including rotations)**"),
                        mo.md("**Example scales:**"),
                        mo.md("\n\n".join([str(s) for s in scales[:5]])),
                    ]
                )
            else:
                scales = None
                modes = None
                mo.md(
                    "**Not enough intervals to generate scales.** "
                    "Try adjusting consonant interval detection parameters."
                )
        except Exception as e:
            scales = None
            modes = None
            intervals_grouped = None
            mo.md(f"**Error generating scales:** {e}")
    else:
        scales = None
        modes = None
        intervals_grouped = None
        mo.md("**Find consonant intervals first**")
    return group, intervals_grouped, modes, octave_approx, s, scales


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
    ## 5. Save Results

    Save the consonant intervals and generated scales to JSON.
    """
    )
    return


@app.cell
def __(mo):
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
def __(
    Path,
    consonant_cents,
    intervals_grouped,
    json,
    modes,
    mo,
    np,
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
        mo.vstack([mo.md(msg) for msg in save_messages])
    else:
        mo.md("_Click button above to save scales_")
    return data_to_save, json_path_scales, save_messages, save_path_scales


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
