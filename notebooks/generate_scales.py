import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")

with app.setup:
    # pyright: basic
    import itertools

    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Generate constrained scales from dissonance curve results""")
    return


@app.cell
def _():
    harmonic_intervals = [
        [0],
        [85, 112, 182, 204, 231],
        [267, 316, 386],
        [471, 498, 583],
        [583, 702, 729],
        [765, 814, 884, 933],
        [969, 996, 1018, 1081, 1088],
        [1200],
    ]

    stretched_intervals = [
        [0],
        [89, 117, 192, 214, 243],
        [280, 331, 406],
        [494, 523, 612],
        [612, 737, 766],
        [803, 854, 929, 980],
        [1017, 1046, 1068, 1135, 1143],
        [1260],
    ]
    return harmonic_intervals, stretched_intervals


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## The constraints

    all intervals have to be
    - larger than the largest previous interval and
    - smaller than the smallest next interval

    ex. all thirds in a scale have to be larger than 231 cents and smaller than 471 cents
    """
    )
    return


@app.function
def mprint(*args, **kwargs):
    with mo.redirect_stdout():
        print(*args, **kwargs)


@app.function
def check_interval(scale, interval, allowed):
    octave = allowed[7][0]
    scale = scale[:-1] + [scale[-1] + item for item in scale]
    for idx in range(len(scale) - interval):
        distance = (scale[idx + interval] - scale[idx]) % octave

        # Since the septimal tritone is both a 4th and a 5th
        # we remove it from 5ths list when checking 4ths and vice versa
        if interval == 3:
            lower_limit = max(allowed[interval - 1])
            upper_limit = min(allowed[interval + 1][1:])
        elif interval == 4:
            lower_limit = max(allowed[interval - 1][:-1])
            upper_limit = min(allowed[interval + 1])
        elif interval == 7:
            if distance != 0:
                return False
            return True
        else:
            lower_limit = max(allowed[interval - 1])
            upper_limit = min(allowed[interval + 1])

        if (distance <= lower_limit) or (distance >= upper_limit):
            return False
    return True


@app.function
def validate(scale, intervals):
    return all(check_interval(scale, i, intervals) for i in range(1, 8))


@app.cell
def _(harmonic_intervals, stretched_intervals):
    def test_validate_interval():
        harm_scale = [0, 182, 386, 498, 702, 884, 1088, 1200]
        stretch_scale = [0, 192, 406, 523, 737, 929, 1143, 1260]

        assert validate(harm_scale, harmonic_intervals)
        assert validate(stretch_scale, stretched_intervals)
    return


@app.function
def get_unique(list_of_scales):
    return [
        list(t_scale)
        for t_scale in set(tuple(scale) for scale in list_of_scales)
    ]


@app.function
def get_all_scales(intervals):
    all_scales = itertools.product(*[i for i in intervals])

    valid_scales = []
    for scale in all_scales:
        scale = list(scale)
        if validate(scale, intervals):
            valid_scales.append(scale)

    return get_unique(valid_scales)


@app.cell
def _(harmonic_intervals, stretched_intervals):
    def test_number_of_scales():
        harmonic_scales = get_all_scales(harmonic_intervals)
        stretched_scales = get_all_scales(stretched_intervals)
        assert len(harmonic_scales) == 576
        assert len(stretched_scales) == 576
    return


@app.function
def rotate(scale):
    tmp = scale[1:]
    tmp.append(tmp[0] + tmp[-1])
    tmp = [t - tmp[0] for t in tmp]

    return tmp


@app.function
def get_rotations(scales):
    rotated = []

    for scale in scales:
        for _ in range(len(scale) - 1):
            scale = rotate(scale)
            rotated.append(scale)

    return get_unique(rotated)


@app.function
def get_all_modes(intervals):
    scales = get_all_scales(intervals)
    scale_rotations = get_rotations(scales)
    modes = get_unique(scales + scale_rotations)

    return modes


@app.cell
def _(harmonic_intervals, stretched_intervals):
    def test_number_of_modes():
        harmonic_modes = get_all_modes(harmonic_intervals)
        stretched_modes = get_all_modes(stretched_intervals)

        assert len(harmonic_modes) == 3353
        assert len(stretched_modes) == 3430
    return


@app.cell
def _(harmonic_intervals, stretched_intervals):
    harmonic_scales = get_all_scales(harmonic_intervals)
    harmonic_modes = get_all_modes(harmonic_intervals)

    stretched_scales = get_all_scales(stretched_intervals)
    stretched_modes = get_all_modes(stretched_intervals)

    mprint(
        f"There are a total of {len(harmonic_scales)} harmonic scales, {len(harmonic_modes)} harmonic modes"
    )
    mprint(
        f"and {len(stretched_scales)} stretched scales, {len(stretched_modes)} stretched modes"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
