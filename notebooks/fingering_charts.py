import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Generate all possible pitches playable on the trumpet or the horn""")
    return


@app.cell
def _():
    # pyright: basic
    import math
    import subprocess
    from typing_extensions import Self

    import librosa
    import marimo as mo
    return Self, librosa, math, mo, subprocess


@app.cell(hide_code=True)
def _():
    head = """
    \\version "2.24.2"
    \\language "english"

    global = {
      \\numericTimeSignature
    }

    \\header {
      title = "Trumpet - All pitches (Concert)"
      subtitle = " "
      tagline = ##f
    }

    \\paper {
      paper-width = 210
      paper-height = 297

      system-system-spacing =
      #'((basic-distance . 12)
         (minimum-distance . 8)
         (padding . 5)
         (stretchability . 100))
    }

    notes = \\absolute {
    """

    tail = """
    }

    music = \\new StaffGroup {
      <<
        \\new Staff \\with {instrumentName = "Overtones" } {
          \\override TextScript.self-alignment-X = #CENTER
          \\cadenzaOn
          \\accidentalStyle forget

          \\notes
        }
      >>
    }

    \\score {
      \\music
      \\layout{
        indent =0.0

        \\context {
          \\Score
          proportionalNotationDuration = #(ly:make-moment 1/8)
        }

        \\context {
          \\Staff
          \\remove "Instrument_name_engraver"
          \\remove "Time_signature_engraver"
        }
      }
    }
    """
    return head, tail


@app.cell
def _(math):
    # Bb trumpet's pedal tone, transposed pitch
    trumpet_pedal = "F♯3"
    # Lowest pedals, all valves pressed
    horn_pedals = ["C2", "F2"]
    horn_transpose = {"C2": "F", "F2": "B♭", "F♯3": "B♭"}
    n_overtones = 16

    octave_map = {
        "9": "''''''",
        "8": "'''''",
        "7": "''''",
        "6": "'''",
        "5": "''",
        "4": "'",
        "3": "",
        "2": ",",
        "1": ",,",
        "0": ",,,",
    }

    valve_map = ["", "2", "1", "12(3)", "23", "13", "123"]


    def nround(n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)
    return (
        horn_transpose,
        n_overtones,
        nround,
        octave_map,
        trumpet_pedal,
        valve_map,
    )


@app.cell
def _(Self, horn_transpose, librosa, nround, octave_map):
    class Microtone:
        def __init__(
            self, f_midi: float, valve: str, overtone: int, side: str = ""
        ):
            self.round_midi_cents = 100 * nround(f_midi) - 200
            self.midi_cents = nround(100 * f_midi) - 200
            self.deviation = self.midi_cents - self.round_midi_cents
            self.name = librosa.midi_to_note(self.round_midi_cents / 100.0)
            self.valve = valve
            self.overtone = overtone
            self.side = side

        @property
        def lily(self):
            lily = list(self.name.replace("♯", "s"))
            lily[0] = lily[0].lower()
            lily[-1] = octave_map[lily[-1]]
            lily = "".join(lily)

            dev_tail = ""
            if "3" in self.valve:
                dev_tail = f" ~ {self.deviation - 100}"
            # If we have 3, we don't need 1.
            elif "1" in self.valve:
                dev_tail = f" ~ {self.deviation - 50}"

            dev = ""
            if self.deviation > 0:
                dev = f"+{self.deviation}"
            elif self.deviation < 0:
                dev = f"{self.deviation}"
            else:
                dev = "0"

            dev = f"{dev}{dev_tail}"

            lily += f'_\\markup {{ \\tiny "{dev}" }}'
            lily += f'^\\markup {{ \\tiny \\bold "{horn_transpose[self.side]}" " {self.valve}" \\circle {{ \\tiny "{self.overtone}" }} }}'

            return lily

        def __repr__(self):
            return f"{self.name} ({self.deviation})"

        __str__ = __repr__

        def __eq__(self, other: Self):
            return self.midi_cents == other.midi_cents

        def __lt__(self, other: Self):
            return self.midi_cents < other.midi_cents
    return (Microtone,)


@app.cell
def _(Microtone, librosa, n_overtones, trumpet_pedal, valve_map):
    microtones = []
    for pedal_tone in [trumpet_pedal]:
        trp_f0 = librosa.note_to_hz(pedal_tone)
        # Skipping pedal tones
        for valve in range(7):
            for n in range(2, n_overtones + 1):
                freq = n * trp_f0
                midi = librosa.hz_to_midi(freq)
                microtones.append(
                    Microtone(midi - valve, valve_map[valve], n, pedal_tone)
                )

    microtones.sort()

    lines = []
    previous_name = librosa.midi_to_note(0)
    count = 0
    prev_idx = 0
    indices = []
    counts = []

    # TODO: add number of ways to get that pitch at the start of bar

    lines.append("\\clef treble")
    for idx, m in enumerate(microtones):
        if m.name != previous_name and idx != 0:
            lines.append('\\bar "|"')
            previous_name = m.name
            counts.append(count)
            count = 0
            indices.append(prev_idx)
            prev_idx = idx
        count += 1

        lines.append(m.lily)

        if (idx + 1) % 7 == 0:
            lines.append("\\break")

        if idx == 36 + 27 - 2:
            lines.append('\\clef "treble^8"')

    # for idx, count in zip(indices[::-1], counts[::-1]):
    #     lines.insert(idx, f'\\mark \\markup {{ \\box {{ \\tiny "{count}" }}  }}')
    return (lines,)


@app.cell
def _(head, lines, mo, subprocess, tail):
    notes = "\n".join(lines)
    score = head + notes + tail

    with open("/tmp/score.ly", "w") as file:
        file.writelines(score)

    try:
        result = subprocess.run(
            ["/usr/bin/lilypond", "-o", "./other/lily/score", "/tmp/score.ly"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        with mo.redirect_stdout():
            print(e.returncode, e.output, e.stderr)
    else:
        with mo.redirect_stdout():
            print(result.stdout, result.stderr)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
