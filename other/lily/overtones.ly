\version "2.24.2"
\language "english"

global = {
  \numericTimeSignature
}

\header {
  title = "Trumpet - All pitches (Transposed)"
  subtitle = " "
  tagline = ##f
}

\paper {
  paper-width = 210
  paper-height = 297

  system-system-spacing =
  #'((basic-distance . 12)
     (minimum-distance . 8)
     (padding . 5)
     (stretchability . 100))
}

notes = \absolute {
  \clef bass
  fs^\markup { \tiny "123 (2)" }
  g^\markup { \tiny "13 (2)" }
  gs^\markup { \tiny "23 (2)" }
  a^\markup { \tiny "12(3) (2)" }
  as^\markup { \tiny "1 (2)" }
  b^\markup { \tiny "2 (2)" }
  c'^\markup { \tiny " (2)" }
  \break
  \clef treble
  cs'_\markup { \tiny "2" }^\markup { \tiny "123 (3)" }
  d'_\markup { \tiny "2" }^\markup { \tiny "13 (3)" }
  ds'_\markup { \tiny "2" }^\markup { \tiny "23 (3)" }
  e'_\markup { \tiny "2" }^\markup { \tiny "12(3) (3)" }
  f'_\markup { \tiny "2" }^\markup { \tiny "1 (3)" }
  fs'^\markup { \tiny "123 (4)" }
  fs'_\markup { \tiny "2" }^\markup { \tiny "2 (3)" }
  \break
  g'^\markup { \tiny "13 (4)" }
  g'_\markup { \tiny "2" }^\markup { \tiny " (3)" }
  gs'^\markup { \tiny "23 (4)" }
  a'^\markup { \tiny "12(3) (4)" }
  as'_\markup { \tiny "-14" }^\markup { \tiny "123 (5)" }
  as'^\markup { \tiny "1 (4)" }
  b'_\markup { \tiny "-14" }^\markup { \tiny "13 (5)" }
  \break
  b'^\markup { \tiny "2 (4)" }
  c''_\markup { \tiny "-14" }^\markup { \tiny "23 (5)" }
  c''^\markup { \tiny " (4)" }
  cs''_\markup { \tiny "-14" }^\markup { \tiny "12(3) (5)" }
  cs''_\markup { \tiny "2" }^\markup { \tiny "123 (6)" }
  d''_\markup { \tiny "-14" }^\markup { \tiny "1 (5)" }
  d''_\markup { \tiny "2" }^\markup { \tiny "13 (6)" }
  \break
  ds''_\markup { \tiny "-14" }^\markup { \tiny "2 (5)" }
  ds''_\markup { \tiny "2" }^\markup { \tiny "23 (6)" }
  e''_\markup { \tiny "-31" }^\markup { \tiny "123 (7)" }
  e''_\markup { \tiny "-14" }^\markup { \tiny " (5)" }
  e''_\markup { \tiny "2" }^\markup { \tiny "12(3) (6)" }
  f''_\markup { \tiny "-31" }^\markup { \tiny "13 (7)" }
  f''_\markup { \tiny "2" }^\markup { \tiny "1 (6)" }
  \break
  fs''_\markup { \tiny "-31" }^\markup { \tiny "23 (7)" }
  fs''^\markup { \tiny "123 (8)" }
  fs''_\markup { \tiny "2" }^\markup { \tiny "2 (6)" }
  g''_\markup { \tiny "-31" }^\markup { \tiny "12(3) (7)" }
  g''^\markup { \tiny "13 (8)" }
  g''_\markup { \tiny "2" }^\markup { \tiny " (6)" }
  gs''_\markup { \tiny "-31" }^\markup { \tiny "1 (7)" }
  \break
  gs''^\markup { \tiny "23 (8)" }
  gs''_\markup { \tiny "4" }^\markup { \tiny "123 (9)" }
  a''_\markup { \tiny "-31" }^\markup { \tiny "2 (7)" }
  a''^\markup { \tiny "12(3) (8)" }
  a''_\markup { \tiny "4" }^\markup { \tiny "13 (9)" }
  as''_\markup { \tiny "-31" }^\markup { \tiny " (7)" }
  as''_\markup { \tiny "-14" }^\markup { \tiny "123 (10)" }
  \break
  as''^\markup { \tiny "1 (8)" }
  as''_\markup { \tiny "4" }^\markup { \tiny "23 (9)" }
  b''_\markup { \tiny "-14" }^\markup { \tiny "13 (10)" }
  b''^\markup { \tiny "2 (8)" }
  b''_\markup { \tiny "4" }^\markup { \tiny "12(3) (9)" }
  \clef "treble^8"
  c'''_\markup { \tiny "-49" }^\markup { \tiny "123 (11)" }
  c'''_\markup { \tiny "-14" }^\markup { \tiny "23 (10)" }
  \break
  c'''^\markup { \tiny " (8)" }
  c'''_\markup { \tiny "4" }^\markup { \tiny "1 (9)" }
  cs'''_\markup { \tiny "-49" }^\markup { \tiny "13 (11)" }
  cs'''_\markup { \tiny "-14" }^\markup { \tiny "12(3) (10)" }
  cs'''_\markup { \tiny "2" }^\markup { \tiny "123 (12)" }
  cs'''_\markup { \tiny "4" }^\markup { \tiny "2 (9)" }
  d'''_\markup { \tiny "-49" }^\markup { \tiny "23 (11)" }
  \break
  d'''_\markup { \tiny "-14" }^\markup { \tiny "1 (10)" }
  d'''_\markup { \tiny "2" }^\markup { \tiny "13 (12)" }
  d'''_\markup { \tiny "4" }^\markup { \tiny " (9)" }
  d'''_\markup { \tiny "41" }^\markup { \tiny "123 (13)" }
  ds'''_\markup { \tiny "-49" }^\markup { \tiny "12(3) (11)" }
  ds'''_\markup { \tiny "-14" }^\markup { \tiny "2 (10)" }
  ds'''_\markup { \tiny "2" }^\markup { \tiny "23 (12)" }
  \break
  ds'''_\markup { \tiny "41" }^\markup { \tiny "13 (13)" }
  e'''_\markup { \tiny "-49" }^\markup { \tiny "1 (11)" }
  e'''_\markup { \tiny "-31" }^\markup { \tiny "123 (14)" }
  e'''_\markup { \tiny "-14" }^\markup { \tiny " (10)" }
  e'''_\markup { \tiny "2" }^\markup { \tiny "12(3) (12)" }
  e'''_\markup { \tiny "41" }^\markup { \tiny "23 (13)" }
  f'''_\markup { \tiny "-49" }^\markup { \tiny "2 (11)" }
  \break
  f'''_\markup { \tiny "-31" }^\markup { \tiny "13 (14)" }
  f'''_\markup { \tiny "-12" }^\markup { \tiny "123 (15)" }
  f'''_\markup { \tiny "2" }^\markup { \tiny "1 (12)" }
  f'''_\markup { \tiny "41" }^\markup { \tiny "12(3) (13)" }
  fs'''_\markup { \tiny "-49" }^\markup { \tiny " (11)" }
  fs'''_\markup { \tiny "-31" }^\markup { \tiny "23 (14)" }
  fs'''_\markup { \tiny "-12" }^\markup { \tiny "13 (15)" }
  \break
  fs'''^\markup { \tiny "123 (16)" }
  fs'''_\markup { \tiny "2" }^\markup { \tiny "2 (12)" }
  fs'''_\markup { \tiny "41" }^\markup { \tiny "1 (13)" }
  g'''_\markup { \tiny "-31" }^\markup { \tiny "12(3) (14)" }
  g'''_\markup { \tiny "-12" }^\markup { \tiny "23 (15)" }
  g'''^\markup { \tiny "13 (16)" }
  g'''_\markup { \tiny "2" }^\markup { \tiny " (12)" }
  \break
  g'''_\markup { \tiny "41" }^\markup { \tiny "2 (13)" }
  gs'''_\markup { \tiny "-31" }^\markup { \tiny "1 (14)" }
  gs'''_\markup { \tiny "-12" }^\markup { \tiny "12(3) (15)" }
  gs'''^\markup { \tiny "23 (16)" }
  gs'''_\markup { \tiny "41" }^\markup { \tiny " (13)" }
  a'''_\markup { \tiny "-31" }^\markup { \tiny "2 (14)" }
  a'''_\markup { \tiny "-12" }^\markup { \tiny "1 (15)" }
  \break
  a'''^\markup { \tiny "12(3) (16)" }
  as'''_\markup { \tiny "-31" }^\markup { \tiny " (14)" }
  as'''_\markup { \tiny "-12" }^\markup { \tiny "2 (15)" }
  as'''^\markup { \tiny "1 (16)" }
  b'''_\markup { \tiny "-12" }^\markup { \tiny " (15)" }
  b'''^\markup { \tiny "2 (16)" }
  c''''^\markup { \tiny " (16)" }
  \break

}

music = \new StaffGroup {
  <<
    \new Staff \with {instrumentName = "Overtones" } {
      \override TextScript.self-alignment-X = #CENTER
      \cadenzaOn
      \accidentalStyle dodecaphonic

      \notes
    }
  >>
}

\score {
  \music
  \layout{
    indent =0.0

    \context {
      \Score
      proportionalNotationDuration = #(ly:make-moment 1/8)
    }

    \context {
      \Staff
      \remove "Instrument_name_engraver"
      \remove "Time_signature_engraver"
    }
  }
}