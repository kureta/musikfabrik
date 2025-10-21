(
SynthDef(\partial, { |out=0, freq=440, amp=0.1, atk=0.01, dec=0.2, sus=0.5, rel=0.5, gate=1|
	var env = EnvGen.kr(Env.adsr(atk, dec, sus, rel, 1), gate, doneAction:2);
	var sig = SinOsc.ar(freq) * amp * env;
	Out.ar(out, [sig, sig]);
}).add;
)

(
~numberOfPartials = 8;
~baseMIDINote = 48;
~partialRatios = (1..~numberOfPartials);
~adsr = [0.02, 0.2, 0.7, 3.0];
~stretch = 1.05;
~decay = 0.9;
~pitches = [0.0, 2.42, 2.80, 3.32, 4.06, 5.22, 6.12, 7.37, 8.54, 9.28, 10.17, 12.60];
~bitches = [0.6, 0.95, 1.20, 1.40, 1.55, 1.68, 1.80, 4.11, 4.35, 4.70, 5.26, 6.18, 6.78, 7.50, 7.96, 8.56, 9.94, 10.24, 11.37, 12.6];

~routine = Routine({
	inf.do {
		var bazikNote = ~baseMIDINote + ~pitches.choose;
		var midiNote = bazikNote + ~bitches.choose;
		var bazikFreq = bazikNote.midicps;
		var baseFreq = midiNote.midicps;
		var synths = Array.fill(~numberOfPartials, { |i|
			Synth(\partial, [
				\freq, ~baseMIDINote.midicps * ((i+1)**~stretch), // i+1 harmonics
				\amp, (~decay**i) * 0.1,
				\atk, ~adsr[0], \dec, ~adsr[1], \sus, ~adsr[2], \rel, ~adsr[3]
			])
		});
		var synths2 = Array.fill(~numberOfPartials, { |i|
        Synth(\partial, [
		\freq, bazikFreq * ((i+1)**~stretch), // i+1 harmonics
		\amp, (~decay**i) * 0.1,
		\atk, ~adsr[0], \dec, ~adsr[1], \sus, ~adsr[2], \rel, ~adsr[3]
		])
		});
		var synths3 = Array.fill(~numberOfPartials, { |i|
		Synth(\partial, [
		\freq, baseFreq * (i+1), // i+1 harmonics
		\amp, (~decay**i) * 0.1,
		\atk, ~adsr[0], \dec, ~adsr[1], \sus, ~adsr[2], \rel, ~adsr[3]
		])
		});
        0.125.wait;
        synths.do(_.set(\gate, 0));
		synths2.do(_.set(\gate, 0));
		synths3.do(_.set(\gate, 0));
        (0.1 + 0.3.rand).wait;
    }
}).play;
)

(
w = Window("Overtone Controls", Rect(100, 100, 300, 200));
w.layout = HLayout(
    StaticText().string_("Overtone Stretch Factor:"),
    Slider().action_({ |sl|
		~stretch = sl.value * 0.2 + 0.9;
        ~label.string = ~stretch.round(0.01).asString;
	}),
	// Define the label and set the initial value
    ~label = StaticText().string_(~stretch.round(0.01).asString),

	StaticText().string_("N Partials:"),
    Slider().action_({ |sl|
		~numberOfPartials = (sl.value * 7 + 1.0).round(1);
        ~label2.string = ~numberOfPartials.asString;
    }),
	// Define the label and set the initial value
    ~label2 = StaticText().string_(~numberOfPartials.asString),

	StaticText().string_("Overtone Amplitude Decay:"),
    Slider().action_({ |sl|
		~decay = sl.value.round(0.01);
		~label3.string = ~decay.round(0.01).asString;
    }),
	// Define the label and set the initial value
    ~label3 = StaticText().string_(~decay.round(0.01).asString),
);
w.alwaysOnTop = true;
w.front;
)