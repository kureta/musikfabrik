(
// ========== OPTIONS & BOOT ==========
Server.default.options.memSize = 256 * 1024; // 1GB for safety; reduce as needed
s.reboot;
)

// ========== SYNTHDEFS ==========
(
SynthDef(\partialBank, { |out=0, freqs=#[0, 0, 0, 0, 0, 0, 0, 0], amps=#[0, 0, 0, 0, 0, 0, 0, 0], adsr=#[0.01,0.2,0.5,0.5], gate=1|
    var env = EnvGen.kr(Env.adsr(*adsr), gate, doneAction:Done.freeSelf) * 0.05;
    var sig = SinOsc.ar(freqs, 0, amps).sum * env;
    Out.ar(out, sig.dup);
}).add;

SynthDef(\irconvo, { |in=0, out=0, irbufL=0, irbufR=0|
    var sig = In.ar(in, 2);
    var left = PartConv.ar(sig[0], 2048, irbufL);
    var right = PartConv.ar(sig[1], 2048, irbufR);
    Out.ar(out, [left, right] * 0.01);
}).add;
)

// ========== CONTROL DATA ==========
(
~numberOfPartials = 8;
~baseMIDINote = 48;
~stretch = 1.05;
~decay = 0.9;
~adsr = [0.02, 0.2, 0.7, 3.0];
~pitches = [2.43, 2.8, 3.31, 4.06, 5.23, 6.12, 7.37, 8.54, 9.29, 10.17, 12.6 ];
~bitches = [0.6, 0.95, 1.39, 1.55, 1.69, 1.8 , 4.11, 4.35, 4.71, 5.26, 6.18, 6.78, 7.51, 7.97, 8.57, 9.94, 10.24, 11.37];
// ~bitches = [0.6, 0.95, 1.20, 1.40, 1.55, 1.68, 1.80, 4.11, 4.35, 4.70, 5.26, 6.18, 6.78, 7.50, 7.96, 8.56, 9.94, 10.24, 11.37, 12.6];
// ~sitches = [0, 2.43, 2.8, 3.31, 4.06, 5.23, -2.43, -2.8, -3.31, -4.06, -5.23];
~sitches = [0, 0, 0, -5.23];

~partialFreqs = { |root, stretch|
    Array.fill(~numberOfPartials, { |i| root * ((i+1) ** stretch) })
};
~harmonicFreqs = { |root|
    Array.fill(~numberOfPartials, { |i| root * (i+1) })
};
~amps = {
    Array.fill(~numberOfPartials, { |i| (~decay ** i) })
};
~fftsize = 2048;
~dryBus = Bus.audio(s, 2);
~irPath = "/home/kureta/Music/performer-IRs/ir-violin.wav"; // <--- EDIT THIS FOR YOUR IR!!!
)

// ========== LOAD, PREPARE IRs, AND LAUNCH CONVOLUTION ==========

(
// This block loads both IR channels, prepares spectrum buffers, and only THEN prints "Both IRs ready."
~prepRoutine = Routine {
    // 1. Load buffers
    b = Buffer.readChannel(s, ~irPath, channels:[0]);
    c = Buffer.readChannel(s, ~irPath, channels:[1]);
    s.sync;
    // 2. Allocate and prepare conv buffers
    ~fftsize = 2048;
    ~bufsize = PartConv.calcBufSize(~fftsize, b);
    ~birspectrum = Buffer.alloc(s, ~bufsize, 1);
    ~cirspectrum = Buffer.alloc(s, ~bufsize, 1);
    s.sync;
    ~birspectrum.preparePartConv(b, ~fftsize);
    ~cirspectrum.preparePartConv(c, ~fftsize);
    s.sync;
    // 3. Now IRs are ready, print message and go on
    "Both IRs ready.".postln;
    // Now launch your convolution synth, etc.
	~irSynth = Synth(\irconvo, [
    \in, ~dryBus.index,
    \out, 0,
    \irbufL, ~birspectrum.bufnum,
    \irbufR, ~cirspectrum.bufnum
]);
}.play;
)

// ========== PERFORMANCE ROUTINE ==========

(
// Run this block ONLY after "Both IRs ready." appears!
~routine = Routine({
    inf.do {
        var fundNote, bazikNote, midiNote, fundFreq, bazikFreq, baseFreq, freqArrays, ampArray, synths;

		fundNote = ~baseMIDINote + ~sitches.choose;
        bazikNote = fundNote + ~pitches.choose;

		/*if ( bazikNote <= ~baseMIDINote,
			{
				bazikNote = bazikNote + 12;
				"Octave corrected".postln;
			},
			{}
		);*/

		if ( ((bazikNote - fundNote) < 3.31).or(bazikNote <= ~baseMIDINote),
		{
			bazikNote = bazikNote + 12;
			"Octave corrected".postln;
		},
		{}
	);

        midiNote = bazikNote + ~bitches.choose;
        fundFreq = fundNote.midicps;
        bazikFreq = bazikNote.midicps;
        baseFreq = midiNote.midicps;

        freqArrays = [
            ~partialFreqs.(fundFreq, ~stretch),
            ~partialFreqs.(bazikFreq, ~stretch),
            ~harmonicFreqs.(baseFreq)
        ];
        ampArray = ~amps.();

		synths = [];
        freqArrays.do { |freqs, i|
            synths = synths.add(Synth(\partialBank, [
                \out, ~dryBus.index,
                \freqs, freqs,
                \amps, ampArray,
                \adsr, ~adsr
			]));
			0.125.wait;
        };
        0.5.wait;
        synths.do(_.set(\gate, 0));
        (0.1 + 0.3.rand).wait;
    }
}).play;
)

// ========== UI ==========

(
w = Window("Overtone Controls", Rect(100, 100, 420, 240));
w.layout = HLayout(
    StaticText().string_("Overtone Stretch Factor:"),
    Slider().action_({ |sl|
		~stretch = sl.value.linlin(0, 1, 1, 1.05).round(0.001);  // (sl.value * 0.05 + 1.0).round(0.01);
        ~label.string = ~stretch.asString;
	}).value_(~stretch.linlin(1, 1.05, 0, 1)),
    ~label = StaticText().string_(~stretch.asString),
    StaticText().string_("Amplitude Decay:"),
    Slider().action_({ |sl|
        ~decay = sl.value.round(0.01);
        ~label3.string = ~decay.asString;
	}).value_(~decay),
    ~label3 = StaticText().string_(~decay.asString),
    StaticText().string_("Base MIDI Note:"),
    Slider().action_({ |sl|
		~baseMIDINote = sl.value.linlin(0, 1, 48, 72).round(1); //(sl.value * 24 + 48).round(1);
        ~label2.string = ~baseMIDINote.asString;
}).value_(~baseMIDINote.linlin(48, 72, 0, 1)),
    ~label2 = StaticText().string_(~baseMIDINote.asString)
);
w.alwaysOnTop = true;
w.front;
)