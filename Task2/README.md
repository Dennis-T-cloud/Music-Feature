# Task 2: NES Multi-track Conditioned Generation

## Project summary

This notebook implements a baseline system for **symbolic conditioned music generation** using the NES-MDB MIDI dataset.

The main task is:

> Given the `P1` / Pulse 1 melody track from an NES song, generate a compatible `TR` / Triangle bass track.

In simpler terms, the model listens to the main melody line and tries to write a bass accompaniment for it.

This is intended as our Assignment 2 Task 2 project. It is different from a general "continue this song" task because it uses the multi-track structure of NES music: one track is used as the condition, and another track is generated as the target.

---

## Dataset setup

Please download the MIDI version of NES-MDB from the official NES-MDB GitHub page:

https://github.com/chrisdonahue/nesmdb#download-links

Download the file listed as:

```text
NES-MDB in MIDI Format
```

Note that after downloading the dataset named `nesmdb_midi.tar.gz`, the notebook can extract it automatically.

After processing it with the notebook, we should have this folder structure:

```text
Task_2/
    task2.ipynb
    nesmdb_midi/
        train/
            *.mid
        valid/
            *.mid
        test/
            *.mid
```

---

## What NES-MDB contains

NES-MDB is a dataset of Nintendo Entertainment System music. Each MIDI file is separated into NES-style instrument voices. The important voices for our project are:

```text
P1 = Pulse 1      usually melodic material
P2 = Pulse 2      secondary melody / harmony
TR = Triangle     often bass or low-register support
NO = Noise        percussion / noise channel
```

Our current project uses:

```text
condition input = P1 melody
generation target = TR bass
```

This mapping is musically easy to explain: the model is asked to generate a bass line that supports a given melody.

---

## What the notebook does

The notebook is organized as a complete baseline pipeline.

### 1. Data loading

The notebook reads MIDI files directly from:

```text
nesmdb_midi/train
nesmdb_midi/valid
nesmdb_midi/test
```

The notebook also includes a helper function to inspect the instrument voices in a MIDI file. This is useful for confirming that the file contains the expected `P1`, `P2`, `TR`, and `NO` voices.

### 2. Voice extraction

For each MIDI file, the notebook extracts:

```text
P1 = instrument index 0
TR = instrument index 2
```

The `P1` track becomes the model input. The `TR` track becomes the target sequence during training.

### 3. Tokenization

The notebook converts each track into a simple sequence of tokens.

The format is:

```text
[event, duration, event, duration, ...]
```

An `event` can be either:

```text
a MIDI pitch token
or
REST
```

A `duration` token represents how long the previous event lasts.

The token vocabulary is:

```text
0      = PAD
1      = BOS
2      = EOS
3      = REST
4-91   = pitch tokens, corresponding to MIDI pitches 21-108
92-107 = duration tokens
108    = SEP, currently reserved
```

The explicit `REST` token is important. Without it, the generated bass line would tend to become too dense because the model would only learn note events and not silence.

### 4. Dataset and DataLoader

The notebook builds paired training examples:

```text
input  = BOS + P1 tokens + EOS
target = BOS + TR tokens + EOS
```

During training, the decoder sees the target shifted by one step:

```text
decoder input  = TR sequence without the last token
decoder target = TR sequence without the first token
```

This is standard next-token prediction.

A custom `collate_fn` pads sequences inside each batch so that examples of different lengths can be trained together.

### 5. Model

The model is a small sequence-to-sequence LSTM.

Conceptually:

```text
P1 melody tokens
    -> Encoder LSTM
    -> hidden state summary
    -> Decoder LSTM
    -> generated TR bass tokens
```

The encoder reads the melody. The decoder generates the bass line one token at a time.

The current baseline uses (the architectual design may be improved later):

```text
Embedding layer
Bidirectional LSTM encoder
LSTM decoder
Linear output layer
```

### 6. Training

The model is trained with cross-entropy loss. It learns to predict the next `TR` token given the previous `TR` tokens and the encoded `P1` melody.

The notebook first runs a small debug training setup:

```text
DEBUG_MAX_FILES = 300
epochs = 3
```

This is only for checking that the pipeline works.

For better results, set:

```python
RUN_FULL_TRAINING = True
```

and try more epochs, such as:

```text
10, 20, 30, or more epochs
```

### 7. Constrained sampling

During generation, the model is not allowed to output arbitrary token sequences.

The expected order is:

```text
event, duration, event, duration, ...
```

So the generation code restricts the choices:

```text
event positions    -> pitch token, REST, or EOS
duration positions -> duration token only
```

This prevents many invalid outputs and makes the generated MIDI easier to decode.

### 8. MIDI output

The notebook can generate a new `TR` bass track from a given `P1` melody and save it as a MIDI file.

The main final output is:

```text
symbolic_conditioned.mid
```

This file contains:

```text
original P1 melody
generated TR bass
```

The notebook also includes a helper function that saves several presentation-friendly versions for a chosen MIDI file:

```text
no_bass.mid                 P1 melody only
truth_bass_only.mid         original ground-truth TR bass only
generated_with_bass.mid     P1 melody + generated TR bass
truth_with_bass.mid         P1 melody + original TR bass
```

This makes it easier to demonstrate what the model is doing.

### 9. Visualization

The notebook includes a piano-roll plotting function. It visualizes:

```text
P1 melody
generated TR bass
optional ground-truth TR bass
```

## Important current limitations

This is a baseline, not a polished final model.

Known limitations:

1. The model is small and may generate short or repetitive bass lines after only a few epochs.
2. The debug run is only a smoke test and should not be used as the final musical result.
3. The model does not currently use attention, so the decoder only receives a compressed summary of the melody.
4. The tokenization is simple. It captures pitch, duration, and rests, but not NES timbre or control-change details.
5. The generated bass may require manual candidate selection.
6. The notebook currently does not have quantifiable evaluation/metrics (other than listening to the generated midi).

---

## Suggested next steps?

### Training improvements

Try:

```text
increase epochs to 20 or 30
increase hidden_dim from 256 to 512
try num_layers = 2
adjust dropout
train on the full dataset instead of the debug subset
```

Recommended first experiment:

```text
RUN_FULL_TRAINING = True
epochs = 20
hidden_dim = 256
num_layers = 1
```

Only increase model size after confirming the training curve looks reasonable.

### Generation improvements

Try different sampling settings:

```text
temperature = 0.65, 0.75, 0.85, 0.95
top_k = 10, 20, 40
max_gen_len = 200, 300, 400
min_events_before_eos = 20, 40, 80
```

Lower temperature usually gives safer and more repetitive output. Higher temperature gives more variety but may sound less stable.

### Bass range and instrument sound

Try different output settings:

```text
generated_transpose = 0, -6, -12
generated_program = 33, 38, 39
generated_velocity = 90, 100, 110
```

If the generated bass is too high, try lowering it. If it is hard to hear, increase velocity or choose a synth bass program.

### Add attention

The current encoder compresses the entire melody into a hidden state. A stronger model could add attention so that the decoder can look back at different parts of the melody while generating each bass token.

This would likely improve the relationship between the melody rhythm and the generated bass rhythm.

### Add evaluation metrics?

Currently we only have `Test Loss & Perplexity` as a metric.

Possible metrics:

```text
number of generated notes
mean pitch
pitch range
note density
pitch-class overlap with melody
onset alignment with melody
comparison with ground-truth TR
```

A good evaluation section should combine plots, statistics, and listening examples.

---

## TL;DR of this task

This task uses NES-MDB, a dataset of symbolic NES music with separated instrument voices. We formulate the problem as melody-conditioned bass generation: the model receives the Pulse 1 melody track and generates the Triangle bass track. Both tracks are converted into event-duration tokens. A sequence-to-sequence LSTM encodes the melody and autoregressively decodes a bass line. During generation, token choices are constrained so that outputs follow a valid event-duration pattern. The final result is saved as a MIDI file containing the original melody and the generated bass accompaniment.
