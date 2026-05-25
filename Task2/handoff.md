# Task 2 Handoff - Windows RTX 3060 Ti Run

**Status as of 2026-05-24**

Task 2 has now been run end-to-end on the Windows GPU machine. The full
Seq2Seq LSTM training completed, the final MIDI output was generated, and the
main evaluation/visualization artifacts were produced.

Work from inside `Task2/` unless noted otherwise.

---

## Current status

| Item | State |
|------|-------|
| NES-MDB MIDI dataset downloaded/extracted | Done |
| Full training on CUDA | Done |
| Best checkpoint saved | Done |
| Final conditioned MIDI generated | Done |
| Output copied to `../outputs/` | Done |
| Piano-roll plot generated | Done |
| Feature-distribution plot generated | Done |
| Multi-temperature candidates generated | Done |
| Demo MIDI velocity fixed for playback | Done |

---

## Training details

Environment used:

```text
GPU: NVIDIA GeForce RTX 3060 Ti
Python env: .conda/envs/cse153
Torch: CUDA-enabled
```

Training setup:

```text
train files found: 4502
valid files found: 403
test files found: 373

usable train samples: 3794
usable valid samples: 325
usable test samples: 246

batch_size = 32
epochs = 20
lr = 5e-4
hidden_dim = 256
num_layers = 1
```

Best validation result:

```text
best validation loss = 1.4149 at epoch 10
final epoch validation loss = 1.4330
final epoch validation perplexity = 4.19
```

Test result after loading `nes_seq2seq_best.pt`:

```text
test loss = 1.4571834019628482
test perplexity = 4.29384843523682
```

This passes the earlier sanity target of test perplexity `< 15`.

---

## Main artifacts

These are the important outputs:

| File | Purpose |
|------|---------|
| `Task2/nes_seq2seq_best.pt` | Best trained checkpoint, about 6 MB |
| `Task2/symbolic_conditioned.mid` | Final P1 melody + generated TR bass |
| `outputs/symbolic_conditioned.mid` | Copied final MIDI for project-level output |
| `Task2/piano_roll_generated.png` | Piano-roll visualization |
| `Task2/feature_distribution.png` | Generated-vs-ground-truth feature distribution |
| `Task2/nes_training_curve.png` | Training/validation loss curve |
| `Task2/generated_candidates/` | Multi-temperature candidate MIDIs and PNGs |
| `Task2/demo_outputs/` | Demo comparison MIDIs |

The final generated MIDI uses this NES-MDB source file:

```text
Task2/nesmdb_midi/test/002_1943_TheBattleofMidway_00_01Title.mid
```

This corresponds to:

```text
1943: The Battle of Midway - Title
```

Final MIDI content check:

```text
Task2/symbolic_conditioned.mid
duration: 12.763 seconds
track 0: Pulse 1 Melody, 41 notes, pitch range 60-105
track 1: Generated Triangle Bass, 40 notes, pitch range 36-57
```

The final MIDI file is small, currently under 1 KB, but that is expected because
it is a short symbolic MIDI file, not audio.

---

## Playback note

The original NES-MDB MIDI files use very low note velocities. Some players make
the original P1/TR tracks almost inaudible. This affected the demo files:

```text
*_no_bass.mid
*_truth_bass_only.mid
*_truth_with_bass.mid
```

Those files have now been normalized for playback. Low velocities were raised to
90 in:

```text
Task2/demo_outputs/*.mid
Task2/symbolic_conditioned.mid
outputs/symbolic_conditioned.mid
Task2/symbolic_conditioned_with_groundtruth.mid
```

Important: any MIDI we hand off or demo should have note velocities high enough
for normal playback. The notebook now includes `normalize_midi_velocity()` and
calls it when exporting conditioned generations and demo comparison files. If a
teammate manually creates or replaces a MIDI, they should do the same check:

```text
P1 / original copied tracks: velocity >= 90
Generated bass: velocity around 100
```

Recommended listening order:

```text
Task2/demo_outputs/*_no_bass.mid
Task2/demo_outputs/*_generated_with_bass.mid
Task2/demo_outputs/*_truth_with_bass.mid
Task2/demo_outputs/*_truth_bass_only.mid
```

Use MuseScore, VLC, or a DAW. `*_generated_with_bass.mid` is the easiest file for
hearing what the model adds.

---

## Runner added

Because the local environment did not have Jupyter/nbconvert installed, a small
runner was added:

```text
Task2/run_task2.py
```

It executes the code cells from `task2.ipynb` in order and applies the Windows
GPU handoff settings:

```text
RUN_FULL_TRAINING = True
epochs = 20
skip debug training cell
replace the notebook's /tmp eval path with a local file
```

To rerun everything:

```powershell
.\.conda\Scripts\conda.exe run -n cse153 python Task2\run_task2.py
```

The notebook itself was also updated so the full-training cell now shows:

```python
RUN_FULL_TRAINING = True
epochs=20
```

---

## Git / sharing note

The current `.gitignore` ignores many generated Task 2 artifacts, including:

```text
Task2/nesmdb_midi/
Task2/nesmdb_midi.tar.gz
Task2/*.pt
Task2/*.png
Task2/*.csv
Task2/generated_candidates/
Task2/demo_outputs/
outputs/*.mid
```

Therefore `git status` only shows the tracked notebook edit and the new runner:

```text
M  Task2/task2.ipynb
?? Task2/run_task2.py
```

If the next teammate needs to commit generated outputs, they must either use
`git add -f` for specific files or add narrow exceptions to `.gitignore`.

Recommended files to share even if they are ignored:

```text
Task2/nes_seq2seq_best.pt
Task2/symbolic_conditioned.mid
outputs/symbolic_conditioned.mid
Task2/piano_roll_generated.png
Task2/feature_distribution.png
Task2/nes_training_curve.png
Task2/demo_outputs/
```

---

## Remaining optional work

No blocker remains for the baseline Task 2 deliverable.

Optional improvements:

1. Listen through `Task2/generated_candidates/` and manually choose a better
   candidate than the current first candidate.
2. Save the chosen candidate over `Task2/symbolic_conditioned.mid`, then copy it
   to `../outputs/symbolic_conditioned.mid`.
3. If the final notebook submission must include live cell outputs, open
   `task2.ipynb` in Jupyter and rerun from the top. The current local runner did
   not execute through the Jupyter UI.
4. Stretch goal: implement the 4-track-to-5th-track model described in earlier
   planning notes. This has not been started.

---

## Sanity checklist

- [x] Training curve shows clear loss decrease.
- [x] `nes_seq2seq_best.pt` exists and is greater than 5 MB.
- [x] `symbolic_conditioned.mid` contains audible melody and generated bass after velocity fix.
- [x] Piano roll shows P1 and generated TR with pitch separation.
- [x] Test perplexity after full training is below 15.
