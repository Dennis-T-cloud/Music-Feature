# Task 2 — Win Machine Handoff

**Status as of 2026-05-24**

The notebook `task2.ipynb` is structurally complete and runs end-to-end on CPU. The remaining
bottleneck is full model training, which requires a GPU machine. Do everything below inside the
`Task2/` directory unless stated otherwise.

---

## What is already done (Mac side)

| Item | State |
|------|-------|
| Data loading & tokenization | ✓ |
| Seq2Seq LSTM model (bidir encoder + decoder) | ✓ |
| Training loop with teacher-forcing decay | ✓ |
| Constrained autoregressive sampling | ✓ |
| MIDI decode & `symbolic_conditioned.mid` output | ✓ |
| Piano-roll visualization | ✓ |
| `summarize_generation` metrics function | ✓ added |
| `plot_feature_distribution` histogram function | ✓ added |
| Copy output → `../outputs/symbolic_conditioned.mid` | ✓ added (last cell) |
| **Full training** | ✗ **needs GPU** |
| Feature distribution plot (needs full-model output) | ✗ run after training |

---

## What to do on the Win machine

### 0. Setup

```bash
pip install pretty_midi torch numpy matplotlib tqdm pandas
```

Make sure `Task2/nesmdb_midi/train`, `valid`, `test` folders exist (≈4 500 + 400 + 373 `.mid`
files). If only the `.tar.gz` is present the notebook extracts it automatically on first run.

---

### 1. Run full training

Open `Task2/task2.ipynb` in Jupyter and locate the cell that contains:

```python
RUN_FULL_TRAINING = False  # Change to True when ready.
```

Change it to `True`, then run **all cells from the top** in order. The training will use:

```
4 502 train files + 403 valid files
batch_size = 32
epochs = 20          ← recommended starting point
lr = 5e-4
hidden_dim = 256, num_layers = 1
```

Expected behaviour: validation loss should drop from ~4.4 to roughly 2.5–3.0 by epoch 20. If the
curve is still clearly declining at epoch 20 try 30 epochs. If it plateaus before epoch 15 stop
early.

The best checkpoint is saved as `Task2/nes_seq2seq_best.pt`.

> **Training time estimate:** ~20–40 min with a mid-range GPU. CPU only would take 6+ hours.

---

### 2. Regenerate the output MIDI

After training finishes, run the generation cell (Section 8 in the notebook). It will:

1. Load `nes_seq2seq_best.pt` automatically.
2. Save `Task2/symbolic_conditioned.mid` (P1 melody + generated TR bass).
3. The **last cell** (Section copy) then copies it to `../outputs/symbolic_conditioned.mid`.

If you want to try different temperatures, run `save_versions_for_midi` with a few test files
and pick the best-sounding one. Then overwrite `symbolic_conditioned.mid` manually before
running the last copy cell.

Recommended generation settings (already in the notebook):
```python
temperature=0.75, top_k=20, max_p1=192, max_gen_len=300, generated_transpose=0
```

---

### 3. Run the evaluation and visualization cells

After generation, run sections 9, 9.5, and 10 in order:

- **Section 9** — Piano roll (saved as `piano_roll_generated.png`)
- **Section 9.5** — Metrics table + feature distribution histogram (saved as
  `feature_distribution.png`)
- **Section 10** — Multi-temperature candidates (saves CSV + per-candidate PNGs)

All plots are saved in `Task2/`. Copy them back to the repo or leave them for the final
notebook submission.

---

### 4. Files to copy back

After everything is done, make sure these files are committed or shared:

| File | Location |
|------|----------|
| `nes_seq2seq_best.pt` | `Task2/` |
| `symbolic_conditioned.mid` | `Task2/` **and** `../outputs/` |
| `piano_roll_generated.png` | `Task2/` |
| `feature_distribution.png` | `Task2/` |
| `task2.ipynb` (with outputs) | `Task2/` |

---

## Optional: stretch goal (4-track → 5th track)

Not started. See `local_docs/2a_and_more.md` Phase 3 for the design. Only attempt if time
allows after the main pipeline is solid.

High-level plan:
1. Merge P1 + P2 + TR + NO tokens into a single flattened sequence per file (add a `SEP=108`
   token between each track).
2. Create a new `Seq2Seq_5th_Track` class with `hidden_dim=512` and `num_layers=2` to handle
   the denser context.
3. Target: generate a new track (e.g. a drumkit or piano chord pattern) conditioned on the
   merged 4-track context.
4. Save output as `outputs/symbolic_5th_track.mid`.

---

## Quick sanity checks

Before considering training done:

- [ ] Training curve shows clear loss decrease (not flat from epoch 1)
- [ ] `nes_seq2seq_best.pt` exists and is > 5 MB
- [ ] `symbolic_conditioned.mid` plays back with audible melody + bass (open in MuseScore or a DAW)
- [ ] Piano roll shows both P1 (blue) and generated TR (red) with reasonable pitch separation
- [ ] Test perplexity after full training should be < 15 (currently 25.15 with debug model)
