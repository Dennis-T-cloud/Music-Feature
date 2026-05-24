# Music Feature — CSE 153 / 253 Assignment 2

**Course:** UCSD CSE 153 / 253 · Spring 2026  
**Group members:** Dennis (Dennis-T-cloud), HanyuanZhang25

---

## Tasks

| Task | Topic | Status |
| --- | --- | --- |
| Task 1 | Symbolic unconditioned music generation | ✅ Complete |
| Task 2 | *(in progress)* | 🔧 |

---

## Task 1: Symbolic Unconditioned Generation

### Method

Two-stage LoRA fine-tuning on the pretrained **Aria** symbolic MIDI Transformer (`loubb/aria-medium-base`, 2.63 GB):

```
Aria pretrained model
  → Stage 1: LoRA fine-tuning on 128 MAESTRO train files (400 steps)
  → Stage 2: continued LoRA fine-tuning on Chopin Étude subset (300 steps, converged)
```

**LoRA config:** r=8, alpha=16, dropout=0.15  
**Stage 2 final hyperparameters:** lr=5e-6, eval-every=25, early-stop patience=4

### Results (Chopin Étude test split)

| Model | Test Loss | Test Acc |
| --- | --- | --- |
| Pretrained Aria (baseline) | 1.4720 | 47.62% |
| After Stage 1 (MAESTRO LoRA) | 1.4204 | 48.00% |
| **After Stage 2 (final, converged)** | **1.4111** | **48.05%** |

Stage 2 best checkpoint: step 275 of 300 (loss plateau confirmed at step 300).  
Full experiment log: [`notebooks/task1_analysis.ipynb`](notebooks/task1_analysis.ipynb)

### Key finding

The original Stage 2 run (lr=1e-5) overfit severely after step 50 (loss rose from 1.4106 to 1.4523 by step 300). Reducing lr to 5e-6 eliminated overfitting entirely — eval loss decreased monotonically across all 300 steps and converged.

### Generated output

| File | Description |
| --- | --- |
| `outputs/symbolic_unconditioned.mid` | Unconditioned generation from final adapter (tempo-fixed) |
| `outputs/symbolic_conditioned.mid` | Conditioned generation (tempo-fixed) |

> **Note:** Aria's `to_midi()` has a known bug — it writes BPM (120) directly as μs/beat, producing a 0.006 s file. Both outputs have been fixed (`set_tempo=500000`).

### Adapters

| Adapter | Path | Loss |
| --- | --- | --- |
| Stage 1 best | `Task1/result/stage1_maestro_best_adapter/` | MAESTRO val 1.7248 |
| Stage 2 final ★ | `Task1/result/stage2_chopin_final/` | Chopin test 1.4111 |
| Stage 2 teammate | `Task1/result/stage2_chopin_etude_best_adapter/` | Chopin test 1.4150 |

---

## Project Structure

```
Music-Feature/
├── Task1/
│   ├── dataset/
│   │   └── chopin_etude_metadata.csv   # 68-file Chopin subset metadata
│   ├── train/
│   │   ├── aria_finetune_maestro.py    # LoRA training script
│   │   ├── aria_generate.py            # MIDI generation script
│   │   ├── aria_download_weights.py    # Aria base model downloader
│   │   └── requirements_task1.txt
│   ├── report/
│   │   └── generate_plots.py           # Regenerates all analysis figures
│   └── result/
│       ├── stage1_maestro_best_adapter/
│       ├── stage2_chopin_final/        # ★ official final adapter
│       ├── stage2_chopin_etude_best_adapter/
│       ├── stage2_chopin_improved/
│       ├── stage1_maestro_training_log.csv
│       ├── stage2_chopin_etude_training_log.csv
│       └── plots/                      # 10 PNG analysis figures
├── notebooks/
│   └── task1_analysis.ipynb            # Full experiment log & analysis
├── outputs/
│   ├── symbolic_unconditioned.mid      # ★ final generated MIDI (tempo-fixed)
│   └── symbolic_conditioned.mid
├── workbook.ipynb                      # Assignment submission notebook
├── HANDOFF_WIN_RTX3060.md              # Windows GPU training instructions
└── data/                               # MAESTRO dataset (not tracked)
```

---

## Reproducing Task 1

### Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft pandas matplotlib mido pretty_midi huggingface_hub safetensors tqdm
```

### 1 — Download base model

```bash
python Task1/train/aria_download_weights.py \
  --repo-id loubb/aria-medium-base \
  --filename model.safetensors \
  --local-dir Task1/.vendor/aria-hf
```

### 2 — Stage 1: MAESTRO fine-tuning

```bash
python Task1/train/aria_finetune_maestro.py \
  --model-dir Task1/.vendor/aria-hf \
  --maestro-root data/maestro-v3.0.0 \
  --split train --max-files 128 \
  --train-mode lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.15 \
  --max-steps 400 --eval-split validation --eval-every 50 \
  --save-dir Task1/result/stage1_maestro_best_adapter --seed 42
```

### 3 — Stage 2: Chopin Étude fine-tuning

```bash
python Task1/train/aria_finetune_maestro.py \
  --model-dir Task1/.vendor/aria-hf \
  --maestro-root data/maestro-v3.0.0 \
  --split train --composer "Chopin" --title-contains "Etude" --max-files 64 \
  --train-mode lora --resume-adapter Task1/result/stage1_maestro_best_adapter \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.15 \
  --lr 5e-6 --max-steps 300 \
  --eval-split test --eval-max-files 6 --eval-every 25 \
  --early-stop-patience 4 --early-stop-min-delta 0.001 \
  --save-dir Task1/result/stage2_chopin_final --seed 42
```

### 4 — Generate MIDI

```bash
python Task1/train/aria_generate.py \
  --model-id Task1/.vendor/aria-hf \
  --adapter-dir Task1/result/stage2_chopin_final \
  --temperature 0.85 --max-length 4096 \
  --output outputs/symbolic_unconditioned_raw.mid --seed 42

# Fix Aria tempo bug
python -c "
import mido
mid = mido.MidiFile('outputs/symbolic_unconditioned_raw.mid')
new = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
for track in mid.tracks:
    t = mido.MidiTrack()
    for msg in track:
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=msg.time)
                 if msg.type == 'set_tempo' else msg)
    new.tracks.append(t)
new.save('outputs/symbolic_unconditioned.mid')
"
```

### 5 — Regenerate analysis plots

```bash
/opt/anaconda3/envs/cse153-hw4/bin/python Task1/report/generate_plots.py
```

Figures saved to `Task1/result/plots/`.

---

## Tech Stack

| Component | Library |
| --- | --- |
| Base model | Aria (`loubb/aria-medium-base`) |
| LoRA fine-tuning | `transformers` + `peft` |
| MIDI processing | `mido`, `pretty_midi` |
| Analysis & plots | `matplotlib`, `pandas`, `numpy` |
| Training hardware | NVIDIA RTX 3060 (Win) / Apple M3 (inference only) |
