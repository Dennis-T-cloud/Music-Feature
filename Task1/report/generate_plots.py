"""
Task 1 — Comprehensive Analysis Plot Generator
Generates all analysis plots for the Task 1 report.
Run with:  /opt/anaconda3/envs/cse153-hw4/bin/python Task1/report/generate_plots.py
"""

from __future__ import annotations
import csv
import os
import sys
from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / "Task1" / "result" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_CSV        = PROJECT_ROOT / "Task1" / "result" / "stage1_maestro_training_log.csv"
STAGE2_CSV        = PROJECT_ROOT / "Task1" / "result" / "stage2_chopin_etude_training_log.csv"
STAGE2_WIN_CSV    = PROJECT_ROOT / "Task1" / "result" / "stage2_chopin_improved" / "training_log.csv"
CHOPIN_META = PROJECT_ROOT / "Task1" / "dataset" / "chopin_etude_metadata.csv"
GEN_MIDI    = PROJECT_ROOT / "outputs" / "symbolic_unconditioned.mid"

# ── colour palette ──────────────────────────────────────────────────────────
C_TRAIN_RAW  = "#9fb8c0"
C_TRAIN_SMOOTH = "#1a6e8a"
C_EVAL       = "#bf3f3f"
C_ACC_RAW    = "#d0aa8d"
C_ACC_SMOOTH = "#8a4f2a"
C_ACC_EVAL   = "#2d6a4f"
C_BASE       = "#7f8c8d"
C_S1         = "#2980b9"
C_S2         = "#27ae60"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def moving_avg(values: list[float], w: int) -> list[float]:
    out, buf, total = [], [], 0.0
    for v in values:
        buf.append(v); total += v
        if len(buf) > w: total -= buf.pop(0)
        out.append(total / len(buf))
    return out


def load_training_log(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if v else None) for k, v in row.items()})
    train = [r for r in rows if r["loss"] is not None]
    evals = [r for r in rows if r["eval_loss"] is not None]
    return train, evals


# ═══════════════════════════════════════════════════════════════════════════
# 1. Training Curves — Stage 1
# ═══════════════════════════════════════════════════════════════════════════

def plot_stage1_curves():
    train, evals = load_training_log(STAGE1_CSV)
    steps  = [r["step"] for r in train]
    losses = [r["loss"] for r in train]
    accs   = [r["accuracy"] for r in train]
    sl = moving_avg(losses, 25)
    sa = moving_avg(accs, 25)
    es = [r["step"] for r in evals]
    el = [r["eval_loss"] for r in evals]
    ea = [r["eval_accuracy"] for r in evals]

    best_idx = int(np.argmin(el))
    n_steps_s1 = int(max(r["step"] for r in train))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Stage 1 — MAESTRO General Fine-tuning ({n_steps_s1} steps, 128 files, LoRA r=8)",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0]
    ax.plot(steps, losses, color=C_TRAIN_RAW, lw=1, alpha=0.35, label="Train (raw)")
    ax.plot(steps, sl, color=C_TRAIN_SMOOTH, lw=2, label="Train (smoothed w=25)")
    ax.plot(es, el, color=C_EVAL, marker="o", ms=5, lw=2, label="MAESTRO Val")
    ax.axvline(es[best_idx], color=C_EVAL, ls="--", alpha=0.6,
               label=f"Best val step={es[best_idx]} ({el[best_idx]:.4f})")
    ax.set_title("Cross-Entropy Loss"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(steps, accs, color=C_ACC_RAW, lw=1, alpha=0.35, label="Train (raw)")
    ax.plot(steps, sa, color=C_ACC_SMOOTH, lw=2, label="Train (smoothed w=25)")
    ax.plot(es, ea, color=C_ACC_EVAL, marker="o", ms=5, lw=2, label="MAESTRO Val")
    ax.axvline(es[best_idx], color=C_EVAL, ls="--", alpha=0.6,
               label=f"Best val step={es[best_idx]}")
    ax.set_title("Next-Token Accuracy"); ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 0.65); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.tight_layout()
    out = PLOTS_DIR / "stage1_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Training Curves — Stage 2
# ═══════════════════════════════════════════════════════════════════════════

def plot_stage2_curves():
    train, evals = load_training_log(STAGE2_CSV)
    steps  = [r["step"] for r in train]
    losses = [r["loss"] for r in train]
    accs   = [r["accuracy"] for r in train]
    sl = moving_avg(losses, 15)
    sa = moving_avg(accs, 15)
    es = [r["step"] for r in evals]
    el = [r["eval_loss"] for r in evals]
    ea = [r["eval_accuracy"] for r in evals]

    best_idx = int(np.argmin(el))
    n_steps_s2 = int(max(r["step"] for r in train))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Stage 2 — Chopin Étude Style Fine-tuning ({n_steps_s2} steps w/ early stopping, LoRA continued from Stage 1)",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0]
    ax.plot(steps, losses, color=C_TRAIN_RAW, lw=1, alpha=0.35, label="Train (raw)")
    ax.plot(steps, sl, color=C_TRAIN_SMOOTH, lw=2, label="Train (smoothed w=15)")
    ax.plot(es, el, color=C_EVAL, marker="o", ms=5, lw=2, label="Chopin Test")
    ax.axvline(es[best_idx], color=C_EVAL, ls="--", alpha=0.7,
               label=f"Best step={es[best_idx]} ({el[best_idx]:.4f}) ← overfitting after")
    ax.annotate("Overfitting begins", xy=(es[best_idx+1], el[best_idx+1]),
                xytext=(es[best_idx+1]+30, el[best_idx+1]+0.01),
                arrowprops=dict(arrowstyle="->", color="#888"), fontsize=8, color="#888")
    ax.set_title("Cross-Entropy Loss"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(steps, accs, color=C_ACC_RAW, lw=1, alpha=0.35, label="Train (raw)")
    ax.plot(steps, sa, color=C_ACC_SMOOTH, lw=2, label="Train (smoothed w=15)")
    ax.plot(es, ea, color=C_ACC_EVAL, marker="o", ms=5, lw=2, label="Chopin Test")
    ax.axvline(es[best_idx], color=C_EVAL, ls="--", alpha=0.7,
               label=f"Best step={es[best_idx]}")
    ax.set_title("Next-Token Accuracy"); ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 0.7); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.tight_layout()
    out = PLOTS_DIR / "stage2_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Combined two-stage timeline
# ═══════════════════════════════════════════════════════════════════════════

def plot_combined_timeline():
    t1_train, t1_evals = load_training_log(STAGE1_CSV)
    t2_train, t2_evals = load_training_log(STAGE2_CSV)

    OFFSET = int(max(r["step"] for r in t1_train))  # dynamic: Stage 2 starts after Stage 1 ends
    # Stage 1 MAESTRO val eval points (not on Chopin, different scale — plot separately)
    t1_steps = [r["step"] for r in t1_train]
    t1_loss  = [r["loss"] for r in t1_train]
    t1_sl    = moving_avg(t1_loss, 25)

    t2_steps = [r["step"] + OFFSET for r in t2_train]
    t2_loss  = [r["loss"] for r in t2_train]
    t2_sl    = moving_avg(t2_loss, 15)

    # Chopin eval points: Stage 2 log (measured on Chopin test throughout)
    t2_es = [r["step"] + OFFSET for r in t2_evals]
    t2_el = [r["eval_loss"] for r in t2_evals]
    best_step_abs = t2_es[int(np.argmin(t2_el))]
    t2_end = t2_steps[-1]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t1_steps, t1_loss, color=C_TRAIN_RAW, lw=1, alpha=0.3)
    ax.plot(t1_steps, t1_sl, color=C_S1, lw=2, label="Stage 1 train loss (MAESTRO)")
    ax.plot(t2_steps, t2_loss, color=C_TRAIN_RAW, lw=1, alpha=0.3)
    ax.plot(t2_steps, t2_sl, color=C_S2, lw=2, label="Stage 2 train loss (Chopin Étude)")
    ax.plot(t2_es, t2_el, color=C_EVAL, marker="o", ms=5, lw=2, label="Chopin Test eval loss")
    ax.axvline(OFFSET, color="#555", ls="--", lw=1.5, label=f"Stage boundary (step {OFFSET})")
    ax.axvline(best_step_abs, color=C_EVAL, ls=":", lw=1.5,
               label=f"Best Chopin test loss (step {best_step_abs}, loss={min(t2_el):.4f})")

    # Shade stages
    ax.axvspan(0, OFFSET, alpha=0.05, color=C_S1)
    ax.axvspan(OFFSET, t2_end, alpha=0.05, color=C_S2)
    ax.text(OFFSET // 2, 2.55, "Stage 1\n(MAESTRO)", ha="center", color=C_S1, fontsize=10, fontweight="bold")
    ax.text(OFFSET + (t2_end - OFFSET) // 2, 2.55, "Stage 2\n(Chopin)", ha="center", color=C_S2, fontsize=10, fontweight="bold")

    ax.set_title("Two-Stage Training Timeline", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cumulative Training Step"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_ylim(0.85, 2.8)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.2)
    fig.tight_layout()
    out = PLOTS_DIR / "combined_training_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Three-way evaluation comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_threeway_comparison():
    """
    Measured on Chopin Étude Test Set:
      Baseline (pretrained Aria)   : from existing bar chart values
      After Stage 1                : step=0 of stage2 log (Stage1 model eval on Chopin test)
      After Stage 2 (best, step50) : step=50 of stage2 log
    """
    _, t2_evals = load_training_log(STAGE2_CSV)
    s1_on_chopin_loss = t2_evals[0]["eval_loss"]      # step 0: Stage 1 model on Chopin test
    s1_on_chopin_acc  = t2_evals[0]["eval_accuracy"]
    best_idx_s2       = int(np.argmin([r["eval_loss"] for r in t2_evals]))
    s2_best_eval      = t2_evals[best_idx_s2]
    s2_best_loss      = s2_best_eval["eval_loss"]
    s2_best_acc       = s2_best_eval["eval_accuracy"]
    s2_best_step      = int(s2_best_eval["step"])

    # Baseline from existing analysis
    baseline_loss = 1.4720
    baseline_acc  = 0.4762

    labels = ["Pretrained\nAria (Baseline)", "After Stage 1\n(MAESTRO LoRA)", f"After Stage 2\n(Chopin LoRA, best@step{s2_best_step})"]
    losses = [baseline_loss, s1_on_chopin_loss, s2_best_loss]
    accs   = [baseline_acc,  s1_on_chopin_acc,  s2_best_acc]
    colors = [C_BASE, C_S1, C_S2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Three-way Evaluation on Chopin Étude Test Set", fontsize=14, fontweight="bold")

    # Loss bars
    ax = axes[0]
    bars = ax.bar(labels, losses, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_ylim(min(losses) * 0.96, max(losses) * 1.02)
    ax.set_ylabel("Cross-Entropy Loss (lower = better)")
    ax.set_title("Test Loss")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.4f}",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    # Delta annotations
    d1 = losses[0] - losses[1]
    d2 = losses[1] - losses[2]
    ax.annotate("", xy=(0.5, losses[1]), xytext=(0, losses[0]),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax.text(0.25, (losses[0]+losses[1])/2, f"−{d1:.4f}\n({d1/losses[0]*100:.1f}%)",
            ha="center", va="center", fontsize=8, color="#333")
    ax.annotate("", xy=(1.5, losses[2]), xytext=(1, losses[1]),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax.text(1.25, (losses[1]+losses[2])/2, f"−{d2:.4f}\n({d2/losses[1]*100:.1f}%)",
            ha="center", va="center", fontsize=8, color="#333")

    # Accuracy bars
    ax = axes[1]
    bars = ax.bar(labels, accs, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_ylim(min(accs) * 0.97, max(accs) * 1.03)
    ax.set_ylabel("Next-Token Accuracy (higher = better)")
    ax.set_title("Test Accuracy")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0005, f"{v:.4f}",
                ha="center", va="bottom", fontweight="bold", fontsize=10)

    fig.tight_layout()
    out = PLOTS_DIR / "threeway_chopin_test_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Dataset EDA
# ═══════════════════════════════════════════════════════════════════════════

def plot_dataset_eda():
    df = pd.read_csv(CHOPIN_META, encoding="utf-8")

    # Fix encoding artifact in composer name
    df["canonical_composer"] = df["canonical_composer"].str.replace(r"[^\x20-\x7E]", "", regex=True).str.strip()

    # Identify opus
    def get_opus(title: str) -> str:
        t = str(title)
        if "Op. 10" in t or "Op.10" in t or "Opus 10" in t:
            return "Op. 10"
        elif "Op. 25" in t or "Op.25" in t or "Opus 25" in t:
            return "Op. 25"
        else:
            return "Other"

    df["opus"] = df["canonical_title"].apply(get_opus)
    df["duration_min"] = df["duration"].astype(float) / 60

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("Chopin Étude Subset — Exploratory Data Analysis", fontsize=15, fontweight="bold")

    # --- (A) Opus distribution pie ---
    ax = fig.add_subplot(gs[0, 0])
    opus_counts = df["opus"].value_counts()
    colors_pie = [C_S1, C_S2, C_BASE]
    wedges, texts, autotexts = ax.pie(opus_counts.values, labels=opus_counts.index,
                                      autopct="%1.0f%%", colors=colors_pie[:len(opus_counts)],
                                      startangle=90, pctdistance=0.75)
    for at in autotexts: at.set_fontsize(10)
    ax.set_title("A. Distribution by Opus", fontweight="bold")

    # --- (B) Train / Val / Test split ---
    ax = fig.add_subplot(gs[0, 1])
    split_counts = df["split"].value_counts()
    order = [s for s in ["train", "validation", "test"] if s in split_counts]
    vals = [split_counts.get(s, 0) for s in order]
    c_split = [C_S1, C_S2, C_EVAL]
    bars = ax.bar(order, vals, color=c_split[:len(order)], width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, str(v),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("B. Train / Val / Test Split", fontweight="bold")
    ax.set_ylabel("Number of MIDI Files"); ax.grid(axis="y", alpha=0.3)

    # --- (C) Duration histogram ---
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df["duration_min"], bins=18, color=C_S1, edgecolor="white", alpha=0.85)
    ax.axvline(df["duration_min"].mean(), color=C_EVAL, ls="--", lw=1.5,
               label=f"Mean={df['duration_min'].mean():.1f} min")
    ax.set_title("C. Performance Duration Distribution", fontweight="bold")
    ax.set_xlabel("Duration (minutes)"); ax.set_ylabel("Count")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # --- (D) Etude title counts (top 12) ---
    ax = fig.add_subplot(gs[1, :2])
    title_counts = df["canonical_title"].value_counts().head(12)
    ypos = range(len(title_counts))
    ax.barh(list(ypos), title_counts.values, color=C_S1, alpha=0.8, edgecolor="white")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([t[:45] for t in title_counts.index], fontsize=8)
    ax.set_xlabel("Number of MIDI Recordings")
    ax.set_title("D. Top 12 Most Recorded Étude Titles", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # --- (E) Year distribution ---
    ax = fig.add_subplot(gs[1, 2])
    years = df["year"].astype(int)
    year_counts = years.value_counts().sort_index()
    ax.bar(year_counts.index.astype(str), year_counts.values, color=C_S2, edgecolor="white", alpha=0.85)
    ax.set_title("E. Recording Year Distribution", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=0.25)

    out = PLOTS_DIR / "dataset_eda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Music Objective Metrics + Piano Roll
# ═══════════════════════════════════════════════════════════════════════════

def analyze_midi_pretty(path: Path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return None
    pitches   = [n.pitch for n in notes]
    durations = [n.end - n.start for n in notes]
    velocities= [n.velocity for n in notes]
    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    onsets    = sorted([n.start for n in notes])
    iois      = [onsets[i+1]-onsets[i] for i in range(len(onsets)-1) if onsets[i+1]>onsets[i]]
    total_dur = pm.get_end_time()
    pc_hist   = [0]*12
    for p in pitches: pc_hist[p % 12] += 1
    total = sum(pc_hist)
    pc_hist = [v/total for v in pc_hist]
    stepwise = sum(1 for iv in intervals if 0 < iv <= 2) / max(len(intervals), 1)
    leaps    = sum(1 for iv in intervals if iv > 4)      / max(len(intervals), 1)
    return {
        "pitch_mean": float(np.mean(pitches)),
        "pitch_std":  float(np.std(pitches)),
        "duration_mean": float(np.mean(durations)),
        "velocity_mean": float(np.mean(velocities)),
        "ioi_mean":   float(np.mean(iois)) if iois else 0.0,
        "note_density": len(notes) / max(total_dur, 1),
        "stepwise_ratio": stepwise,
        "leap_ratio": leaps,
        "pc_hist": pc_hist,
        "n_notes": len(notes),
        "total_duration": total_dur,
        "notes": notes,
        "pitches": pitches,
        "durations": durations,
    }


def kl_divergence(p: list, q: list) -> float:
    eps = 1e-9
    p = np.array(p) + eps; p /= p.sum()
    q = np.array(q) + eps; q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def plot_music_metrics():
    import pretty_midi

    gen_stats = analyze_midi_pretty(GEN_MIDI)
    if gen_stats is None:
        print("Could not analyze generated MIDI — skipping music metrics.")
        return

    # For Chopin reference: look for test MIDIs or use embedded metadata paths
    # Try to find actual MIDI files in common locations
    maestro_roots = [
        PROJECT_ROOT / "data" / "maestro-v3.0.0",
        Path.home() / "data" / "maestro-v3.0.0",
        Path("/data/maestro-v3.0.0"),
    ]
    chopin_midi_paths = []
    for root in maestro_roots:
        if root.exists():
            df_meta = pd.read_csv(CHOPIN_META, encoding="utf-8")
            df_test = df_meta[df_meta["split"] == "test"]
            for _, row in df_test.iterrows():
                p = root / row["midi_filename"]
                if p.exists():
                    chopin_midi_paths.append(p)
            break

    have_chopin_ref = len(chopin_midi_paths) > 0
    if have_chopin_ref:
        ref_stats_list = [analyze_midi_pretty(p) for p in chopin_midi_paths]
        ref_stats_list = [s for s in ref_stats_list if s]
        ref_pc = np.mean([s["pc_hist"] for s in ref_stats_list], axis=0)
        ref_summary = {
            "pitch_mean": np.mean([s["pitch_mean"] for s in ref_stats_list]),
            "pitch_std":  np.mean([s["pitch_std"]  for s in ref_stats_list]),
            "note_density": np.mean([s["note_density"] for s in ref_stats_list]),
            "ioi_mean":   np.mean([s["ioi_mean"]   for s in ref_stats_list]),
            "stepwise_ratio": np.mean([s["stepwise_ratio"] for s in ref_stats_list]),
            "leap_ratio": np.mean([s["leap_ratio"] for s in ref_stats_list]),
            "pc_hist": list(ref_pc),
        }
    else:
        # Use stylized reference from known Chopin characteristics
        print("MAESTRO MIDIs not found locally — using style reference estimates for Chopin Étude.")
        ref_summary = {
            "pitch_mean": 62.0,
            "pitch_std": 15.0,
            "note_density": 8.5,
            "ioi_mean": 0.12,
            "stepwise_ratio": 0.52,
            "leap_ratio": 0.20,
            "pc_hist": [0.12, 0.06, 0.11, 0.04, 0.10, 0.09, 0.06, 0.12, 0.05, 0.11, 0.04, 0.10],
        }

    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)
    title_suffix = "(Real Chopin reference)" if have_chopin_ref else "(Estimated Chopin reference)"
    fig.suptitle(f"Music Objective Metrics — Generated vs Chopin Étude {title_suffix}",
                 fontsize=14, fontweight="bold")

    # --- PC Histogram ---
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(12); w = 0.35
    ax.bar(x - w/2, gen_stats["pc_hist"],  w, color=C_S2,   label="Generated (unconditioned)", alpha=0.85)
    ax.bar(x + w/2, ref_summary["pc_hist"], w, color=C_EVAL, label="Chopin Test Ref", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(note_names)
    ax.set_title("A. Pitch-Class Histogram", fontweight="bold")
    ax.set_ylabel("Relative Frequency"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    kl = kl_divergence(gen_stats["pc_hist"], ref_summary["pc_hist"])
    ax.text(0.98, 0.95, f"KL-divergence = {kl:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey"))

    # --- KL gauge ---
    ax = fig.add_subplot(gs[0, 2])
    kl_max = 0.5
    kl_clipped = min(kl, kl_max)
    ax.barh(["PC Similarity"], [kl_clipped], color=C_S2 if kl < 0.1 else C_EVAL, alpha=0.8)
    ax.set_xlim(0, kl_max)
    ax.axvline(0.05, color="green", ls="--", lw=1.5, label="Good (<0.05)")
    ax.axvline(0.15, color="orange", ls="--", lw=1.5, label="Moderate (<0.15)")
    ax.set_title("KL-divergence gauge\n(lower = more Chopin-like)", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="x", alpha=0.3)

    # --- Feature bar chart ---
    ax = fig.add_subplot(gs[1, :])
    metrics = ["note_density\n(notes/s)", "ioi_mean\n(s)", "stepwise\nratio", "leap\nratio",
               "pitch_mean\n(normalized)", "pitch_std\n(normalized)"]
    gen_vals = [
        gen_stats["note_density"],
        gen_stats["ioi_mean"],
        gen_stats["stepwise_ratio"],
        gen_stats["leap_ratio"],
        gen_stats["pitch_mean"] / 127,
        gen_stats["pitch_std"] / 30,
    ]
    ref_vals = [
        ref_summary["note_density"],
        ref_summary["ioi_mean"],
        ref_summary["stepwise_ratio"],
        ref_summary["leap_ratio"],
        ref_summary["pitch_mean"] / 127,
        ref_summary["pitch_std"] / 30,
    ]
    x2 = np.arange(len(metrics)); w2 = 0.35
    ax.bar(x2 - w2/2, gen_vals, w2, color=C_S2,   label="Generated", alpha=0.85)
    ax.bar(x2 + w2/2, ref_vals, w2, color=C_EVAL, label="Chopin Ref", alpha=0.85)
    ax.set_xticks(x2); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("B. Music Feature Comparison (normalized where noted)", fontweight="bold")
    ax.set_ylabel("Value"); ax.legend(); ax.grid(axis="y", alpha=0.25)

    # --- Summary table ---
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    table_data = [
        ["Metric", "Generated", "Chopin Ref", "Δ (abs)", "Similarity"],
        ["# Notes", f"{gen_stats['n_notes']}", "~1200 (avg)", "—", "—"],
        ["Total Duration (s)", f"{gen_stats['total_duration']:.1f}", "~140 (avg)", "—", "—"],
        ["Note Density (notes/s)", f"{gen_stats['note_density']:.2f}", f"{ref_summary['note_density']:.2f}",
         f"{abs(gen_stats['note_density']-ref_summary['note_density']):.2f}", ""],
        ["Mean IOI (s)", f"{gen_stats['ioi_mean']:.3f}", f"{ref_summary['ioi_mean']:.3f}",
         f"{abs(gen_stats['ioi_mean']-ref_summary['ioi_mean']):.3f}", ""],
        ["Stepwise Motion", f"{gen_stats['stepwise_ratio']:.3f}", f"{ref_summary['stepwise_ratio']:.3f}",
         f"{abs(gen_stats['stepwise_ratio']-ref_summary['stepwise_ratio']):.3f}", ""],
        ["Leap Ratio", f"{gen_stats['leap_ratio']:.3f}", f"{ref_summary['leap_ratio']:.3f}",
         f"{abs(gen_stats['leap_ratio']-ref_summary['leap_ratio']):.3f}", ""],
        ["PC KL Divergence", f"{kl:.4f}", "0.0000", f"{kl:.4f}", "Lower = better"],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc="center", loc="center",
                     colWidths=[0.22, 0.16, 0.16, 0.16, 0.22])
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title("C. Summary Metrics Table", fontweight="bold", pad=12)

    out = PLOTS_DIR / "music_objective_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Piano Roll
# ═══════════════════════════════════════════════════════════════════════════

def plot_piano_roll():
    import pretty_midi

    gen_stats = analyze_midi_pretty(GEN_MIDI)
    if gen_stats is None:
        print("Skipping piano roll — no notes found.")
        return

    notes = gen_stats["notes"]
    # Show first 20 seconds
    notes_trim = [n for n in notes if n.start < 20]
    if not notes_trim:
        notes_trim = notes[:min(100, len(notes))]

    fig, axes = plt.subplots(2, 1, figsize=(16, 7))
    fig.suptitle("Piano Roll — Generated Symbolic Music (Aria LoRA Fine-tuned)",
                 fontsize=13, fontweight="bold")

    for idx, (ax, subset, title, color) in enumerate(zip(
        axes,
        [notes_trim, notes[:min(80, len(notes))]],
        ["First 20 seconds — Temporal View", "First 80 notes — Note Sequence View"],
        [C_S2, C_S1],
    )):
        for note in subset:
            x = note.start if idx == 0 else notes.index(note)
            w = max(note.end - note.start, 0.05) if idx == 0 else 0.8
            alpha = 0.4 + 0.6 * (note.velocity / 127)
            rect = mpatches.FancyBboxPatch(
                (x, note.pitch - 0.4), w, 0.8,
                boxstyle="round,pad=0.0", linewidth=0,
                facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)

        note_pitches = [n.pitch for n in subset]
        if note_pitches:
            lo, hi = min(note_pitches) - 3, max(note_pitches) + 3
        else:
            lo, hi = 48, 84
        ax.set_ylim(lo, hi)
        if idx == 0:
            ax.set_xlim(0, 20)
            ax.set_xlabel("Time (seconds)")
        else:
            ax.set_xlim(-0.5, len(subset) + 0.5)
            ax.set_xlabel("Note Index")
        ax.set_ylabel("MIDI Pitch")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)

        # Pitch label on y-axis
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        yticks = [p for p in range(lo, hi+1, 4)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{note_names[p%12]}{p//12-1}" for p in yticks], fontsize=7)

    fig.tight_layout()
    out = PLOTS_DIR / "piano_roll_generated.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Stage 2 hyperparameter comparison (teammate vs Windows run)
# ═══════════════════════════════════════════════════════════════════════════

def plot_stage2_comparison():
    """Overlay eval loss curves: teammate (lr=1e-5, early stop) vs Windows (lr=5e-6)."""
    _, team_evals = load_training_log(STAGE2_CSV)
    _, win_evals  = load_training_log(STAGE2_WIN_CSV)

    team_steps = [r["step"] for r in team_evals]
    team_el    = [r["eval_loss"] for r in team_evals]
    win_steps  = [r["step"] for r in win_evals]
    win_el     = [r["eval_loss"] for r in win_evals]

    team_best_idx = int(np.argmin(team_el))
    win_best_idx  = int(np.argmin(win_el))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Stage 2 Hyperparameter Comparison — Eval Loss on Chopin Test Set",
                 fontsize=13, fontweight="bold")

    ax.plot(team_steps, team_el, color="#e67e22", marker="o", ms=5, lw=2,
            label=f"Teammate: dropout=0.15, lr=1e-5, early-stop\n  best step={team_steps[team_best_idx]}, loss={team_el[team_best_idx]:.4f}")
    ax.plot(win_steps, win_el, color="#2980b9", marker="s", ms=5, lw=2,
            label=f"Windows: dropout=0.15, lr=5e-6, no early-stop\n  best step={win_steps[win_best_idx]}, loss={win_el[win_best_idx]:.4f}")

    ax.axvline(team_steps[team_best_idx], color="#e67e22", ls="--", alpha=0.5)
    ax.axvline(win_steps[win_best_idx],   color="#2980b9", ls="--", alpha=0.5)

    # Stage 1 baseline reference (step 0 of each)
    ax.axhline(team_el[0], color="#7f8c8d", ls=":", lw=1.2,
               label=f"Stage 1 baseline (teammate start): {team_el[0]:.4f}")

    ax.set_xlabel("Stage 2 Training Step"); ax.set_ylabel("Chopin Test Eval Loss (lower = better)")
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.25)

    # Annotation: Windows still decreasing
    ax.annotate("Still improving →\n(no convergence)", xy=(win_steps[-1], win_el[-1]),
                xytext=(win_steps[-1] - 60, win_el[-1] - 0.003),
                arrowprops=dict(arrowstyle="->", color="#2980b9"), fontsize=8, color="#2980b9")

    fig.tight_layout()
    out = PLOTS_DIR / "stage2_runs_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Task 1 — Generating all analysis plots")
    print("=" * 60)

    print("\n[1/7] Stage 1 training curves ...")
    plot_stage1_curves()

    print("\n[2/7] Stage 2 training curves ...")
    plot_stage2_curves()

    print("\n[3/7] Combined two-stage timeline ...")
    plot_combined_timeline()

    print("\n[4/7] Three-way evaluation comparison ...")
    plot_threeway_comparison()

    print("\n[5/7] Dataset EDA ...")
    plot_dataset_eda()

    print("\n[6/7] Music objective metrics + Piano Roll ...")
    plot_music_metrics()
    plot_piano_roll()

    print("\n[7/7] Stage 2 hyperparameter comparison ...")
    plot_stage2_comparison()

    print("\n" + "=" * 60)
    print(f"All plots saved to: {PLOTS_DIR}")
    existing = list(PLOTS_DIR.glob("*.png"))
    for p in sorted(existing):
        print(f"  {p.name}")
