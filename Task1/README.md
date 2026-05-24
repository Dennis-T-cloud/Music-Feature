# Task1: Symbolic Unconditioned Music Generation

This folder contains the reproducible pipeline and final artifacts for Task 1.

The method is a two-stage fine-tuning setup:

```text
Aria pretrained symbolic MIDI model
  -> Stage 1: LoRA fine-tuning on MAESTRO
  -> Stage 2: continued LoRA fine-tuning on Chopin Etude subset
```

## Folder Structure

```text
Task1/
  dataset/   Dataset download instructions and Chopin Etude metadata
  train/     Training and generation scripts
  result/    Final adapters, training curves, logs, plots, and generated MIDI
```

## What Is Included

- Training scripts for Aria LoRA fine-tuning
- Dataset acquisition instructions
- Pretrained model acquisition instructions
- Two-stage training commands
- Best LoRA adapters from Stage 1 and Stage 2
- Training logs and curves
- Baseline-vs-fine-tuned comparison plot
- Generated MIDI sample from the fine-tuned model

## What Is Not Included

The full MAESTRO dataset and the Aria pretrained base model are not committed because they are large external assets.
See `dataset/README.md` and `train/README.md` for how to download them.
