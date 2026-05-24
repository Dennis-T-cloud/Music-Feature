# Two-Stage Aria Fine-Tuning Summary

This experiment follows the pipeline:

```text
Aria pretrained model -> MAESTRO general LoRA fine-tuning -> Chopin Etude style LoRA fine-tuning
```

## Stage 1: General MAESTRO Fine-Tuning

- Base model: Aria pretrained symbolic MIDI model
- Method: LoRA fine-tuning
- Train data: 128 MAESTRO train MIDI files
- Evaluation data: 32 MAESTRO validation MIDI files
- Best validation result: step 250

## Stage 2: Chopin Etude Style Fine-Tuning

- Starting point: best Stage 1 MAESTRO LoRA checkpoint
- Method: continued LoRA fine-tuning
- Train data: MAESTRO train split, Chopin Etude subset
- Test data: MAESTRO test split, Chopin Etude subset
- Best test-loss result: step 50

## Interpretation

The two-stage training improves the pretrained Aria baseline on the Chopin Etude test subset.
Most of the improvement comes from the general MAESTRO adaptation stage, while the Chopin-specific
stage gives the best test loss but does not improve accuracy beyond the Stage 1 checkpoint.

The zoomed bar chart for this comparison is saved as:

```text
Task1/result/plots/baseline_vs_twostage_zoomed_bar.png
```

The best model to report for the two-stage setup is:

```text
Task1/result/stage2_chopin_etude_best_adapter
```

The corresponding training logs are saved in:

```text
Task1/result/stage1_maestro_training_log.csv
Task1/result/stage2_chopin_etude_training_log.csv
```
