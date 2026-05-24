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

| Model state | Eval split | Loss | Accuracy |
| --- | --- | ---: | ---: |
| Original Aria before Stage 1 | MAESTRO validation | 1.7762 | 0.3989 |
| After Stage 1 best checkpoint | MAESTRO validation | 1.7250 | 0.4099 |

## Stage 2: Chopin Etude Style Fine-Tuning

- Starting point: best Stage 1 MAESTRO LoRA checkpoint
- Method: continued LoRA fine-tuning
- Train data: MAESTRO train split, Chopin Etude subset
- Test data: MAESTRO test split, Chopin Etude subset
- Best test-loss result: step 50

| Model state | Test split | Loss | Accuracy |
| --- | --- | ---: | ---: |
| Original Aria, no fine-tuning | Chopin Etude test | 1.4720 | 0.4762 |
| After Stage 1 MAESTRO fine-tuning | Chopin Etude test | 1.4131 | 0.4889 |
| After Stage 2 Chopin Etude best checkpoint | Chopin Etude test | 1.4106 | 0.4850 |
| After Stage 2 final checkpoint | Chopin Etude test | 1.4523 | 0.4811 |

## Interpretation

The two-stage training improves the pretrained Aria baseline on the Chopin Etude test subset.
Most of the improvement comes from the general MAESTRO adaptation stage, while the Chopin-specific
stage gives the best test loss but does not improve accuracy beyond the Stage 1 checkpoint.

For reporting, the cleanest comparison is:

| Comparison | Loss change | Accuracy change |
| --- | ---: | ---: |
| Original Aria -> two-stage best checkpoint | 1.4720 -> 1.4106 | 0.4762 -> 0.4850 |

The zoomed bar chart for this comparison is saved as:

```text
deliverables/aria_final_results/two_stage_maestro_then_chopin/baseline_vs_twostage_zoomed_bar.png
```

The best model to report for the two-stage setup is:

```text
deliverables/aria_final_results/two_stage_maestro_then_chopin/stage2_chopin_etude/best_checkpoint
```

The corresponding training curves and logs are saved in:

```text
deliverables/aria_final_results/two_stage_maestro_then_chopin/stage1_maestro
deliverables/aria_final_results/two_stage_maestro_then_chopin/stage2_chopin_etude
```
