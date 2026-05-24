# Results

This folder contains the final Task 1 artifacts.

## Important Files

- `stage1_maestro_best_adapter/`: best LoRA adapter after MAESTRO general fine-tuning
- `stage2_chopin_etude_best_adapter/`: final best LoRA adapter after Chopin Etude style fine-tuning
- `plots/baseline_vs_twostage_zoomed_bar.png`: baseline-vs-fine-tuned comparison plot
- `stage1_maestro_training_log.csv`: Stage 1 log
- `stage2_chopin_etude_training_log.csv`: Stage 2 log
- `midi/generated_twostage_chopin_best.mid`: generated MIDI from the final fine-tuned model
- `two_stage_maestro_then_chopin_summary.md`: detailed result summary

## Using the Final Adapter

To generate MIDI with the final adapter, first download the Aria base model as described in `../train/README.md`.
Then run:

```powershell
python Task1\train\aria_generate.py `
  --adapter-dir Task1\result\stage2_chopin_etude_best_adapter `
  --output outputs\generated_twostage_chopin_best.mid `
  --max-length 1024 `
  --temperature 0.9 `
  --top-p 0.95 `
  --seed 253
```
