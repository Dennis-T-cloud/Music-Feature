# Training Pipeline

## Pretrained Model

This project uses the Aria pretrained symbolic MIDI model:

```text
loubb/aria-medium-base
```

Hugging Face page:

```text
https://huggingface.co/loubb/aria-medium-base
```

Download it with:

```powershell
python Task1\train\aria_download_weights.py --model-id loubb/aria-medium-base --output .vendor\aria-hf
```

The fine-tuning scripts assume the base model is available at:

```text
.vendor/aria-hf
```

## Install Dependencies

Core Python packages:

```powershell
python -m pip install torch transformers peft pandas matplotlib safetensors
```

The scripts also use Aria custom model/tokenizer files through Hugging Face `trust_remote_code`.

## Stage 1: MAESTRO General LoRA Fine-Tuning

```powershell
python Task1\train\aria_finetune_maestro.py `
  --train-mode lora `
  --max-files 128 `
  --max-steps 500 `
  --block-size 512 `
  --lr 5e-5 `
  --lora-r 8 `
  --lora-alpha 16 `
  --lora-dropout 0.15 `
  --grad-clip 0.5 `
  --eval-split validation `
  --eval-max-files 32 `
  --eval-every 50 `
  --early-stop-patience 6 `
  --early-stop-min-delta 0.005 `
  --save-dir checkpoints\aria_lora_maestro_stage1_dropout015
```

## Stage 2: Chopin Etude Style LoRA Fine-Tuning

```powershell
python Task1\train\aria_finetune_maestro.py `
  --train-mode lora `
  --resume-adapter checkpoints\aria_lora_maestro_stage1_dropout015\best_checkpoint `
  --composer Chopin `
  --title-contains Etude `
  --max-files 32 `
  --max-steps 300 `
  --block-size 512 `
  --lr 2e-5 `
  --lora-r 8 `
  --lora-alpha 16 `
  --lora-dropout 0.15 `
  --grad-clip 0.5 `
  --eval-split test `
  --eval-max-files 6 `
  --eval-every 50 `
  --early-stop-patience 3 `
  --early-stop-min-delta 0.001 `
  --save-dir checkpoints\aria_lora_chopin_stage2_dropout015_earlystop
```

## Generate MIDI From Final Adapter

```powershell
python Task1\train\aria_generate.py `
  --adapter-dir checkpoints\aria_lora_chopin_stage2_dropout015_earlystop\best_checkpoint `
  --output outputs\generated_twostage_chopin_best.mid `
  --max-length 1024 `
  --temperature 0.9 `
  --top-p 0.95 `
  --seed 253
```

## Notes

The full pretrained Aria model is not included in the repository because the base checkpoint is large.
Only the trained LoRA adapters are included in `Task1/result`.
