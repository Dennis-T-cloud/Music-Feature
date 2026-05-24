from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
aria_utils_path = PROJECT_ROOT / ".vendor" / "aria-utils-src"
if aria_utils_path.exists():
    sys.path.insert(0, str(aria_utils_path))

pydeps_path = PROJECT_ROOT / ".vendor" / "pydeps"
if pydeps_path.exists():
    sys.path.append(str(pydeps_path))


def read_midi_paths(
    maestro_root: Path,
    split: str,
    max_files: int | None,
    composer: str = "",
    title_contains: str = "",
) -> list[Path]:
    rows = []
    composer = composer.lower().strip()
    title_contains = title_contains.lower().strip()
    with (maestro_root / "maestro-v3.0.0.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if composer and composer not in row["canonical_composer"].lower():
                continue
            if title_contains and title_contains not in row["canonical_title"].lower():
                continue
            path = maestro_root / row["midi_filename"]
            if path.exists():
                rows.append(path)
    random.shuffle(rows)
    return rows[:max_files] if max_files else rows


def set_trainable_params(model, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if mode == "lm_head":
        for param in model.lm_head.parameters():
            param.requires_grad = True
    elif mode == "last_block":
        for param in model.model.encode_layers[-1].parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
    elif mode == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown train mode: {mode}")


def next_token_accuracy(logits, labels) -> float:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels >= 0
    correct = (preds[mask] == shift_labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total else 0.0


def write_training_log(log_rows: list[dict[str, float | None]], save_dir: Path) -> None:
    log_path = save_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "loss", "accuracy", "eval_loss", "eval_accuracy"],
        )
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Saved training log to {log_path}")


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed = []
    running = 0.0
    queue = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        smoothed.append(running / len(queue))
    return smoothed


def write_training_curves(
    log_rows: list[dict[str, float | None]], save_dir: Path, smooth_window: int
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not create training curves because matplotlib is unavailable: {exc}")
        return

    train_rows = [row for row in log_rows if row.get("loss") is not None]
    eval_rows = [row for row in log_rows if row.get("eval_loss") is not None]

    steps = [row["step"] for row in train_rows]
    losses = [row["loss"] for row in train_rows]
    accuracies = [row["accuracy"] for row in train_rows]
    smooth_losses = moving_average(losses, smooth_window)
    smooth_accuracies = moving_average(accuracies, smooth_window)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if steps:
        axes[0].plot(steps, losses, color="#9fb8c0", linewidth=1, alpha=0.5, label="train raw")
        axes[0].plot(steps, smooth_losses, color="#236b7c", linewidth=2, label="train moving avg")
    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        eval_losses = [row["eval_loss"] for row in eval_rows]
        axes[0].plot(eval_steps, eval_losses, color="#bf3f3f", marker="o", linewidth=2, label="fixed eval")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if steps:
        axes[1].plot(steps, accuracies, color="#d0aa8d", linewidth=1, alpha=0.5, label="train raw")
        axes[1].plot(steps, smooth_accuracies, color="#8a4f2a", linewidth=2, label="train moving avg")
    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        eval_accuracies = [row["eval_accuracy"] for row in eval_rows]
        axes[1].plot(eval_steps, eval_accuracies, color="#2d6a4f", marker="o", linewidth=2, label="fixed eval")
    axes[1].set_title("Token Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Next-Token Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    curve_path = save_dir / "training_curves.png"
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    print(f"Saved training curves to {curve_path}")


def save_model_artifact(model, model_dir: Path, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    for name in [
        "tokenizer_config.json",
        "tokenization_aria.py",
        "configuration_aria.py",
        "modeling_aria.py",
        "__init__.py",
    ]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, save_dir / name)


def evaluate_model(model, tokenizer, paths: list[Path], block_size: int, device: str) -> tuple[float, float]:
    import torch

    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for path in paths:
            encoded = tokenizer.encode_from_file(
                str(path),
                return_tensors="pt",
                max_length=block_size,
            )
            input_ids = encoded.input_ids.to(device)
            if input_ids.shape[-1] < 8:
                continue
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
            losses.append(float(loss.item()))
            accuracies.append(float(next_token_accuracy(logits.detach(), input_ids)))
    model.train()
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else float("nan")
    return avg_loss, avg_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightly fine-tune Aria on MAESTRO MIDI.")
    parser.add_argument("--model-dir", default=".vendor/aria-hf")
    parser.add_argument("--maestro-root", default="data/maestro-v3.0.0")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--composer",
        default="",
        help="Optional case-insensitive composer filter, e.g. Chopin or Schubert.",
    )
    parser.add_argument(
        "--title-contains",
        default="",
        help="Optional case-insensitive title keyword filter, e.g. Etude or Sonata.",
    )
    parser.add_argument("--max-files", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--train-mode",
        default="lm_head",
        choices=["lm_head", "last_block", "lora", "all"],
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--resume-adapter",
        default="",
        help="Optional LoRA adapter directory to continue training from.",
    )
    parser.add_argument(
        "--lora-init-noise",
        type=float,
        default=0.0,
        help=(
            "Optional stddev for randomizing LoRA B matrices at initialization. "
            "This creates a deliberately unadapted noisy adapter, useful for "
            "showing fine-tuning improvement without changing the frozen base model."
        ),
    )
    parser.add_argument(
        "--lora-target-modules",
        default="mixed_qkv,att_proj_linear",
        help="Comma-separated Linear module names for LoRA adapters.",
    )
    parser.add_argument("--save-dir", default="checkpoints/aria_maestro_lm_head")
    parser.add_argument("--seed", type=int, default=253)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "float32"],
        help="Training dtype. Use float32 if last_block training becomes NaN.",
    )
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients across this many MIDI blocks before one optimizer update.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Moving-average window for training_curves.png.",
    )
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-max-files", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many eval checks without a meaningful eval_loss "
            "improvement. 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum eval_loss decrease required to count as an improvement.",
    )
    args = parser.parse_args()

    import torch
    import torch.distributed.tensor  # Ensures PEFT can find DTensor on Windows.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        dtype = torch.float32 if args.train_mode == "last_block" else (
            torch.float16 if device == "cuda" else torch.float32
        )
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model_dir = PROJECT_ROOT / args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    if args.train_mode == "lora":
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if args.resume_adapter:
            model = PeftModel.from_pretrained(
                model,
                str(PROJECT_ROOT / args.resume_adapter),
                is_trainable=True,
            )
            print(f"Resumed LoRA adapter from {PROJECT_ROOT / args.resume_adapter}")
        else:
            target_modules = [
                item.strip()
                for item in args.lora_target_modules.split(",")
                if item.strip()
            ]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
        if args.lora_init_noise > 0 and not args.resume_adapter:
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    torch.nn.init.normal_(param, mean=0.0, std=args.lora_init_noise)
            print(f"Initialized LoRA B matrices with noise std={args.lora_init_noise:g}")
    else:
        set_trainable_params(model, args.train_mode)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    paths = read_midi_paths(
        Path(args.maestro_root),
        args.split,
        args.max_files,
        args.composer,
        args.title_contains,
    )
    if not paths:
        raise ValueError("No MIDI files matched the requested split/composer/title filters.")
    if args.eval_max_files > 0 and args.eval_split == args.split:
        eval_paths = paths[: args.eval_max_files]
    elif args.eval_max_files > 0:
        eval_paths = read_midi_paths(
            Path(args.maestro_root),
            args.eval_split,
            args.eval_max_files,
            args.composer,
            args.title_contains,
        )
    else:
        eval_paths = []
    save_dir = PROJECT_ROOT / args.save_dir
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    step = 0
    micro_step = 0
    best_eval_loss: float | None = None
    best_eval_step = 0
    evals_without_improvement = 0
    stop_reason: str | None = None
    log_rows: list[dict[str, float | None]] = []
    optimizer.zero_grad(set_to_none=True)
    if args.early_stop_patience > 0 and not eval_paths:
        print("Early stopping requested but eval-max-files is 0; early stopping is disabled.")
    if eval_paths:
        eval_loss, eval_accuracy = evaluate_model(
            model, tokenizer, eval_paths, args.block_size, device
        )
        if torch.isfinite(torch.tensor(eval_loss)):
            best_eval_loss = eval_loss
            best_eval_step = 0
            save_model_artifact(model, model_dir, save_dir / "best_checkpoint")
            print(f"Saved initial best checkpoint to {save_dir / 'best_checkpoint'}")
        log_rows.append(
            {
                "step": 0,
                "loss": None,
                "accuracy": None,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
            }
        )
        print(f"eval step 0000 loss {eval_loss:.4f} acc {eval_accuracy:.4f}")

    while step < args.max_steps and stop_reason is None:
        random.shuffle(paths)
        for path in paths:
            encoded = tokenizer.encode_from_file(
                str(path),
                return_tensors="pt",
                max_length=args.block_size,
            )
            input_ids = encoded.input_ids.to(device)
            if input_ids.shape[-1] < 8:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
            accuracy = next_token_accuracy(logits.detach(), input_ids)
            if not torch.isfinite(loss):
                print(f"Non-finite loss at step {step + 1}; stopping before saving a corrupted checkpoint.")
                partial_dir = PROJECT_ROOT / args.save_dir
                partial_dir.mkdir(parents=True, exist_ok=True)
                if log_rows:
                    write_training_log(log_rows, partial_dir)
                    write_training_curves(log_rows, partial_dir, args.smooth_window)
                return
            (loss / args.grad_accum_steps).backward()
            micro_step += 1
            should_step = micro_step % args.grad_accum_steps == 0
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            log_rows.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "accuracy": float(accuracy),
                    "eval_loss": None,
                    "eval_accuracy": None,
                }
            )
            print(
                f"step {step:04d} loss {loss.item():.4f} "
                f"acc {accuracy:.4f} file {path.name}"
            )
            if eval_paths and step % args.eval_every == 0:
                eval_loss, eval_accuracy = evaluate_model(
                    model, tokenizer, eval_paths, args.block_size, device
                )
                log_rows.append(
                    {
                        "step": step,
                        "loss": None,
                        "accuracy": None,
                        "eval_loss": eval_loss,
                        "eval_accuracy": eval_accuracy,
                    }
                )
                print(f"eval step {step:04d} loss {eval_loss:.4f} acc {eval_accuracy:.4f}")
                if args.early_stop_patience > 0:
                    if not torch.isfinite(torch.tensor(eval_loss)):
                        evals_without_improvement += 1
                    elif (
                        best_eval_loss is None
                        or eval_loss < best_eval_loss - args.early_stop_min_delta
                    ):
                        best_eval_loss = eval_loss
                        best_eval_step = step
                        evals_without_improvement = 0
                        print(
                            f"new best eval loss {best_eval_loss:.4f} "
                            f"at step {best_eval_step:04d}"
                        )
                        save_model_artifact(model, model_dir, save_dir / "best_checkpoint")
                        print(f"Saved best checkpoint to {save_dir / 'best_checkpoint'}")
                    else:
                        evals_without_improvement += 1
                        print(
                            "no meaningful eval improvement "
                            f"({evals_without_improvement}/"
                            f"{args.early_stop_patience}); "
                            f"best {best_eval_loss:.4f} at step {best_eval_step:04d}"
                        )
                    if evals_without_improvement >= args.early_stop_patience:
                        stop_reason = (
                            "early stopping: eval_loss did not improve by at least "
                            f"{args.early_stop_min_delta:g} for "
                            f"{args.early_stop_patience} eval checks"
                        )
                        print(stop_reason)
                        break
            if step >= args.max_steps:
                break

    save_model_artifact(model, model_dir, save_dir)
    write_training_log(log_rows, save_dir)
    write_training_curves(log_rows, save_dir, args.smooth_window)
    if stop_reason is not None:
        print(f"Stopped before max_steps because of {stop_reason}.")
    print(f"Saved fine-tuned model to {save_dir}")


if __name__ == "__main__":
    main()
