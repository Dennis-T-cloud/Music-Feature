import argparse
import os
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "transformers"))
aria_utils_path = PROJECT_ROOT / ".vendor" / "aria-utils-src"
if aria_utils_path.exists():
    sys.path.insert(0, str(aria_utils_path))

pydeps_path = PROJECT_ROOT / ".vendor" / "pydeps"
if pydeps_path.exists():
    sys.path.append(str(pydeps_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate symbolic MIDI with the pretrained Aria model."
    )
    parser.add_argument(
        "--model-id",
        default=".vendor/aria-hf",
        help="Local Aria model/code directory. Run scripts/aria_download_weights.py first.",
    )
    parser.add_argument("--prompt", default=None, help="Optional MIDI prompt path.")
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional PEFT/LoRA adapter directory to load on top of the base model.",
    )
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.97)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=253)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

    model_ref = str((PROJECT_ROOT / args.model_id).resolve()) if not args.model_id.startswith("loubb/") else args.model_id

    print(f"Loading tokenizer: {model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    print(f"Loading model: {model_ref} ({dtype}, {device})")
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if args.adapter_dir:
        from peft import PeftModel

        adapter_ref = str((PROJECT_ROOT / args.adapter_dir).resolve())
        print(f"Loading LoRA adapter: {adapter_ref}")
        model = PeftModel.from_pretrained(model, adapter_ref)
    model.to(device)
    model.eval()

    if args.prompt:
        prompt_path = Path(args.prompt)
        print(f"Encoding prompt MIDI: {prompt_path}")
        prompt = tokenizer.encode_from_file(str(prompt_path), return_tensors="pt")
        input_ids = prompt.input_ids[..., : args.prompt_tokens]
        print(f"Prompt tokens used: {input_ids.shape[-1]}")
    else:
        piano_prefix_id = tokenizer._convert_token_to_id(("prefix", "instrument", "piano"))
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        input_ids = torch.tensor([[piano_prefix_id, bos_id]], dtype=torch.long)
        print("No MIDI prompt supplied; starting from Aria piano prefix + BOS for Task 1.")

    input_ids = input_ids.to(device)
    if args.max_length <= input_ids.shape[-1]:
        raise ValueError("--max-length must be larger than the number of prompt tokens")

    with torch.no_grad():
        continuation = model.generate(
            input_ids,
            max_length=args.max_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi_dict = tokenizer.decode(continuation[0].detach().cpu().tolist())
    midi_dict.to_midi().save(out_path)
    print(f"Saved generated MIDI: {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
