import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Aria safetensors weight file into the patched local model folder."
    )
    parser.add_argument("--repo-id", default="loubb/aria-medium-base")
    parser.add_argument(
        "--filename",
        default="model.safetensors",
        choices=["model.safetensors", "model-gen.safetensors", "model-demo.safetensors"],
    )
    parser.add_argument("--local-dir", default=".vendor/aria-hf")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(project_root / ".cache" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_root / ".cache" / "huggingface" / "transformers"))

    from huggingface_hub import hf_hub_download

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.filename} to {path}")


if __name__ == "__main__":
    main()
