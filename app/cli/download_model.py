"""Download the default GGUF model from Hugging Face into ./models/."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ID = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
HF_FILENAME = "Qwen3-4B-Instruct-2507-Q5_K_M.gguf"
LOCAL_FILENAME = "qwen3-4b-instruct-2507-q5_k_m.gguf"
DEST_DIR = Path("models")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download GGUF model from Hugging Face")
    parser.add_argument("--repo", default=REPO_ID, help="HF repo id")
    parser.add_argument("--file", default=HF_FILENAME, help="Remote filename inside the HF repo")
    parser.add_argument("--local-name", default=LOCAL_FILENAME, help="Local filename (lowercase)")
    parser.add_argument("--dest", type=Path, default=DEST_DIR, help="Local directory")
    parser.add_argument("--sha256", default=None, help="Expected SHA-256 (optional)")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument(
        "--token", default=None,
        help="HF token (falls back to HF_TOKEN env var or ~/.cache/huggingface/token)",
    )
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("HF_TOKEN")

    dest_path = args.dest / args.local_name
    if dest_path.exists() and not args.force:
        print(f"Already exists: {dest_path}")
        if args.sha256:
            actual = sha256_file(dest_path)
            if actual != args.sha256:
                print(f"SHA-256 MISMATCH: expected {args.sha256}, got {actual}")
                sys.exit(1)
            print("SHA-256 OK")
        return

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError:
        print("Install huggingface-hub first: pip install huggingface-hub", file=sys.stderr)
        sys.exit(1)

    args.dest.mkdir(parents=True, exist_ok=True)
    hf_path = args.dest / args.file
    print(f"Downloading {args.repo}/{args.file} → {dest_path} …")
    hf_hub_download(
        repo_id=args.repo,
        filename=args.file,
        local_dir=str(args.dest),
        token=token,
    )
    if hf_path != dest_path:
        hf_path.rename(dest_path)
    print(f"Saved to {dest_path}")

    if args.sha256:
        actual = sha256_file(dest_path)
        if actual != args.sha256:
            print(f"SHA-256 MISMATCH: expected {args.sha256}, got {actual}")
            sys.exit(1)
        print("SHA-256 OK")


if __name__ == "__main__":
    main()
