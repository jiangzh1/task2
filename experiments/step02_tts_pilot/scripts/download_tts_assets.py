#!/usr/bin/env python3
"""Download the minimum official EmoVoice inference assets with wget resume support."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
from pathlib import Path


def tree(repo: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true&expand=false"
    result = subprocess.run(
        ["curl", "-fsSL", "--retry", "10", "--retry-delay", "3", "--connect-timeout", "30", url],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def download(repo: str, path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(path, safe='/')}"
    subprocess.run(["wget", "-c", "--tries=10", "--timeout=60", "-O", str(destination), url], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root

    emovoice_files = []
    for item in tree("yhaha/EmoVoice"):
        path = item.get("path", "")
        if item.get("type") != "file":
            continue
        if path in {"EmoVoice.pt", "Qwen2.5-0.5B-phn.zip"} or path.startswith("ckpts/CosyVoice/CosyVoice-300M-SFT/"):
            emovoice_files.append((path, int(item.get("size", 0))))
    qwen_files = [
        (item["path"], int(item.get("size", 0)))
        for item in tree("Qwen/Qwen2.5-0.5B")
        if item.get("type") == "file" and item["path"] != ".gitattributes"
    ]

    manifest = {"repositories": {"yhaha/EmoVoice": emovoice_files, "Qwen/Qwen2.5-0.5B": qwen_files}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for path, _ in emovoice_files:
        download("yhaha/EmoVoice", path, root / "EmoVoice" / path)
    for path, _ in qwen_files:
        download("Qwen/Qwen2.5-0.5B", path, root / "Qwen2.5-0.5B" / path)
    print(json.dumps({"downloaded": sum(len(items) for items in manifest["repositories"].values()), "root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
