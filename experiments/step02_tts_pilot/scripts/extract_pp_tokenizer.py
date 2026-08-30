#!/usr/bin/env python3
"""Extract and validate the tokenizer released for EmoVoice-PP."""

from pathlib import Path
from zipfile import ZipFile


STEP = Path("/data/jzh/2026/task2/experiments/step02_tts_pilot")
ARCHIVE = STEP / "assets" / "EmoVoice" / "Qwen2.5-0.5B-phn.zip"
OUT = STEP / "assets" / "EmoVoice" / "Qwen2.5-0.5B-phn"
BASE = STEP / "assets" / "Qwen2.5-0.5B"


def main() -> None:
    # The release zip contains tokenizer files only.  It must be overlaid on
    # the matching Qwen base directory so AutoTokenizer can read config.json.
    if not OUT.exists() or not (OUT / "config.json").exists():
        OUT.mkdir(parents=True, exist_ok=True)
        for source in BASE.iterdir():
            destination = OUT / source.name
            if not destination.exists():
                destination.hardlink_to(source)
    with ZipFile(ARCHIVE) as archive:
        print("members:", archive.namelist()[:10])
        archive.extractall(OUT)
    nested = OUT / "Qwen2.5-0.5B-phn"
    for source in nested.iterdir():
        destination = OUT / source.name
        source.replace(destination)
    nested.rmdir()
    if not (OUT / "config.json").is_file():
        raise RuntimeError(f"Missing base config: {OUT / 'config.json'}")
    print(OUT)


if __name__ == "__main__":
    main()
