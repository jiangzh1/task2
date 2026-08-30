#!/usr/bin/env python3
"""Extract only the reference wav required by the official matched control."""

from pathlib import Path
from zipfile import ZipFile


STEP = Path("/data/jzh/2026/task2/experiments/step02_tts_pilot")
FILENAME = "gpt4o_23948_neutral_ash.wav"
OUTPUT = STEP / "official_reference" / "audio" / "neutral" / "gpt4o_23948_neutral_ash.wav"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(STEP / "official_reference" / "neutral.zip") as archive:
        matches = [
            name for name in archive.namelist()
            if name.endswith(FILENAME) and not name.startswith("__MACOSX/")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one archive member for {FILENAME}, got: {matches}")
        OUTPUT.write_bytes(archive.read(matches[0]))
    print(f"extracted: {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
