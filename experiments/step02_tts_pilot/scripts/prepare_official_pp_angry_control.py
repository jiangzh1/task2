#!/usr/bin/env python3
"""Prepare the official PP angry record without changing its phoneme fields."""

from __future__ import annotations

import json
from pathlib import Path


STEP = Path("/data/jzh/2026/task2/experiments/step02_tts_pilot")
SOURCE = STEP / "official_reference" / "emovoice_official_test_with_phn.jsonl"
OUT = STEP / "artifacts" / "official_pp_angry_control.ready.jsonl"
REFERENCE = STEP / "official_reference" / "audio" / "neutral" / "gpt4o_23948_neutral_ash.wav"


def main() -> None:
    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw_line)
        if record.get("key") == "gpt4o_388_angry_ash":
            record["neutral_speaker_wav"] = str(REFERENCE)
            OUT.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({
                "key": record["key"],
                "source_text": record["source_text"],
                "target_text": record["target_text"],
                "has_target_text_phn": bool(record.get("target_text_phn")),
                "num_latency_tokens": 5,
            }, ensure_ascii=False, indent=2))
            return
    raise SystemExit("PP angry sample not found")


if __name__ == "__main__":
    main()
