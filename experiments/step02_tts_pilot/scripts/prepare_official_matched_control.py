#!/usr/bin/env python3
"""Prepare exactly one official EmoVoice record for a matched-control inference."""

from __future__ import annotations

import json
from pathlib import Path


STEP = Path("/data/jzh/2026/task2/experiments/step02_tts_pilot")
SOURCE = STEP / "official_reference" / "emovoice_official_test.jsonl"
OUTPUT = STEP / "artifacts" / "official_angry_matched_control.ready.jsonl"
REFERENCE = STEP / "official_reference" / "audio" / "neutral" / "gpt4o_23948_neutral_ash.wav"


def main() -> None:
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("key") == "gpt4o_388_angry_ash":
            record["neutral_speaker_wav"] = str(REFERENCE)
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({
                "key": record["key"],
                "emotion": record["emotion"],
                "source_text": record["source_text"],
                "emotion_text_prompt": record["emotion_text_prompt"],
                "reference_wav": record["neutral_speaker_wav"],
                "audio_token_count": len(record["answer_cosyvoice_speech_token"]),
            }, ensure_ascii=False, indent=2))
            return
    raise SystemExit("Official angry control record not found")


if __name__ == "__main__":
    main()
