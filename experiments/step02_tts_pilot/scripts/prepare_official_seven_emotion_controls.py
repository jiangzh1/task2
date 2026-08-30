#!/usr/bin/env python3
"""Prepare one official, matched-condition EmoVoice control for each emotion."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile


STEP = Path("/data/jzh/2026/task2/experiments/step02_tts_pilot")
SOURCE = STEP / "official_reference" / "emovoice_official_test.jsonl"
OUT = STEP / "artifacts" / "official_seven_emotion_controls.ready.jsonl"
REF_ZIP = STEP / "official_reference" / "neutral.zip"
REF_DIR = STEP / "official_reference" / "audio" / "neutral"
EMOTIONS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")


def main() -> None:
    chosen: dict[str, dict] = {}
    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw_line)
        emotion = record.get("emotion")
        if emotion in EMOTIONS and emotion not in chosen:
            chosen[emotion] = record
    missing = set(EMOTIONS) - set(chosen)
    if missing:
        raise RuntimeError(f"Missing official emotions: {sorted(missing)}")

    REF_DIR.mkdir(parents=True, exist_ok=True)
    prepared: list[dict] = []
    with ZipFile(REF_ZIP) as archive:
        names = archive.namelist()
        for emotion in EMOTIONS:
            record = chosen[emotion]
            filename = Path(record["neutral_speaker_wav"]).name
            members = [name for name in names if name.endswith(filename) and not name.startswith("__MACOSX/")]
            if len(members) != 1:
                raise RuntimeError(f"Reference lookup for {filename}: {members}")
            local_ref = REF_DIR / filename
            if not local_ref.exists():
                local_ref.write_bytes(archive.read(members[0]))
            record["neutral_speaker_wav"] = str(local_ref)
            record["key"] = f"official_{emotion}_control"
            prepared.append(record)

    OUT.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in prepared), encoding="utf-8")
    manifest = [
        {
            "emotion": record["emotion"],
            "key": record["key"],
            "source_text": record["source_text"],
            "emotion_text_prompt": record["emotion_text_prompt"],
            "reference_wav": record["neutral_speaker_wav"],
            "audio_token_count": len(record["answer_cosyvoice_speech_token"]),
        }
        for record in prepared
    ]
    (STEP / "artifacts" / "official_seven_emotion_controls.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
