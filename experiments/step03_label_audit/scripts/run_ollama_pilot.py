#!/usr/bin/env python3
"""Run blind sticker-label and text-polarity inference through local Ollama."""

from __future__ import annotations

import argparse
import base64
import collections
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


STICKER_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_label": {
            "type": "string",
            "enum": ["Happiness", "Sadness", "Anger", "Surprise", "Disgust", "Fear", "Neutral"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "visual_evidence": {"type": "string"},
        "ambiguous": {"type": "boolean"},
    },
    "required": ["primary_label", "confidence", "visual_evidence", "ambiguous"],
}

TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "polarity": {"type": "string", "enum": ["Positive", "Negative", "Neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
        "ambiguous": {"type": "boolean"},
    },
    "required": ["polarity", "confidence", "evidence", "ambiguous"],
}


def call_ollama(url: str, payload: dict, retries: int = 3) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url.rstrip("/") + "/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
            message = result["message"]
            content = message.get("content") or message.get("thinking") or ""
            source_field = "content" if message.get("content") else "thinking"
            return {
                "parsed": json.loads(content),
                "raw_content": content,
                "response_field": source_field,
                "api": result,
            }
        except (OSError, KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = repr(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama request failed after {retries} attempts: {last_error}")


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sticker_request(model: str, image_path: Path) -> dict:
    prompt = (
        "Classify the dominant emotion visibly expressed by this sticker into exactly one of: "
        "Happiness, Sadness, Anger, Surprise, Disgust, Fear, Neutral. Judge only the actual image. "
        "Do not infer a label from filenames. Happiness includes joy, affection, amusement, and positive "
        "warmth; Neutral is used only when no dominant emotion is visible. Return concise evidence."
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [encode_image(image_path)]}],
        "format": STICKER_SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 20260810, "num_predict": 256},
    }


def text_request(model: str, record: dict) -> dict:
    context = record.get("context", [])[-4:]
    prompt = (
        "Determine the literal sentiment polarity expressed by CURRENT_UTTERANCE: Positive, Negative, "
        "or Neutral. Context is supplied only to resolve references. Classify the speaker's expressed "
        "attitude in the current utterance, not merely emotion words quoted from someone else. Do not "
        "predict a sticker and do not speculate about any image.\n"
        f"CONTEXT={json.dumps(context, ensure_ascii=False)}\n"
        f"CURRENT_UTTERANCE={json.dumps(record['current_text'], ensure_ascii=False)}"
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": TEXT_SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 20260810, "num_predict": 256},
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["sample_id"] for row in read_jsonl(path) if row.get("status") == "ok"}


def resolve_image(data_root: Path, image_value: str) -> Path:
    relative = image_value[2:] if image_value.startswith("./") else image_value
    path = data_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-class-limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.per_class_limit:
        counts: dict[str, int] = collections.defaultdict(int)
        selected = []
        for row in rows:
            label = row["sticker"]["origin_anno"]
            if counts[label] < args.per_class_limit:
                selected.append(row)
                counts[label] += 1
        rows = selected
    if args.limit:
        rows = rows[: args.limit]
    completed = load_completed(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(rows, 1):
            if record["sample_id"] in completed:
                continue
            started = time.time()
            output = {
                "sample_id": record["sample_id"],
                "model": args.model,
                "prompt_version": "sticker-v1_text-v1_no-think",
                "source_origin_anno": record["sticker"]["origin_anno"],
            }
            try:
                image_path = resolve_image(args.data_root, record["sticker"]["image_path"])
                output["sticker_inference"] = call_ollama(
                    args.ollama_url, sticker_request(args.model, image_path)
                )
                output["text_inference"] = call_ollama(
                    args.ollama_url, text_request(args.model, record)
                )
                output["status"] = "ok"
            except Exception as exc:  # preserve the failed record for audit and resume
                output["status"] = "error"
                output["error"] = repr(exc)
            output["elapsed_seconds"] = round(time.time() - started, 3)
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {record['sample_id']} {output['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
