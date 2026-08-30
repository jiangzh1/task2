#!/usr/bin/env python3
"""Resume-safe blind text-polarity inference for one official manifest split."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


SCHEMA = {
    "type": "object",
    "properties": {
        "polarity": {"type": "string", "enum": ["Positive", "Negative", "Neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
        "ambiguous": {"type": "boolean"},
    },
    "required": ["polarity", "confidence", "evidence", "ambiguous"],
}


def request(url: str, model: str, record: dict) -> dict:
    context = record.get("context", [])[-4:]
    prompt = (
        "Determine the literal sentiment polarity expressed by CURRENT_UTTERANCE: Positive, Negative, or Neutral. "
        "Context is supplied only to resolve references. Classify the speaker's expressed attitude in the current "
        "utterance, not merely emotion words quoted from someone else. Do not infer a sticker or use any image.\n"
        f"CONTEXT={json.dumps(context, ensure_ascii=False)}\n"
        f"CURRENT_UTTERANCE={json.dumps(record['current']['text'], ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 20260811, "num_predict": 256},
    }
    body = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(3):
        try:
            http_request = urllib.request.Request(
                url.rstrip("/") + "/api/chat", body, {"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(http_request, timeout=600) as response:
                api = json.loads(response.read().decode("utf-8"))
            message = api["message"]
            content = message.get("content") or message.get("thinking") or ""
            return {
                "parsed": json.loads(content),
                "raw_content": content,
                "response_field": "content" if message.get("content") else "thinking",
            }
        except (OSError, KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = repr(exc)
            time.sleep(2**attempt)
    raise RuntimeError(f"Ollama request failed after 3 attempts: {last_error}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["sample_id"] for row in read_jsonl(path) if row.get("status") == "ok"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    done = successful_ids(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_total = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(rows, 1):
            if record["sample_id"] in done:
                continue
            started = time.time()
            item = {"sample_id": record["sample_id"], "model": args.model, "prompt_version": "text-polarity-v2-blind", "status": "error"}
            try:
                item["text_inference"] = request(args.ollama_url, args.model, record)
                item["status"] = "ok"
            except Exception as exc:
                item["error"] = repr(exc)
            item["elapsed_seconds"] = round(time.time() - started, 3)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {record['sample_id']} {item['status']} {item['elapsed_seconds']}s", flush=True)
    print(f"completed_seconds={round(time.time() - started_total, 2)}", flush=True)


if __name__ == "__main__":
    main()
