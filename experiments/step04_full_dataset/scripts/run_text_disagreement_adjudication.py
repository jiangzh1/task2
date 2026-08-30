#!/usr/bin/env python3
"""Blindly adjudicate only full-corpus Gemma3/Qwen3 text-polarity disagreements."""

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


def read_jsonl(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                result[item["sample_id"]] = item
    return result


def request(url: str, model: str, record: dict) -> dict:
    context = record.get("context", [])[-4:]
    prompt = (
        "Determine the literal sentiment polarity expressed by CURRENT_UTTERANCE: Positive, Negative, or Neutral. "
        "Use context only to resolve references. Classify the speaker's expressed attitude in the current utterance, "
        "not merely emotion words quoted from someone else. Do not infer or use any sticker, image, or prior labels.\n"
        f"CONTEXT={json.dumps(context, ensure_ascii=False)}\n"
        f"CURRENT_UTTERANCE={json.dumps(record['current']['text'], ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 20260812, "num_predict": 256},
    }
    raw = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(3):
        try:
            call = urllib.request.Request(url.rstrip("/") + "/api/chat", raw, {"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(call, timeout=600) as response:
                api = json.loads(response.read().decode("utf-8"))
            message = api["message"]
            content = message.get("content") or message.get("thinking") or ""
            return {"parsed": json.loads(content), "raw_content": content, "response_field": "content" if message.get("content") else "thinking"}
        except (OSError, KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = repr(exc)
            time.sleep(2**attempt)
    raise RuntimeError(f"Ollama request failed after 3 attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    args = parser.parse_args()

    manifest, gemma, qwen = read_jsonl(args.manifest), read_jsonl(args.gemma), read_jsonl(args.qwen)
    completed = read_jsonl(args.output) if args.output.exists() else {}
    selected = []
    for sample_id in sorted(set(manifest) & set(gemma) & set(qwen)):
        g, q = gemma[sample_id], qwen[sample_id]
        if g.get("status") != "ok" or q.get("status") != "ok":
            continue
        gp = g["text_inference"]["parsed"]["polarity"]
        qp = q["text_inference"]["parsed"]["polarity"]
        if gp != qp and sample_id not in completed:
            selected.append((sample_id, gp, qp))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for index, (sample_id, gemma_label, qwen_label) in enumerate(selected, 1):
            started = time.time()
            item = {"sample_id": sample_id, "model": args.model, "gemma_label": gemma_label, "qwen3_label": qwen_label, "prompt_version": "text-polarity-v2-blind-third-judge"}
            try:
                item["adjudication"] = request(args.ollama_url, args.model, manifest[sample_id])
                item["status"] = "ok"
            except Exception as exc:
                item["status"] = "error"
                item["error"] = repr(exc)
            item["elapsed_seconds"] = round(time.time() - started, 3)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(selected)} {sample_id} {item['status']} {item['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
