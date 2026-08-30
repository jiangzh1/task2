#!/usr/bin/env python3
"""Use a third blind text-only model to adjudicate prior text-polarity disagreements."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


SCHEMA = {
    "type": "object",
    "properties": {
        "polarity": {"type": "string", "enum": ["Positive", "Negative", "Neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": ["polarity", "confidence", "evidence"],
}


def jsonl(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row
    return result


def request(url: str, model: str, record: dict) -> dict:
    context = record.get("context", [])[-4:]
    prompt = (
        "Determine the literal sentiment polarity expressed by CURRENT_UTTERANCE: Positive, Negative, or Neutral. "
        "Use the context only to resolve references. Classify the speaker's own attitude in the current utterance, "
        "not an emotion merely quoted or attributed to another speaker. Do not infer any image or sticker.\n"
        f"CONTEXT={json.dumps(context, ensure_ascii=False)}\n"
        f"CURRENT_UTTERANCE={json.dumps(record['current_text'], ensure_ascii=False)}"
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
            req = urllib.request.Request(url.rstrip("/") + "/api/chat", body, {"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=600) as response:
                api = json.loads(response.read().decode("utf-8"))
            message = api["message"]
            content = message.get("content") or message.get("thinking") or ""
            return {"parsed": json.loads(content), "response_field": "content" if message.get("content") else "thinking", "raw_content": content}
        except Exception as exc:  # retain a clear audit record rather than silently dropping a sample
            last_error = repr(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(last_error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--qwen-vl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    args = parser.parse_args()

    pilot, gemma, qwen_vl = jsonl(args.pilot), jsonl(args.gemma), jsonl(args.qwen_vl)
    completed = jsonl(args.output) if args.output.exists() else {}
    selected = []
    for sample_id in sorted(set(pilot) & set(gemma) & set(qwen_vl)):
        g, q = gemma[sample_id], qwen_vl[sample_id]
        if g.get("status") != "ok" or q.get("status") != "ok":
            continue
        gp = g["text_inference"]["parsed"]["polarity"]
        qp = q["text_inference"]["parsed"]["polarity"]
        if gp != qp and sample_id not in completed:
            selected.append((sample_id, gp, qp))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for index, (sample_id, gemma_label, qwen_vl_label) in enumerate(selected, 1):
            started = time.time()
            item = {"sample_id": sample_id, "gemma_label": gemma_label, "qwen_vl_label": qwen_vl_label, "model": args.model}
            try:
                item["adjudication"] = request(args.ollama_url, args.model, pilot[sample_id])
                item["status"] = "ok"
            except Exception as exc:
                item["status"] = "error"
                item["error"] = repr(exc)
            item["elapsed_seconds"] = round(time.time() - started, 3)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(selected)} {sample_id} {item['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
