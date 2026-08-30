#!/usr/bin/env python3
import json
import urllib.request

payload = {
    "model": "qwen3-vl:8b",
    "messages": [{"role": "user", "content": "Return JSON with one key label and value Happiness."}],
    "stream": False,
    "think": False,
    "format": "json",
    "options": {"temperature": 0, "num_predict": 64},
}
request = urllib.request.Request(
    "http://127.0.0.1:11434/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    print(response.read().decode("utf-8"))
