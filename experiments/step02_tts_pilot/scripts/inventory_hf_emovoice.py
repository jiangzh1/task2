#!/usr/bin/env python3
"""List files in the official EmoVoice model repository."""

import json
import urllib.request


url = "https://huggingface.co/api/models/yhaha/EmoVoice/tree/main?recursive=true&expand=false"
with urllib.request.urlopen(url, timeout=60) as response:
    items = json.load(response)
for item in items:
    if item.get("type") == "file":
        print(f"{item.get('size', 0)}\t{item['path']}")
