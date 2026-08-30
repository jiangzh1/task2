#!/usr/bin/env bash
set -u
STEP=/data/jzh/2026/task2/experiments/step02_tts_pilot
OUT="$STEP/assets/EmoVoice/EmoVoice-PP.pt"
URL='https://huggingface.co/yhaha/EmoVoice/resolve/main/EmoVoice-PP.pt?download=true'
EXPECTED=2199065114

while true; do
  SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  if [ "$SIZE" -ge "$EXPECTED" ]; then
    echo "completed: $SIZE bytes"
    exit 0
  fi
  echo "resuming from $SIZE bytes"
  curl -L --fail --retry 5 --retry-all-errors --retry-delay 5 -C - -o "$OUT" "$URL" || true
  sleep 5
done
