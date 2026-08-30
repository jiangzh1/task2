#!/usr/bin/env bash
set -euo pipefail

STEP=/data/jzh/2026/task2/experiments/step02_tts_pilot
RAVDESS="$STEP/reference_data/RAVDESS/Audio_Speech_Actors_01-24.zip"

while pgrep -f '[d]ownload_reference_speech.sh' >/dev/null; do
  sleep 30
done

EXPECTED=bc696df654c87fed845eb13823edef8a
ACTUAL=$(md5sum "$RAVDESS" | awk '{print $1}')
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "RAVDESS MD5 mismatch: $ACTUAL" >&2
  exit 1
fi

"$STEP/.venv/bin/python" "$STEP/scripts/build_reference_library.py" \
  --ravdess-zip "$RAVDESS" \
  --crema-dir "$STEP/reference_data/CREMA-D" \
  --output-dir "$STEP/reference_library"
