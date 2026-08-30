#!/usr/bin/env bash
set -euo pipefail

STEP=/data/jzh/2026/task2/experiments/step02_tts_pilot
ENV="$STEP/.venv"
CODE="$STEP/code/EmoVoice"
RUNTIME="$STEP/runtime"
mkdir -p "$RUNTIME"

if ! "$ENV/bin/python" -m pip --version >/dev/null 2>&1; then
  curl -L --retry 5 --retry-delay 3 https://bootstrap.pypa.io/get-pip.py -o "$RUNTIME/get-pip.py"
  "$ENV/bin/python" "$RUNTIME/get-pip.py" --index-url https://pypi.org/simple
fi

"$ENV/bin/python" -m pip install --index-url https://pypi.org/simple --upgrade pip setuptools wheel
"$ENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
"$ENV/bin/python" -m pip install --index-url https://pypi.org/simple -r "$CODE/requirements.txt"
"$ENV/bin/python" -c 'import torch, torchaudio, transformers, soundfile; print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count(), "torchaudio": torchaudio.__version__, "transformers": transformers.__version__})'
