#!/usr/bin/env python3
"""Exploratory real-text screen: 5 official utterances per emotion, current CosyVoice zero-shot only."""
from __future__ import annotations
import argparse, json, sys, zipfile
from collections import defaultdict
from pathlib import Path
import torch, torchaudio
from funasr import AutoModel
ROOT = '/data/jzh/2026/task2'
CV = '/data/jzh/2025/advio2emo/CosyVoice'
MODEL = f'{ROOT}/models/CosyVoice-300M-Instruct'

def install_marker_fix(frontend):
    import types
    def fixed(self, tts_text, spk_id, instruct_text):
        payload = self.frontend_sft(tts_text, spk_id)
        del payload['llm_embedding']
        token, length = self._extract_text_token(instruct_text + '<|endofprompt|>')
        payload['prompt_text'], payload['prompt_text_len'] = token, length
        return payload
    frontend.frontend_instruct = types.MethodType(fixed, frontend)

PRODUCTION = Path(f"{ROOT}/experiments/step04_full_dataset/production_v1")
OUT = PRODUCTION / "reference_audio" / "zero_shot_v3"
ZIP = Path(f"{ROOT}/experiments/step02_tts_pilot/reference_data/RAVDESS/Audio_Speech_Actors_01-24.zip")
CONTROL_MANIFEST = PRODUCTION / "prompt_audio" / "frozen_v1" / "manifest.json"
PROMPT_TEXT = "Kids are talking by the door."
EMOTIONS = {"Happiness": "03", "Sadness": "04", "Anger": "05", "Fear": "06", "Disgust": "07", "Surprise": "08", "Neutral": "01"}
RAW = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unk")
ACTORS = (1, 2, 3, 1, 2)

def member_for(z, emotion_code, actor):
    base = f"03-01-{emotion_code}-01-01-01-{actor:02d}.wav"
    return next(x for x in z.namelist() if x.endswith(f"Actor_{actor:02d}/" + base))

def select_items():
    buckets = defaultdict(list)
    for row in json.loads(CONTROL_MANIFEST.read_text(encoding="utf-8"))["items"]:
        buckets[row["sticker_emotion_official"]].append({"sample_id": row["sample_id"], "split": "frozen_prompt_control", "text": row["speech_text"]})
    selected = {}
    for label in EMOTIONS:
        options = buckets[label]
        if len(options) < 5:
            raise RuntimeError(f"Need 5 valid user utterances for {label}, found {len(options)}")
        selected[label] = options[:5]
    return selected

def main():
    global OUT
    parser = argparse.ArgumentParser(description="RAVDESS zero-shot reference screen (latest retained reference route).")
    parser.add_argument("--output", type=Path, default=OUT, help="output directory; defaults to the retained zero-shot-v3 archive")
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    selected = select_items()
    (OUT / "selection.json").write_text(json.dumps({"selection_policy": "exact five items per class from frozen_cosyvoice_control_bank_v1_20260919/manifest.json; no prompt-control audio regenerated", "items": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.path[:0] = [CV, f"{CV}/third_party/Matcha-TTS"]
    import wetext
    class N:
        def __init__(self, *args, **kwargs): pass
        def normalize(self, text): return text
    wetext.Normalizer = N
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
    tts = CosyVoice(MODEL, load_jit=False, load_trt=False, fp16=False)
    install_marker_fix(tts.frontend)
    ser = AutoModel(model="iic/emotion2vec_plus_large")
    rows = []
    with zipfile.ZipFile(ZIP) as z:
        for label, code in EMOTIONS.items():
            for index, item in enumerate(selected[label]):
                actor = ACTORS[index]
                ref = OUT / f"ref__{label.lower()}__actor_{actor:02d}.wav"
                ref.write_bytes(z.read(member_for(z, code, actor)))
                chunks = list(tts.inference_zero_shot(item["text"], PROMPT_TEXT, load_wav(str(ref), 16000), stream=False, text_frontend=False))
                speech = torch.cat([chunk["tts_speech"] for chunk in chunks], dim=1).cpu()
                raw_wav = OUT / f"generated_raw__{label.lower()}__{index + 1:02d}.wav"
                torchaudio.save(str(raw_wav), speech, tts.sample_rate)
                speech = speech * (0.95 / speech.abs().max().clamp_min(1e-6))
                wav = OUT / f"generated__{label.lower()}__{index + 1:02d}.wav"
                torchaudio.save(str(wav), speech, tts.sample_rate)
                result = ser.generate(str(wav), granularity="utterance", extract_embedding=False)[0]
                scores = {key: float(value) for key, value in zip(RAW, result["scores"])}
                row = {"target_emotion": label, "reference_actor": actor, "reference_audio": ref.name, "sample_id": item["sample_id"], "split": item["split"], "speech_text": item["text"], "raw_audio_file": raw_wav.name, "audio_file": wav.name, "normalization": "peak normalized to 0.95; raw model waveform retained", "prediction": max(scores, key=scores.get), "scores": scores}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    (OUT / "report.json").write_text(json.dumps({"purpose": "exploratory 7-emotion, exact five frozen prompt-control texts per emotion, reference-style zero-shot screen; current CosyVoice-300M-Instruct only", "prompt_text": PROMPT_TEXT, "items": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
