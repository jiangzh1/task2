#!/usr/bin/env python3
"""Generate production-v1 prompt-controlled speech with checkpointed quality gates.

The input is an official JSONL manifest.  For each record this program speaks only
the final non-empty ``user`` text in ``context``: the utterance paired with the
sticker, never the preceding conversation.  It does not regenerate a sample once
one attempt has passed the text and emotion terminal gates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path

import torch
import torchaudio
import whisper
from funasr import AutoModel

ROOT = "/data/jzh/2026/task2"
COSYVOICE = "/data/jzh/2025/advio2emo/CosyVoice"
MODEL = f"{ROOT}/models/CosyVoice-300M-Instruct"
WHISPER_CACHE = f"{ROOT}/experiments/step14_real_audio_smoke/model_cache/whisper"
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
RAW_LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unk")
OFFICIAL_TO_RAW = {"Happiness": "happy", "Sadness": "sad", "Anger": "angry", "Surprise": "surprised", "Disgust": "disgusted", "Fear": "fearful", "Neutral": "neutral"}


def words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("’", "'"))


def levenshtein(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j] + (a != b)))
        row = nxt
    return row[-1]


def has_repeated_5gram(tokens: list[str]) -> bool:
    grams = [tuple(tokens[i:i + 5]) for i in range(len(tokens) - 4)]
    return len(grams) != len(set(grams))


def install_marker_fix(frontend) -> None:
    def fixed(self, tts_text, spk_id, instruct_text):
        payload = self.frontend_sft(tts_text, spk_id)
        del payload["llm_embedding"]
        token, length = self._extract_text_token(instruct_text + "<|endofprompt|>")
        payload["prompt_text"], payload["prompt_text_len"] = token, length
        return payload
    frontend.frontend_instruct = types.MethodType(fixed, frontend)


def final_user_text(record: dict) -> str | None:
    for turn in reversed(record.get("context", [])):
        if turn.get("role") == "user" and (text := turn.get("text", "").strip()):
            return text
    return None


def load_done(progress: Path) -> set[str]:
    if not progress.exists():
        return set()
    return {json.loads(line)["sample_id"] for line in progress.read_text(encoding="utf-8").splitlines()
            if line and json.loads(line).get("terminal")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("production_v1_config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / "generation.jsonl"
    done = load_done(progress)

    sys.path[:0] = [COSYVOICE, f"{COSYVOICE}/third_party/Matcha-TTS"]
    import wetext
    class IdentityNormalizer:
        def __init__(self, *args, **kwargs): pass
        def normalize(self, text): return text
    wetext.Normalizer = IdentityNormalizer
    from cosyvoice.cli.cosyvoice import CosyVoice
    tts = CosyVoice(MODEL, load_jit=False, load_trt=False, fp16=False)
    install_marker_fix(tts.frontend)
    asr = whisper.load_model("base", download_root=WHISPER_CACHE, device="cuda" if torch.cuda.is_available() else "cpu")
    emotion = AutoModel(model="iic/emotion2vec_plus_large")
    max_attempts = config["text_gate"]["maximum_attempts"]
    threshold = config["other_unk_policy"]["official_class_score_threshold"]

    for line in args.manifest.open(encoding="utf-8"):
        record = json.loads(line)
        sample_id = record["sample_id"]
        label = record["sticker"]["origin_anno"]
        text = final_user_text(record)
        if sample_id in done or label not in config["prompts"] or not text:
            continue
        terminal = None
        for attempt in range(1, max_attempts + 1):
            wav = args.output / f"{sample_id}__{label.lower()}__a{attempt}.wav"
            chunks = list(tts.inference_instruct(text, config["speaker"], config["prompts"][label], stream=False, text_frontend=False))
            audio = torch.cat([part["tts_speech"] for part in chunks], dim=1).cpu()
            audio = audio * (0.95 / audio.abs().max().clamp_min(1e-6))
            torchaudio.save(str(wav), audio, tts.sample_rate)
            wave, sr = torchaudio.load(str(wav)); wave = wave.mean(dim=0)
            if sr != 16000: wave = torchaudio.functional.resample(wave, sr, 16000)
            heard = asr.transcribe(wave.numpy(), language="en", fp16=torch.cuda.is_available(), verbose=False)["text"].strip()
            wer = levenshtein(words(text), words(heard)) / max(1, len(words(text)))
            text_ok = wer <= config["text_gate"]["maximum_word_error_rate"] and not has_repeated_5gram(words(heard))
            result = emotion.generate(str(wav), granularity="utterance", extract_embedding=False)[0]
            scores = {name: float(score) for name, score in zip(RAW_LABELS, result["scores"])}
            raw = max(scores, key=scores.get)
            formal_scores = {raw_name: scores[raw_name] for raw_name in OFFICIAL_TO_RAW.values()}
            best_formal = max(formal_scores, key=formal_scores.get)
            is_other = raw in {"other", "unk", "<unk>"}
            retry_other = is_other and formal_scores[best_formal] <= threshold
            final_label = best_formal if is_other and not retry_other else raw
            terminal = text_ok and not retry_other
            row = {"sample_id": sample_id, "target_emotion": label, "speech_text": text, "prompt": config["prompts"][label], "attempt": attempt, "audio_file": wav.name, "whisper_transcript": heard, "word_error_rate": round(wer, 4), "repeated_5gram": has_repeated_5gram(words(heard)), "text_gate_pass": text_ok, "raw_prediction": raw, "scores": scores, "prediction": final_label, "other_relabelled": is_other and not retry_other, "terminal": terminal}
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if terminal:
                break
        print(f"{sample_id}: {'accepted' if terminal else 'manual-review'}", flush=True)


if __name__ == "__main__":
    main()
