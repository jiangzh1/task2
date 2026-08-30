"""缓存论文小样本真实特征试运行所需的公开预训练模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import whisper
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--roberta-model", default="roberta-base")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # 仅下载与加载确认；这里不进行训练，也不占用 GPU 0 的全量 TTS。
    tokenizer = AutoTokenizer.from_pretrained(args.roberta_model, cache_dir=args.cache_dir)
    roberta = AutoModel.from_pretrained(args.roberta_model, cache_dir=args.cache_dir)
    extractor = AutoFeatureExtractor.from_pretrained(args.wavlm_model, cache_dir=args.cache_dir)
    wavlm = AutoModel.from_pretrained(args.wavlm_model, cache_dir=args.cache_dir)
    whisper_model = whisper.load_model(args.whisper_model, download_root=str(args.cache_dir / "whisper"), device="cpu")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "roberta": args.roberta_model,
                "roberta_hidden_size": roberta.config.hidden_size,
                "wavlm": args.wavlm_model,
                "wavlm_hidden_size": wavlm.config.hidden_size,
                "wavlm_hidden_layers": wavlm.config.num_hidden_layers,
                "whisper": args.whisper_model,
                "whisper_device": str(whisper_model.device),
                "torch_cuda_available": torch.cuda.is_available(),
                "purpose": "小样本真实特征冒烟测试的模型缓存；不构成正式训练结果。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.report)


if __name__ == "__main__":
    main()
