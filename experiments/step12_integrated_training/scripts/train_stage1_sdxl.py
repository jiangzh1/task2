#!/usr/bin/env python3
"""SpchConvSti 阶段一的正式 SDXL 训练入口。

本脚本只接受已完成的真实音频特征缓存与 SDXL VAE latent；不会提取特征、不会
生成音频。checkpoint 仅保存论文新训练模块，基础 SDXL 权重始终从本地资产重载。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import fields
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for source in ("step07_method_stage1/src", "step08_method_stage2/src", "step12_integrated_training/src"):
    sys.path.insert(0, str(ROOT / source))
from spchconvsti.real_features import CachedFeatureDataset, collate_cached_features
from spchconvsti.stage1 import SpeechTextConflictReasoner
from spchconvsti.stage2 import ConflictAwareConditioner
from spchconvsti_diffusion.sdxl_conditions import build_empty_sdxl_conditions
from spchconvsti_diffusion.sdxl_loader import load_sdxl_base
from spchconvsti_diffusion.unet_adapter import ConflictAwareUNetAdapter
from spchconvsti_integrated import SpchConvStiStageOne


def read_ready_rows(index: Path, feature_dir: Path) -> list[dict]:
    rows = []
    for line in index.open(encoding="utf-8"):
        row = json.loads(line)
        feature = feature_dir / f"{row['sample_id']}.pt"
        if row.get("latent_ready") and row.get("audio_ready") and feature.is_file():
            row["feature_path"] = str(feature)
            rows.append(row)
    if not rows:
        raise RuntimeError("没有同时具备 latent、音频和真实特征缓存的训练样本")
    return rows


def move_features(features, device: torch.device):
    return type(features)(**{item.name: getattr(features, item.name).to(device) for item in fields(features)})


def load_latents(rows: list[dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = []
    for row in rows:
        value = torch.load(row["latent_path"], map_location="cpu", weights_only=True)
        latent = value["latent"] if isinstance(value, dict) else value
        if tuple(latent.shape) != (4, 64, 64):
            raise ValueError(f"{row['sample_id']} 的 SDXL latent 形状不是 [4,64,64]")
        values.append(latent)
    return torch.stack(values).to(device=device, dtype=dtype)


def trainable_state(model: torch.nn.Module) -> dict:
    return {key: value.cpu() for key, value in model.state_dict().items() if not key.startswith("unet_adapter.unet.")}


def save_checkpoint(path: Path, model, optimizer, epoch: int, step: int, config: dict) -> None:
    payload = {"schema_version": "sdxl-stage1-1", "trainable_model": trainable_state(model), "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": step, "config": config}
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(path: Path, model, optimizer) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["trainable_model"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith("unet_adapter.unet.")]
    if unexpected:
        raise RuntimeError(f"checkpoint 含未知参数: {unexpected}")
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["epoch"]) + 1, int(payload["global_step"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-index", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="本地 SDXL Base safetensors")
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lambda-code", type=float, required=True)
    parser.add_argument("--lambda-align", type=float, required=True)
    parser.add_argument("--codebook-size", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.checkpoint_every <= 0:
        raise ValueError("epochs、batch-size、learning-rate、checkpoint-every 必须为正")
    random.seed(args.seed); torch.manual_seed(args.seed)
    device, dtype = torch.device(args.device), torch.float16
    if device.type != "cuda":
        raise ValueError("正式 SDXL 训练入口要求 CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_ready_rows(args.training_index, args.feature_dir)
    dataset = CachedFeatureDataset([Path(row["feature_path"]) for row in rows])
    by_id = {row["sample_id"]: row for row in rows}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_cached_features, drop_last=False)
    pipeline = load_sdxl_base(checkpoint_path=args.checkpoint, vae_path=args.vae, torch_dtype=dtype).to(device)
    pipeline.set_progress_bar_config(disable=True)
    for module in (pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2, pipeline.unet):
        module.eval().requires_grad_(False)
    reasoner = SpeechTextConflictReasoner(768, 768, 25, 768).to(device)
    conditioner = ConflictAwareConditioner(256, 256, 64, codebook_size=args.codebook_size).to(device)
    adapter = ConflictAwareUNetAdapter(pipeline.unet, 256, 64, 128, num_content_tokens=4, layer_paths=("mid_block",)).to(device)
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, prediction_type="epsilon")
    model = SpchConvStiStageOne(reasoner, conditioner, adapter, scheduler, args.lambda_code, args.lambda_align).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    start_epoch, global_step = (0, 0) if args.resume is None else restore_checkpoint(args.resume, model, optimizer)
    config = vars(args) | {"ready_samples": len(rows), "sdxl_base_frozen": True}
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    history = (args.output_dir / "train_metrics.jsonl").open("a", encoding="utf-8")
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train(); pipeline.unet.eval()
            for cached_batch in loader:
                ids = list(cached_batch.sample_ids)
                batch_rows = [by_id[sample_id] for sample_id in ids]
                features = move_features(cached_batch.features, device)
                latents = load_latents(batch_rows, device, dtype)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (len(ids),), device=device).long()
                conditions = build_empty_sdxl_conditions(pipeline, len(ids), device, dtype)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output = model(features, latents, timesteps, base_encoder_hidden_states=conditions.encoder_hidden_states, unet_kwargs={"added_cond_kwargs": conditions.added_cond_kwargs})
                if not torch.isfinite(output.total_loss):
                    raise FloatingPointError(f"step {global_step} 出现非有限损失")
                output.total_loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0); optimizer.step()
                global_step += 1
                history.write(json.dumps({"epoch": epoch, "step": global_step, "loss": float(output.total_loss.detach()), "content": float(output.content_loss.detach()), "codebook": float(output.codebook_loss.detach()), "alignment": float(output.alignment_loss.detach())}) + "\n"); history.flush()
                if global_step % args.checkpoint_every == 0:
                    save_checkpoint(args.output_dir / "latest.pt", model, optimizer, epoch, global_step, config)
            save_checkpoint(args.output_dir / "latest.pt", model, optimizer, epoch, global_step, config)
    finally:
        history.close(); adapter.remove_hooks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
