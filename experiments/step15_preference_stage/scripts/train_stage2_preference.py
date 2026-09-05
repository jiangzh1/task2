#!/usr/bin/env python3
"""阶段二正式批内偏好训练入口：冻结阶段一与 E_lat，仅训练三组双塔评分头。"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import fields
from pathlib import Path
import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for part in ("step07_method_stage1/src", "step15_preference_stage/src"):
    sys.path.insert(0, str(ROOT / part))
from spchconvsti.real_features import CachedFeatureDataset, collate_cached_features
from spchconvsti.stage1 import SpeechTextConflictReasoner
from spchconvsti.stage2 import ConflictAwareConditioner
from spchconvsti_preference import InBatchPreferenceTrainer, StageTwoPreferenceObjective, TwoTowerProjectionHead, freeze_modules
from spchconvsti_preference.latent_clip import load_official_latent_clip

def move(features, device): return type(features)(**{field.name: getattr(features, field.name).to(device) for field in fields(features)})
def ready(index, feature_dir):
    out=[]
    for line in index.open(encoding="utf-8"):
        row=json.loads(line); feature=feature_dir/f"{row['sample_id']}.pt"
        if row.get("latent_ready") and row.get("audio_ready") and feature.is_file(): row["feature_path"]=str(feature); out.append(row)
    if len(out)<2: raise RuntimeError("阶段二要求至少两条具备真实特征的样本")
    return out
def latents(rows, device):
    return torch.stack([(torch.load(row["latent_path"], map_location="cpu", weights_only=True)["latent"]) for row in rows]).to(device, dtype=torch.float16)
def load_stage1(path, reasoner, conditioner):
    saved=torch.load(path, map_location="cpu", weights_only=False)["trainable_model"]
    reasoner.load_state_dict({key.removeprefix("reasoner."): value for key,value in saved.items() if key.startswith("reasoner.")}, strict=True)
    conditioner.load_state_dict({key.removeprefix("conditioner."): value for key,value in saved.items() if key.startswith("conditioner.")}, strict=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--training-index",type=Path,required=True); p.add_argument("--feature-dir",type=Path,required=True); p.add_argument("--stage1-checkpoint",type=Path,required=True); p.add_argument("--latent-clip-source",type=Path,required=True); p.add_argument("--latent-clip-checkpoint",type=Path,required=True); p.add_argument("--vae",type=Path,required=True); p.add_argument("--scheduler-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--epochs",type=int,required=True); p.add_argument("--batch-size",type=int,required=True); p.add_argument("--learning-rate",type=float,required=True); p.add_argument("--margin",type=float,required=True); p.add_argument("--device",default="cuda"); args=p.parse_args()
    if args.batch_size<2: raise ValueError("阶段二 batch-size 必须至少为 2")
    device=torch.device(args.device); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=ready(args.training_index,args.feature_dir); by_id={r["sample_id"]:r for r in rows}; data=CachedFeatureDataset([Path(r["feature_path"]) for r in rows]); loader=DataLoader(data,batch_size=args.batch_size,shuffle=True,drop_last=True,collate_fn=collate_cached_features)
    reasoner=SpeechTextConflictReasoner(768,768,25,768).to(device); conditioner=ConflictAwareConditioner(256,256,64,codebook_size=7).to(device); load_stage1(args.stage1_checkpoint,reasoner,conditioner); freeze_modules((reasoner,conditioner))
    encoder=load_official_latent_clip(source_dir=args.latent_clip_source,checkpoint_path=args.latent_clip_checkpoint,vae_path=args.vae,device=device)
    scheduler=DDPMScheduler.from_pretrained(str(args.scheduler_dir),local_files_only=True)
    heads=[TwoTowerProjectionHead(dim,640,256).to(device) for dim in (256,64,256)]
    trainer=InBatchPreferenceTrainer(lambda batch: batch["references"],encoder,heads,StageTwoPreferenceObjective(scheduler,args.margin),frozen_stage_one=(reasoner,conditioner)).to(device)
    optimizer=torch.optim.AdamW((x for h in heads for x in h.parameters()),lr=args.learning_rate); metrics=(args.output_dir/"metrics.jsonl").open("a",encoding="utf-8"); step=0
    try:
      for epoch in range(args.epochs):
       for cached in loader:
        features=move(cached.features,device)
        with torch.no_grad():
          module1=reasoner(features); module2=conditioner(module1.h_joint)
          references={"sem":module1.completed_semantic,"emo":module2.emotion_quantized,"atm":module1.pooled_context}
        batch_rows=[by_id[x] for x in cached.sample_ids]; z=latents(batch_rows,device); t=torch.randint(0,scheduler.config.num_train_timesteps,(z.shape[0],),device=device).long()
        optimizer.zero_grad(set_to_none=True); result=trainer({"references":references},z,t)
        if not torch.isfinite(result["loss"]): raise FloatingPointError("阶段二出现非有限损失")
        result["loss"].backward(); optimizer.step(); step+=1; metrics.write(json.dumps({"epoch":epoch,"step":step,"loss":float(result["loss"].detach())})+"\n"); metrics.flush()
       payload={"schema_version":"sdxl-stage2-1","projection_heads":[h.state_dict() for h in heads],"optimizer":optimizer.state_dict(),"epoch":epoch,"global_step":step,"stage1_checkpoint":str(args.stage1_checkpoint)}; tmp=args.output_dir/"latest.tmp.pt"; torch.save(payload,tmp); os.replace(tmp,args.output_dir/"latest.pt")
    finally: metrics.close()
if __name__=="__main__": main()
