#!/usr/bin/env python3
"""把已验证的 14 条真实特征与正式图片 latent 组成最终验收索引。"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--subset",type=Path,required=True); p.add_argument("--formal-train",type=Path,required=True); p.add_argument("--latent-dir",type=Path,required=True); p.add_argument("--feature-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    formal={x["sample_id"]:x for x in rows(args.formal_train)}; output=[]
    for item in rows(args.subset):
        row=formal[item["sample_id"]]; latent=args.latent_dir/f"{row['sticker']['image_sha256']}.pt"; feature=args.feature_dir/f"{item['sample_id']}.pt"
        if not latent.is_file() or not feature.is_file(): raise FileNotFoundError(item["sample_id"])
        output.append({"sample_id":item["sample_id"],"split":"train","latent_path":str(latent),"audio_path":item["audio_path"],"latent_ready":True,"audio_ready":True})
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text("".join(json.dumps(x)+"\n" for x in output),encoding="utf-8"); print(json.dumps({"samples":len(output),"status":"ready"}))
if __name__=="__main__": main()
