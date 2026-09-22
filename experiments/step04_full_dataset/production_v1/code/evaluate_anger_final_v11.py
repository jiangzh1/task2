#!/usr/bin/env python3
"""Checkpointed emotion2vec evaluation for final unseen Anger A/B test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from funasr import AutoModel

OUT=Path('/data/jzh/2026/task2/experiments/step04_full_dataset/artifacts/anger_final_unseen_v11_20260921')
MAP={'angry':'Anger','disgusted':'Disgust','fearful':'Fear','happy':'Happiness','neutral':'Neutral','sad':'Sadness','surprised':'Surprise'}
ORDER=['Happiness','Sadness','Anger','Surprise','Disgust','Fear','Neutral','Other']
def main():
    rows=json.loads((OUT/'generation_report.json').read_text(encoding='utf-8'))['items']; progress=OUT/'emotion_items.jsonl'; done={}
    if progress.exists():
        for line in progress.read_text(encoding='utf-8').splitlines():
            x=json.loads(line); done[(x['group'],x['sample_id'])]=x
    model=AutoModel(model='iic/emotion2vec_plus_large',disable_update=True)
    for row in rows:
        key=(row['group'],row['sample_id'])
        if key in done: continue
        result=model.generate(input=str(OUT/row['audio_file']),granularity='utterance',extract_embedding=False)[0]
        labels=result['labels']; scores=result['scores']; idx=int(np.argmax(scores)); raw=str(labels[idx]).lower().split('/')[-1]
        x={**row,'recognized_as':MAP.get(raw,'Other'),'raw_label':str(labels[idx]),'scores':[float(v) for v in scores]}
        with progress.open('a',encoding='utf-8') as f: f.write(json.dumps(x,ensure_ascii=False)+'\n')
        done[key]=x; print(json.dumps(x,ensure_ascii=False),flush=True)
    groups={}
    for group in ('frozen_v1','revised_v10'):
        xs=[done[(group,r['sample_id'])] for r in rows if r['group']==group]
        matrix={x:{y:0 for y in ORDER} for x in ['Anger']}
        for x in xs: matrix['Anger'][x['recognized_as']]+=1
        groups[group]={'n':len(xs),'correct':matrix['Anger']['Anger'],'accuracy':round(matrix['Anger']['Anger']/len(xs),4),'confusion_matrix':matrix,'items':xs}
    (OUT/'emotion_report.json').write_text(json.dumps({'labels_order':ORDER,**groups},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({g:{k:v for k,v in x.items() if k!='items'} for g,x in groups.items()},ensure_ascii=False))
if __name__=='__main__': main()
