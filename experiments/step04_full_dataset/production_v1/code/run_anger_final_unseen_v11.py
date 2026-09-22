#!/usr/bin/env python3
"""Checkpointed final unseen-text Anger A/B test: frozen v1 versus v10."""
from __future__ import annotations

import json, re, sys, types
from pathlib import Path
import torch, torchaudio, whisper

ROOT = '/data/jzh/2026/task2'
CV = '/data/jzh/2025/advio2emo/CosyVoice'
MODEL = f'{ROOT}/models/CosyVoice-300M-Instruct'
ART = Path(f'{ROOT}/experiments/step04_full_dataset/artifacts')
OUT = ART / 'anger_final_unseen_v11_20260921'
MANIFESTS = [ART / f'official_manifest_{split}.jsonl' for split in ('train','validation','test')]
CACHE = f'{ROOT}/experiments/step14_real_audio_smoke/model_cache/whisper'
PROMPTS = {'frozen_v1':'Angry: forceful, harsh, clipped, emphatic, falling.', 'revised_v10':'Angry, indignant, controlled; emphatic key words, varied pace, falling stress.'}
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

def words(text): return WORD.findall(text.lower().replace('’', "'"))
def lev(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b): nxt.append(min(nxt[-1]+1,row[j+1]+1,row[j]+(x!=y)))
        row=nxt
    return row[-1]
def repeated(items):
    grams=[tuple(items[i:i+5]) for i in range(len(items)-4)]
    return len(grams)!=len(set(grams))
def marker(frontend):
    def fixed(self, tts_text, spk_id, instruct_text):
        payload=self.frontend_sft(tts_text,spk_id); del payload['llm_embedding']
        token,length=self._extract_text_token(instruct_text+'<|endofprompt|>')
        payload['prompt_text'],payload['prompt_text_len']=token,length
        return payload
    frontend.frontend_instruct=types.MethodType(fixed,frontend)
def seen_ids():
    found=set()
    def walk(x):
        if isinstance(x,dict):
            if isinstance(x.get('sample_id'),str): found.add(x['sample_id'])
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    for p in ART.rglob('*.json'):
        try: walk(json.loads(p.read_text(encoding='utf-8')))
        except Exception: pass
    return found
def selection():
    path=OUT/'selection.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))['items']
    used=seen_ids(); rows=[]
    for p in MANIFESTS:
        for line in p.open(encoding='utf-8'):
            r=json.loads(line)
            if r['sample_id'] in used or r['sticker']['origin_anno']!='Anger': continue
            users=[t['text'].strip() for t in r['context'] if t['role']=='user' and t['text'].strip()]
            if users and users[-1].rstrip().endswith(('.', '?', '!')): rows.append({'sample_id':r['sample_id'],'split':r['split'],'speech_text':users[-1]})
    short=sorted([r for r in rows if 6<=len(r['speech_text'].split())<=60],key=lambda r:(len(r['speech_text'].split()),r['sample_id']))[:3]
    long=sorted([r for r in rows if 61<=len(r['speech_text'].split())<=120],key=lambda r:(len(r['speech_text'].split()),r['sample_id']))[:2]
    if len(short)!=3 or len(long)!=2: raise RuntimeError(f'Need 3 short and 2 long, got {len(short)}/{len(long)}')
    items=short+long
    path.write_text(json.dumps({'policy':'unseen official Anger texts; 3 complete 6-60-word short and 2 complete 61-120-word long utterances','items':items},ensure_ascii=False,indent=2),encoding='utf-8')
    return items
def main():
    OUT.mkdir(parents=True,exist_ok=True); items=selection(); progress=OUT/'items.jsonl'
    done=set()
    if progress.exists():
        for line in progress.read_text(encoding='utf-8').splitlines():
            try:
                x=json.loads(line)
                if x.get('content_gate_pass'): done.add((x['group'],x['sample_id']))
            except Exception: pass
    sys.path[:0]=[CV,f'{CV}/third_party/Matcha-TTS']; import wetext
    class N:
        def __init__(self,*a,**k): pass
        def normalize(self,x): return x
    wetext.Normalizer=N
    from cosyvoice.cli.cosyvoice import CosyVoice
    tts=CosyVoice(MODEL,load_jit=False,load_trt=False,fp16=False); marker(tts.frontend)
    asr=whisper.load_model('base',download_root=CACHE,device='cuda' if torch.cuda.is_available() else 'cpu')
    for group,prompt in PROMPTS.items():
        for i,item in enumerate(items,1):
            if (group,item['sample_id']) in done: continue
            final=None
            for attempt in range(1,4):
                wav=OUT/f'{group}__anger__{i:02d}__{item["sample_id"]}__a{attempt}.wav'
                chunks=list(tts.inference_instruct(item['speech_text'],'英文女',prompt,stream=False,text_frontend=False))
                audio=torch.cat([x['tts_speech'] for x in chunks],dim=1).cpu(); audio=audio*(0.95/audio.abs().max().clamp_min(1e-6))
                torchaudio.save(str(wav),audio,tts.sample_rate)
                wave,sr=torchaudio.load(str(wav)); wave=wave.mean(dim=0)
                if sr!=16000: wave=torchaudio.functional.resample(wave,sr,16000)
                heard=asr.transcribe(wave.numpy(),language='en',fp16=torch.cuda.is_available(),verbose=False)['text'].strip()
                wer=lev(words(item['speech_text']),words(heard))/max(1,len(words(item['speech_text']))); rep=repeated(words(heard))
                rec={**item,'index':i,'group':group,'prompt':prompt,'attempt':attempt,'audio_file':wav.name,'whisper_transcript':heard,'word_error_rate':round(wer,4),'repeated_5gram':rep,'content_gate_pass':wer<=.20 and not rep}
                with progress.open('a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                if rec['content_gate_pass']: final=rec; break
            if not final: raise RuntimeError(f'No valid audio for {group} {item["sample_id"]}')
    winners={}
    for line in progress.read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['content_gate_pass']: winners.setdefault((r['group'],r['sample_id']),r)
    ordered=[winners[(g,x['sample_id'])] for g in PROMPTS for x in items]
    (OUT/'generation_report.json').write_text(json.dumps({'prompts':PROMPTS,'items':ordered},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
