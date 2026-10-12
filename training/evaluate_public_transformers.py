"""Evaluate publisher-pinned public Transformers with their own processors, no remote code."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from train_transformer import CLASSES, metrics_from_confusion

ROOT=Path(__file__).resolve().parents[1]
ALIASES={'angry':'anger','anger':'anger','disgust':'disgust','fear':'fear','fearful':'fear',
         'happy':'happiness','happiness':'happiness','neutral':'neutral','sad':'sadness','sadness':'sadness','surprise':'surprise','surprised':'surprise'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--dataset',nargs=2,action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=24);p.add_argument('--exclude-contempt',action='store_true',help='For a documented eight-class model only, omit contempt and evaluate seven conditional scores; record original contempt probability.');args=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets={}
    for name,path in args.dataset:
        paths=[]
        for i,label in enumerate(CLASSES):
            paths.extend((file,i) for file in sorted((Path(path)/label).glob('*')) if file.suffix.lower() in {'.jpg','.png','.jpeg'})
        if not paths:raise ValueError(f'No samples: {path}')
        datasets[name]=paths
    report={'scope':'Diagnostic comparison. Public model pretraining may overlap local FER/RAVDESS/ExpW benchmarks; these scores are NOT unbiased unseen webcam accuracy. No custom model code executed.','device':str(device),'models':{}}
    for name in args.models:
        directory=ROOT/'models/research'/name
        if not (directory/'source.json').is_file():raise ValueError(f'Model download unverified: {name}')
        config=json.loads((directory/'config.json').read_text())
        mapping_file=directory/'label-map.json'
        documented_mapping=json.loads(mapping_file.read_text()) if mapping_file.exists() else None
        mapping=[];excluded=[]
        for i in range(len(config['id2label'])):
            label=config['id2label'][str(i)].lower()
            if label==f'label_{i}' and documented_mapping is not None:
                label=documented_mapping['id2label'][str(i)]
            if label=='contempt' and args.exclude_contempt:
                mapping.append(None);excluded.append(i);continue
            if label not in ALIASES:raise ValueError(f'Unverified label mapping: {name}:{label}')
            mapping.append(CLASSES.index(ALIASES[label]))
        if sorted(x for x in mapping if x is not None)!=list(range(7)) or len(excluded)>1:raise ValueError('Not exactly seven verified expression classes after explicitly excluded contempt')
        processor=AutoImageProcessor.from_pretrained(directory,local_files_only=True,trust_remote_code=False,use_fast=False)
        model=AutoModelForImageClassification.from_pretrained(directory,local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval()
        result={'source':json.loads((directory/'source.json').read_text()),'id_mapping':mapping,'label_mapping_evidence':documented_mapping,'excluded_classes':['contempt'] if excluded else [],'score_scope':'Softmax over the seven requested expressions; original contempt mass recorded separately.' if excluded else 'Original seven-class softmax.','datasets':{}}
        with torch.inference_mode():
            for dataset_name,rows in datasets.items():
                confusion=torch.zeros((7,7),dtype=torch.int64);records=[];start=perf_counter()
                for offset in range(0,len(rows),args.batch_size):
                    batch=rows[offset:offset+args.batch_size]
                    images=[]
                    for path,_ in batch:
                        with Image.open(path) as image:images.append(image.convert('RGB'))
                    inputs=processor(images=images,return_tensors='pt').to(device)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        logits=model(**inputs).logits
                    order=[mapping.index(i) for i in range(7)]
                    probabilities=torch.softmax(logits.float()[:,order],dim=-1).cpu()
                    excluded_mass=torch.softmax(logits.float(),dim=-1)[:,excluded].sum(dim=-1).cpu() if excluded else None
                    predicted=probabilities.argmax(dim=-1)
                    targets=torch.tensor([target for _,target in batch])
                    confusion+=torch.bincount(targets*7+predicted,minlength=49).reshape(7,7)
                    for j,(path,target) in enumerate(batch):
                        records.append({'file':str(path),'label':CLASSES[target],'predicted':CLASSES[int(predicted[j])],'probabilities':probabilities[j].tolist()})
                        if excluded_mass is not None:records[-1]['excluded_contempt_probability']=float(excluded_mass[j])
                metrics=metrics_from_confusion(confusion,0)
                result['datasets'][dataset_name]={'examples':len(rows),'accuracy':metrics.accuracy,'macro_f1':metrics.macro_f1,'per_class':metrics.per_class,'confusion':confusion.tolist(),'elapsed_seconds':round(perf_counter()-start,2),'records':records}
                print(json.dumps({'model':name,'dataset':dataset_name,'accuracy':metrics.accuracy,'macro_f1':metrics.macro_f1,'anger_recall':metrics.per_class['anger']['recall'],'fear_recall':metrics.per_class['fear']['recall']}),flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
        report['models'][name]=result
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
