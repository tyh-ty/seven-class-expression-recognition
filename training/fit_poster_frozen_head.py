"""Fit a new known-label classifier on frozen POSTER regional/landmark features.

No meaning is inferred for the checkpoint's anonymous output logits: its entire
old classifier is removed. Only original training labels supervise a new head.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,TensorDataset,WeightedRandomSampler

from dataset_contract import CLASSES
from emotion_preprocessing import emotion_tensor_path
from poster_research import build_features,SHA
from train_transformer import metrics_from_confusion


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return torch.from_numpy(emotion_tensor_path(Path(self.rows[i]['file']),grayscale=False))


def read_rows(root):
    result=[]
    for label in CLASSES:
        for path in sorted((root/label).glob('*')):
            if path.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                result.append({'file':str(path),'label':label,'sha256':sha(path)})
    if not result:raise ValueError('Empty image dataset')
    return result


def measure(logits,labels):
    matrix=torch.bincount(labels*7+logits.argmax(-1),minlength=49).reshape(7,7)
    m=metrics_from_confusion(matrix,0.)
    return {'examples':len(labels),'accuracy':m.accuracy,'macro_f1':m.macro_f1,
            'per_class':m.per_class,'confusion':matrix.tolist()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data/seven-class/expw-enterface-v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',nargs=2,action='append',required=True)
    p.add_argument('--per-class',type=int,default=3000)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(919);random.seed(919);np.random.seed(919)
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((args.data/'dataset.json').read_text(encoding='utf-8'))
    eyes={Path(v['path']).name for v in metadata['images'] if v.get('split')=='train' and v.get('real_eyewear')}
    training=read_rows(args.data/'train')
    selected=[]
    for label in CLASSES:
        subset=[v for v in training if v['label']==label]
        subset.sort(key=lambda v:(Path(v['file']).name not in eyes,hashlib.sha256(('919|'+v['file']).encode()).hexdigest()))
        selected.extend(subset[:args.per_class])
    for v in selected:v['real_eyewear']=Path(v['file']).name in eyes
    datasets={name:read_rows(Path(path)) for name,path in args.dataset}
    if set(v['sha256'] for v in training)&{v['sha256'] for rows in datasets.values() for v in rows}:
        raise ValueError('Train/validation exact byte leakage')
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'classes':CLASSES,'train':selected,
              'validation':datasets,'feature_weights_sha256':SHA,'smoke':args.smoke,
              'policy':'Known labels only; SHA-selected capped class-balanced training with all real-eyewear prioritized. No predictions used to select examples.',
              'preprocessing':'Product RGB OpenCV area/down, linear/up 224; ImageNet mean/std. Intentional product contract, not claim of exact author benchmark reproduction.'}
    (args.output/'image-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    model=build_features().to(args.device).eval()
    arrays={};start=time.perf_counter();parity=[]
    for name,rows in {'train':selected,**datasets}.items():
        dest=args.output/(name+'-features.npz')
        signature=hashlib.sha256(json.dumps({'rows':rows,'weights':SHA,'preprocess':manifest['preprocessing']},sort_keys=True).encode()).hexdigest()
        sigfile=dest.with_suffix('.json')
        if dest.exists() and sigfile.exists():
            old=json.loads(sigfile.read_text(encoding='utf-8'))
            if old['signature']!=signature or old['sha256']!=sha(dest):raise ValueError('Stale feature cache')
            with np.load(dest,allow_pickle=False) as cache:arrays[name]=torch.from_numpy(cache['features'].copy())
            continue
        loader=DataLoader(Images(rows),batch_size=args.batch_size,num_workers=args.workers,shuffle=False,
                          pin_memory=args.device=='cuda',persistent_workers=args.workers>0)
        output=[]
        with torch.inference_mode():
            for bi,images in enumerate(loader):
                features=model(images.to(args.device)).cpu()
                if name=='train' and not parity:
                    if args.device=='cuda':
                        model.cpu();reference=model(images[:2]).cpu();model.to(args.device)
                        difference=float((reference-features[:2]).abs().max())
                        if difference>1e-3:raise ValueError('GPU/CPU feature mismatch')
                        parity.append(difference)
                    else:parity.append(0.)
                if features.shape[1]!=768 or not torch.isfinite(features).all():raise ValueError('Bad feature tensor')
                output.append(features)
                if bi%100==0:print(json.dumps({'dataset':name,'batch':bi,'elapsed_seconds':round(time.perf_counter()-start,1)}),flush=True)
        arrays[name]=torch.cat(output)
        np.savez_compressed(dest,features=arrays[name].numpy())
        sigfile.write_text(json.dumps({'signature':signature,'sha256':sha(dest),'shape':list(arrays[name].shape)},indent=2),encoding='utf-8')
    del model
    if args.device=='cuda':torch.cuda.empty_cache()
    # Fit on CPU so no second GPU job is needed. Normalization is fitted only on training.
    mean=arrays['train'].mean(0);std=arrays['train'].std(0).clamp_min(.1)
    arrays={k:(v-mean)/std for k,v in arrays.items()}
    labels={name:torch.tensor([CLASSES.index(r['label']) for r in rows]) for name,rows in {'train':selected,**datasets}.items()}
    head=nn.Linear(768,7);optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.0003)
    weights=torch.tensor([2. if v['real_eyewear'] else 1. for v in selected],dtype=torch.double)
    loader=DataLoader(TensorDataset(arrays['train'],labels['train']),batch_size=256,
                      sampler=WeightedRandomSampler(weights,len(selected),replacement=True))
    config={k:v for k,v in manifest.items() if k not in ('train','validation')}
    config.update(train_examples=len(selected),train_real_eyewear=sum(v['real_eyewear'] for v in selected),
                  epochs=args.epochs,gpu_cpu_feature_max_error=parity,seed=919,
                  method='Frozen official-code POSTER features, newly trained seven-class linear head; not original checkpoint labels/benchmark.')
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    best=-1.;history=[]
    for epoch in range(1,args.epochs+1):
        head.train();loss_sum=0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True);logits=head(x)
            loss=F.cross_entropy(logits,y,label_smoothing=.05)
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.detach())*len(y)
        scheduler.step();head.eval()
        with torch.inference_mode():
            scores={name:head(arrays[name]) for name in datasets}
            metrics={name:measure(scores[name],labels[name]) for name in datasets}
        row={'epoch':epoch,'loss':loss_sum/len(selected),'validation':metrics}
        history.append(row)
        target=metrics.get('expw_original',next(iter(metrics.values())))['macro_f1']
        save_names=[]
        if target>best:best=target;save_names.append('best.pt')
        if epoch in (10,20,30,args.epochs):save_names.append(f'epoch-{epoch:03d}.pt')
        for name in save_names:
            torch.save({'head_state':head.state_dict(),'mean':mean,'std':std,'epoch':epoch,
                        'config':config,'validation':metrics},args.output/name)
            report={n:{**metrics[n],'records':[{'file':r['file'],'label':r['label'],
                    'predicted':CLASSES[int(score.argmax())],'probabilities':score.softmax(0).tolist()}
                    for r,score in zip(datasets[n],scores[n])]} for n in datasets}
            (args.output/(Path(name).stem+'-photos.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps({'epoch':epoch,'loss':row['loss'],'metrics':{n:{'accuracy':v['accuracy'],'F1':v['macro_f1']} for n,v in metrics.items()}}),flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
