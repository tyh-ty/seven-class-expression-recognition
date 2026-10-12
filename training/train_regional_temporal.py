"""Train only a small causal head on frozen regional ViT features."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

from dataset_contract import CLASSES
from regional_temporal import RegionalTemporal,causal_indices,load_features
from temporal_residual import video_metrics


class Windows(Dataset):
    def __init__(self,videos,supervised=False):
        self.arrays=[load_features(v) for v in videos]
        self.items=[];self.weights=[]
        counts=Counter((v['source'],v['label']) for v in videos)
        for vi,v in enumerate(videos):
            windows=causal_indices(v)
            if supervised:
                windows=[(fi,idx) for fi,idx in windows if v['known_label_window'][0]<=v['frames'][fi]['fraction']<=v['known_label_window'][1]]
            for fi,idx in windows:
                self.items.append((vi,fi,idx,CLASSES.index(v['label'])))
                self.weights.append((2 if v.get('eyewear') else 1)/(max(1,len(windows))*counts[v['source'],v['label']]))

    def __len__(self):return len(self.items)

    def __getitem__(self,index):
        vi,fi,indices,label=self.items[index]
        features,logits=self.arrays[vi]
        return torch.from_numpy(features[indices]),torch.from_numpy(logits[indices[-1]]),label,index


def classify(model,dataset,videos,batch_size):
    transformed=deepcopy(videos);model.eval()
    with torch.inference_mode():
        for x,base,_,indices in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            probabilities=model(x,base).softmax(-1).numpy()
            for idx,p in zip(indices.tolist(),probabilities):
                vi,fi,_,_=dataset.items[idx]
                transformed[vi]['frames'][fi]['probabilities']=p.tolist()
    return transformed


def check_splits(train,val):
    if train['policy']['split']!='train' or val['policy']['split']!='validation':raise ValueError('Wrong data split')
    for key in ('checkpoint_sha256','graph_sha256','capture_logic_ast_sha256'):
        if train['policy'][key]!=val['policy'][key]:raise ValueError('Feature contract mismatch')
    if set(v['id'] for v in train['videos'])&set(v['id'] for v in val['videos']):raise ValueError('Video overlap')
    def subjects(data,source,key):return {v[key] for v in data['videos'] if v['source']==source}
    if subjects(train,'RAVDESS-train','actor') & subjects(val,'RAVDESS-val','actor'):raise ValueError('RAV actor leak')
    if subjects(train,'eNTERFACE-train','subject') & subjects(val,'eNTERFACE-val','subject'):raise ValueError('eNTER actor leak')
    for v in train['videos']:
        if v.get('split')!='train':raise ValueError('Nontraining clip')
    for v in train['videos']+val['videos']:
        if hashlib.sha256(Path(v['feature_file']).read_bytes()).hexdigest()!=v['feature_sha256']:
            raise ValueError('Changed feature bytes')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    random.seed(918);np.random.seed(918);torch.manual_seed(918)
    train=json.loads(args.train.read_text(encoding='utf-8'))
    val=json.loads(args.validation.read_text(encoding='utf-8'))
    if (train['partial_smoke_only'] or val['partial_smoke_only']) and not args.smoke:
        raise ValueError('Partial cache must not be used for actual training')
    check_splits(train,val)
    validation=[v for v in val['videos'] if v['source'] in ('RAVDESS-val','eNTERFACE-val')]
    training=Windows(train['videos'],supervised=True);evaluation=Windows(validation)
    model=RegionalTemporal();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=3e-5)
    loader=DataLoader(training,batch_size=args.batch_size,num_workers=0,
                      sampler=WeightedRandomSampler(torch.tensor(training.weights,dtype=torch.double),len(training),replacement=True))
    baseline=video_metrics(deepcopy(validation))
    score=lambda m:sum(v['macro_f1_present'] for v in m.values())/len(m)
    config={'architecture':'frozen ViT CLS/upper/lower, regional-temporal Transformer64/4/1, eight past/current frames',
            'classes':CLASSES,'train':str(args.train),'validation':str(args.validation),'train_videos':len(train['videos']),
            'validation_videos':len(validation),'windows':len(training),'base_policy':train['policy'],'smoke':args.smoke,
            'epochs':args.epochs,'seed':918,'baseline':baseline,'selection':'Repeated selection validation, not new blind test',
            'paper_inspiration':'https://arxiv.org/abs/2307.02227; regional/motion principles only, no MAE pretraining reproduced'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    def save(name,epoch,metrics):
        torch.save({'model_state':model.state_dict(),'epoch':epoch,'config':config,'metrics':metrics},args.output/name)
    best=score(baseline);save('best.pt',0,baseline);history=[];start=time.perf_counter()
    print(json.dumps({'initial_score':best,'windows':len(training),'train_videos':len(train['videos'])}),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0.
        for features,base,labels,_ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(features,base)
            loss=F.cross_entropy(logits,labels,label_smoothing=.05)+.5*F.kl_div(
                F.log_softmax(logits,dim=-1),F.softmax(base,dim=-1),reduction='batchmean')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite temporal loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.detach())*len(labels)
        scheduler.step()
        measured=video_metrics(classify(model,evaluation,validation,args.batch_size))
        row={'epoch':epoch,'loss':loss_sum/len(training),'validation_score':score(measured),
             'metrics':measured,'elapsed_seconds':round(time.perf_counter()-start,1)}
        history.append(row)
        if row['validation_score']>best:best=row['validation_score'];save('best.pt',epoch,measured)
        if epoch in (5,10,20,args.epochs):save(f'epoch-{epoch:03d}.pt',epoch,measured)
        (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
