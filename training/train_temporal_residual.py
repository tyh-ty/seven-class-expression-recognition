"""CPU-only short-sequence research, with fixed subject-disjoint validation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from temporal_residual import CLASSES, TemporalResidual, video_windows, classify_videos, video_metrics, source_graph


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-data',type=Path,required=True)
    p.add_argument('--validation-report',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--base-name', default='baseline-int8')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(20260928);np.random.seed(20260928);torch.manual_seed(20260928)
    args.output.mkdir(parents=True,exist_ok=True)
    train=json.loads(args.train_data.read_text(encoding='utf-8'))
    if train.get('partial_smoke_only') and not args.smoke:
        raise ValueError('Training cache incomplete; only --smoke may use it')
    validation=[]
    for path in args.validation_report:
        data=json.loads(path.read_text(encoding='utf-8'))
        if source_graph(data, args.base_name) != train['policy']['model_sha256']:
            raise ValueError('Validation model hash differs from training input graph')
        validation.extend(data['videos'])
    assert len({v['id'] for v in validation})==len(validation)
    assert not {v['id'] for v in validation}&{v['id'] for v in train['videos']}
    for v in train['videos']:
        if v['split']!='train':raise ValueError('Nontraining clip in training cache')
        if v['source']=='RAVDESS-train' and not 1<=v['actor']<=16:raise ValueError('Nontraining RAVactor')
    validation_fit=[v for v in validation if v['source'] in {'RAVDESS-val','eNTERFACE-val'}]
    if args.smoke:
        validation_fit=validation_fit[:2]
    clip_counts=Counter((v['source'],v['label']) for v in train['videos'])
    windows,labels,weights=[],[],[]
    for video in train['videos']:
        active=[(i,w) for i,w in video_windows(video) if video['known_label_window'][0]<=video['frames'][i]['fraction']<=video['known_label_window'][1]]
        for _,window in active:
            windows.append(window);labels.append(CLASSES.index(video['label']))
            weights.append((2 if video['eyewear'] else 1)/(max(1,len(active))*clip_counts[video['source'],video['label']]))
    x=torch.from_numpy(np.stack(windows));y=torch.tensor(labels,dtype=torch.long)
    loader=DataLoader(TensorDataset(x,y),batch_size=args.batch_size,
                      sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(x),replacement=True),num_workers=0)
    model=TemporalResidual()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=3e-5)
    baseline=video_metrics(validation_fit)
    config={'model':'TemporalResidual64/4/1','classes':CLASSES,'window':8,'seed':20260928,'api_probability_decimal_places':6,
            'base_model_sha256':train['policy']['model_sha256'],'base_name': args.base_name,'train_manifest':str(args.train_data),
            'train_manifest_sha256':hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
            'validation_reports':[str(p) for p in args.validation_report], 'train_videos':len(train['videos']),
            'train_windows':len(x),'epochs':args.epochs,'baseline':baseline,'smoke':args.smoke,
            'scope':'Clip-label supervised controlled speech, causalinput; repeatedvalidation selection, not untouchedtest or naturalmicroexpression accuracy.'}
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    score=lambda m:sum(r['macro_f1_present'] for r in m.values())/len(m)
    best=score(baseline)
    def save(path,epoch,metrics):
        torch.save({'state_dict':model.state_dict(),'epoch':epoch,'config':config,'validation':metrics},path)
    save(args.output/'best.pt',0,baseline)
    history=[]
    started=time.perf_counter()
    print(json.dumps({'baseline_score':best,'training_windows':len(x),'videos':len(train['videos'])}),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        loss_total=0.
        for inputs,target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(inputs)
            original=inputs[:,-1]
            loss=F.cross_entropy(logits,target,label_smoothing=.05)+.5*F.kl_div(F.log_softmax(logits,dim=-1),original,reduction='batchmean')
            if not torch.isfinite(loss):raise ValueError('Nonfinite temporal training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_total+=float(loss.detach())*len(inputs)
        schedule.step()
        measured=video_metrics(classify_videos(model,validation_fit,args.batch_size))
        row={'epoch':epoch,'loss':loss_total/len(x),'validation_score':score(measured),'validation':measured,
             'elapsed_seconds':round(time.perf_counter()-started,1)}
        history.append(row)
        if row['validation_score']>best:
            best=row['validation_score'];save(args.output/'best.pt',epoch,measured)
        if epoch in {10,20,30,args.epochs}:
            save(args.output/f'epoch-{epoch:03d}.pt',epoch,measured)
        (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='validation'}),flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
