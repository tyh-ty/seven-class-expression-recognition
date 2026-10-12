"""Bounded regional residual on frozen released ViT, supervised by known photos.

Unlike the failed temporal-only residual, this can use local evidence while a
face holds its expression. One visual pass is retained. The gain bound is the
same for all classes, and no detection/confidence threshold is altered.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,TensorDataset,WeightedRandomSampler

from dataset_contract import CLASSES
from emotion_preprocessing import emotion_tensor_path
from regional_temporal import RegionalVisual
from fit_poster_frozen_head import measure


BASE=Path('training/runs/vit-small-expw-enterface-v1/weight-blends/enterface2-6-average-fp32/derived.pt')
GRAPH=Path('models/vit_emotion_seven_fp32.onnx')
RUN=Path('training/runs/vit-regional-static-v1')
WORK=Path('training/noon-20260928')


class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return torch.from_numpy(emotion_tensor_path(Path(self.rows[i]['file']),grayscale=True))


class RegionalStaticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1152,7)
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
    def forward(self,features,base_logits):return base_logits+2*torch.tanh(self.linear(features))


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(920);np.random.seed(920)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    RUN.mkdir(parents=True,exist_ok=True)
    manifest_path=Path('training/runs/poster-frozen-seven-v1/image-manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['smoke']:raise ValueError('Full training manifest required')
    if hashlib.sha256(GRAPH.read_bytes()).hexdigest()!='f9753ef6496707319576794704a0bae7674b46368305edbf366d538b2a7ce54f':
        raise ValueError('Released baseline changed')
    plan={'declared_at':datetime.now(timezone.utc).isoformat(),'seed':920,'epochs':30,
          'hypothesis':'Temporal static-invariance prevented correcting sustained poses. Test a small trained regional residual that can correct stationary as well as moving faces without a second backbone.',
          'inspiration':'Regional/local evidence from MAE-DFER/POSTER, not a reproduction of either network.',
          'source':str(manifest_path),'source_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'data':'Exactly same21000known training photos and fixed validation as preceding headtrial. New grayscale regional visual features; no validation enters training.',
          'architecture':'Frozen released ViT CLS+top/bottom patch means -> train-normalized1152 -> linear7 -> uniform2tanh residual on original logits.',
          'optimization':'AdamW3e-4,30epochs,batch256,CE smoothing.05; KL to original1.0 for correct training predictions,.25 otherwise; realeyewear2x sampler.',
          'candidates':['best_target_under_cleanFERfloor','epoch10','epoch20','epoch30'],
          'acceptance':'Unchanged sam-paired-plan photo gates; if passes actual CPU and full538video/perclass/eyes/stability/latency required.',
          'limits':'Repeated selection validation, no new independent test, no new data. Previouslyopened2610/180excluded.'}
    (WORK/'regional-static-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    checkpoint=torch.load(BASE,map_location='cpu',weights_only=True)
    backbone=timm.create_model(checkpoint['model_name'],pretrained=False,num_classes=7,**checkpoint.get('model_kwargs',{}))
    backbone.load_state_dict(checkpoint['model_state']);model=RegionalVisual(backbone).cuda().eval()
    settings=ort.SessionOptions();settings.intra_op_num_threads=2;settings.inter_op_num_threads=1
    session=ort.InferenceSession(str(GRAPH),sess_options=settings,providers=['CPUExecutionProvider'])
    datasets={'train':manifest['train'],**manifest['validation']};arrays={};base_arrays={};parity=[];start=time.perf_counter()
    for rows in datasets.values():
        for row in rows:
            if hashlib.sha256(Path(row['file']).read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Image changed after training/validation manifest was frozen')
    for name,rows in datasets.items():
        dest=RUN/(name+'-features.npz');sidecar=dest.with_suffix('.json')
        signature=hashlib.sha256(json.dumps({'rows':rows,'base':hashlib.sha256(BASE.read_bytes()).hexdigest(),
                                            'feature_contract':'grayCLS-upper7-lower7-FP32'},sort_keys=True).encode()).hexdigest()
        if dest.exists() and sidecar.exists():
            old=json.loads(sidecar.read_text(encoding='utf-8'))
            if old['signature']!=signature or old['sha256']!=hashlib.sha256(dest.read_bytes()).hexdigest():raise ValueError('Invalid cached features')
            with np.load(dest,allow_pickle=False) as d:
                arrays[name]=torch.from_numpy(d['features'].copy());base_arrays[name]=torch.from_numpy(d['logits'].copy())
            continue
        loader=DataLoader(Images(rows),batch_size=40,num_workers=2,shuffle=False,pin_memory=True)
        features=[];logits=[]
        with torch.inference_mode():
            for index,x in enumerate(loader):
                pred,f=model(x.cuda());pred=pred.cpu();f=f.cpu().reshape(len(x),-1)
                if not parity:
                    for i in range(min(4,len(x))):
                        actual=session.run(None,{session.get_inputs()[0].name:x[i:i+1].numpy()})[0]
                        error=float(np.abs(actual-pred[i:i+1].numpy()).max());parity.append(error)
                        if error>5e-4:raise ValueError('Frozen backbone disagrees with actual deployed graph')
                features.append(f);logits.append(pred)
        arrays[name]=torch.cat(features);base_arrays[name]=torch.cat(logits)
        np.savez_compressed(dest,features=arrays[name].numpy(),logits=base_arrays[name].numpy())
        sidecar.write_text(json.dumps({'signature':signature,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
        print(json.dumps({'features':name,'images':len(rows),'elapsed_seconds':round(time.perf_counter()-start,1)}),flush=True)
    del model,backbone,session;torch.cuda.empty_cache()
    mean=arrays['train'].mean(0);std=arrays['train'].std(0).clamp_min(.1)
    arrays={k:(v-mean)/std for k,v in arrays.items()}
    labels={n:torch.tensor([CLASSES.index(r['label']) for r in rows]) for n,rows in datasets.items()}
    head=RegionalStaticHead();optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30,eta_min=.00003)
    weights=torch.tensor([2. if r['real_eyewear'] else 1. for r in datasets['train']],dtype=torch.double)
    loader=DataLoader(TensorDataset(arrays['train'],base_arrays['train'],labels['train']),batch_size=256,
                      sampler=WeightedRandomSampler(weights,len(weights),replacement=True))
    config={**plan,'classes':CLASSES,'base_checkpoint':str(BASE),'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
            'base_graph_sha256':hashlib.sha256(GRAPH.read_bytes()).hexdigest(),'cpu_onnx_parity_errors':parity,'smoke':False}
    def evaluate():
        with torch.inference_mode():
            logits={n:head(arrays[n],base_arrays[n]) for n in manifest['validation']}
        return logits,{n:measure(v,labels[n]) for n,v in logits.items()}
    def save(name,epoch,scores,metrics):
        torch.save({'model_state':head.state_dict(),'mean':mean,'std':std,'epoch':epoch,'config':config,'validation':metrics},RUN/(name+'.pt'))
        report={n:{**metrics[n],'records':[{'file':r['file'],'label':r['label'],'predicted':CLASSES[int(s.argmax())],
                 'probabilities':s.softmax(0).tolist()} for r,s in zip(manifest['validation'][n],scores[n])]} for n in metrics}
        (RUN/(name+'-photos.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    scores,metrics=evaluate();best=metrics['expw_original']['macro_f1'];save('best',0,scores,metrics)
    history=[]
    for epoch in range(1,31):
        head.train();loss_sum=0.
        for features,base,y in loader:
            optimizer.zero_grad(set_to_none=True);logits=head(features,base)
            retain=torch.where(base.argmax(-1)==y,1.,.25)
            kl=F.kl_div(logits.log_softmax(-1),base.softmax(-1),reduction='none').sum(-1)
            loss=F.cross_entropy(logits,y,label_smoothing=.05)+(kl*retain).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.detach())*len(y)
        scheduler.step();head.eval();scores,metrics=evaluate()
        row={'epoch':epoch,'loss':loss_sum/len(weights),'validation':metrics};history.append(row)
        if metrics['ferplus_clean']['macro_f1']>=.792222 and metrics['expw_original']['macro_f1']>best:
            best=metrics['expw_original']['macro_f1'];save('best',epoch,scores,metrics)
        if epoch in (10,20,30):save(f'epoch-{epoch:03d}',epoch,scores,metrics)
        (RUN/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps({'epoch':epoch,'loss':row['loss'],'metrics':{n:{'accuracy':v['accuracy'],'F1':v['macro_f1']} for n,v in metrics.items()}}),flush=True)
    prior=json.loads((WORK/'paired-photos.json').read_text(encoding='utf-8'))
    key=next(k for k in prior['reports'] if Path(k).resolve()==BASE.resolve())
    reports={key:prior['reports'][key]}
    for name in ('best','epoch-010','epoch-020','epoch-030'):
        reports[str(RUN/(name+'.pt'))]=json.loads((RUN/(name+'-photos.json')).read_text(encoding='utf-8'))
    dest=WORK/'regional-static-photos.json';dest.write_text(json.dumps({'inference_precision':'float32','reports':reports},indent=2),encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','training/assess_noon_photos.py','--report',str(dest),
                    '--output',str(WORK/'regional-static-assessment.json')],check=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
