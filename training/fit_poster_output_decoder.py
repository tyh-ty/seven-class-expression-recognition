"""Supervised constrained classifier fit; never claim an official class mapping.

Unknown pretrained logits are seven anonymous features. Fit a permutation-only
decoder with ORIGINAL TRAINING labels, freeze it before reading validation
features, and evaluate as a newly fitted model. Predictions never become labels.
"""
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

from dataset_contract import CLASSES
from fit_poster_frozen_head import measure
from poster_research import WEIGHTS,SHA

work=Path('training/noon-20260928');run=Path('training/runs/poster-frozen-seven-v1')
out=Path('training/runs/poster-supervised-decoder-v1');out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
plan={'declared_at':datetime.now(timezone.utc).isoformat(),
      'hypothesis':'A permutation-only supervised decoder preserves pretrained classifier geometry with far fewer fitted parameters than a new768to7head.',
      'protocol':'Use all21000 fixed labeled training images only. Minimize class-balanced train cross-entropy over5040permutations of anonymous logits. Freeze decoder before reading any validation feature arrays.',
      'prohibition':'Do not assign model predictions as groundtruth or alter original labels. Never describe fitted order as author-confirmed original mapping.',
      'candidates':1,'scope':'New supervised decoder trial, existing selection validation only; not independent newtest.'}
(work/'poster-supervised-decoder-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
manifest=json.loads((run/'image-manifest.json').read_text(encoding='utf-8'))
if manifest['smoke'] or manifest['feature_weights_sha256']!=SHA:raise ValueError('Wrong feature source')
if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()!=SHA:raise ValueError('Changed weights')
checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=True)['model']
w=checkpoint['module.VIT.head.linear.weight'];b=checkpoint['module.VIT.head.linear.bias']
with np.load(run/'train-features.npz',allow_pickle=False) as d:features=torch.from_numpy(d['features'].copy())
labels=torch.tensor([CLASSES.index(r['label']) for r in manifest['train']])
logits=F.linear(features,w,b);logp=logits.log_softmax(-1)
cost=torch.stack([-logp[labels==i].mean(0) for i in range(7)]).numpy()
ranked=sorted((float(sum(cost[i,p[i]] for i in range(7))),p) for p in itertools.permutations(range(7)))
mapping=list(ranked[0][1])
freeze={'created_at':datetime.now(timezone.utc).isoformat(),'classes':CLASSES,'decoder_indices':mapping,
        'feature_weights_sha256':SHA,'training_feature_sha256':hashlib.sha256((run/'train-features.npz').read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256((run/'image-manifest.json').read_bytes()).hexdigest(),
        'class_balanced_training_nll':ranked[0][0]/7,'runner_up_nll':ranked[1][0]/7,
        'train_confusion':measure(logits[:,mapping],labels),
        'official_class_mapping_confirmed':False,'fitted_with_original_training_labels':True,
        'validation_used_to_fit':False,'label_updates':0}
dest=out/'frozen-decoder.json'
if dest.exists():
    old=json.loads(dest.read_text(encoding='utf-8'))
    for key in ('decoder_indices','training_feature_sha256','manifest_sha256'):
        if old[key]!=freeze[key]:raise ValueError('Refuse to overwrite a different frozen decoder')
else:dest.write_text(json.dumps(freeze,indent=2),encoding='utf-8')
# Validation is read only after the training-fitted decoder is published.
reports={}
for name,rows in manifest['validation'].items():
    with np.load(run/(name+'-features.npz'),allow_pickle=False) as d:features=torch.from_numpy(d['features'].copy())
    scores=F.linear(features,w,b)[:,mapping]
    y=torch.tensor([CLASSES.index(r['label']) for r in rows]);m=measure(scores,y)
    m['records']=[{'file':r['file'],'label':r['label'],'predicted':CLASSES[int(s.argmax())],
                   'probabilities':s.softmax(0).tolist()} for r,s in zip(rows,scores)]
    reports[name]=m
(out/'photos.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
torch.save({'head_state':{'weight':w[mapping].clone(),'bias':b[mapping].clone()},
            'mean':torch.zeros(768),'std':torch.ones(768),'epoch':0,
            'config':{'classes':CLASSES,'smoke':False,'feature_weights_sha256':SHA,
                      'method':'Supervised permutation-constrained decoder fitted on fixed21000trainingimages',
                      'fitted_decoder':str(dest),'official_mapping_confirmed':False}},out/'decoder.pt')
paired=json.loads((work/'paired-photos.json').read_text(encoding='utf-8'))
base_plan=json.loads((work/'sam-paired-plan.json').read_text(encoding='utf-8'))
key=next(k for k in paired['reports'] if Path(k).resolve()==Path(base_plan['baseline']).resolve())
combined={'device':'Cached CUDA FP32 features / CPU FP32 fitted constrained head','inference_precision':'float32',
          'reports':{key:paired['reports'][key],str(out/'decoder.pt'):reports}}
report=work/'poster-decoder-photos.json';report.write_text(json.dumps(combined,indent=2),encoding='utf-8')
subprocess.run([sys.executable,'-X','utf8','training/assess_noon_photos.py','--report',str(report),
                '--output',str(work/'poster-decoder-assessment.json')],check=True)
print(json.dumps({'training_fitted_decoder':mapping,'official_mapping_confirmed':False,
                 'results':{n:{k:v[k] for k in ('accuracy','macro_f1')} for n,v in reports.items()}}),flush=True)
