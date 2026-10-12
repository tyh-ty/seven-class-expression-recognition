"""Fixed candidate replays: visible selection diagnostics, never blind accuracy."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch
from regional_temporal import RegionalTemporal
from train_regional_temporal import Windows,classify
from temporal_residual import video_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    data=json.loads(args.validation.read_text(encoding='utf-8'))
    if data['partial_smoke_only'] or data['policy']['split']!='validation':raise ValueError('Invalid validation cache')
    windows=Windows(data['videos']);args.output.mkdir(parents=True,exist_ok=True)
    summary={}
    variants=[('released-fp32',None)]+[(p.stem,p) for p in args.checkpoint]
    if len(set(n for n,_ in variants))!=len(variants):raise ValueError('Colliding candidate names')
    for name,path in variants:
        if path is None:
            videos=deepcopy(data['videos']);metadata={'epoch':None}
        else:
            c=torch.load(path,map_location='cpu',weights_only=True)
            if c['config']['smoke']:raise ValueError('Smoke is not a trained candidate')
            if c['config']['base_policy']['graph_sha256']!=data['policy']['graph_sha256']:raise ValueError('Visual base mismatch')
            model=RegionalTemporal();model.load_state_dict(c['model_state']);model.eval()
            videos=classify(model,windows,data['videos'],128)
            metadata={'epoch':c['epoch'],'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        measured=video_metrics(videos)
        report={'model':name,'metadata':metadata,'target_sampling_fps':10,'crop_format':'png',
                'scope':'Causal regional residual; repeated subject-heldback selection validation; clip labels are not per-frame facial truth.',
                'runtime':'CPU temporal head on cached FP32 visual features; not end-to-end camera latency.',
                'visual_feature_policy':data['policy'],'metrics':measured,'videos':videos}
        dest=args.output/(name+'.json');dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
        node=shutil.which('node')
        if not node:raise ValueError('Node required for actual product smoothing replay')
        subprocess.run([node,'scripts/evaluate_video_smoothing.cjs',str(dest),str(dest.with_stem(name+'-display'))],
                       check=True,stdout=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        summary[name]={'metadata':metadata,'metrics':measured}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps({'candidate':name,'epoch':metadata['epoch'],'metrics':{k:{n:m[n] for n in ('accuracy','macro_f1_present')} for k,m in measured.items()}}),flush=True)


if __name__=='__main__':main()
