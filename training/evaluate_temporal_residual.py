"""Replay fixed actualINT8 video predictions through research temporal checkpoints."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch

from temporal_residual import TemporalResidual, classify_videos, video_metrics, source_graph

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,nargs='+',required=True)
    p.add_argument('--validation-report',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--base-name', default='baseline-int8')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    videos=[]
    base_hash=None
    for path in args.validation_report:
        data=json.loads(path.read_text(encoding='utf-8'))
        graph=source_graph(data, args.base_name)
        if base_hash is not None and graph!=base_hash:
            raise ValueError('Different base graphs across video sets')
        base_hash=graph
        videos.extend(data['videos'])
    if len({v['id'] for v in videos})!=len(videos):raise ValueError('Duplicate validation video IDs')
    args.output.mkdir(parents=True,exist_ok=True)
    summary={}
    variants=[(args.base_name,None),*[(p.stem,p) for p in args.checkpoint]]
    if len({n for n,_ in variants})!=len(variants):raise ValueError('Checkpoint output names collide')
    for name,path in variants:
        if path is None:
            transformed=copy.deepcopy(videos)
            metadata={'epoch':None,'base_model_sha256':base_hash}
        else:
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            if ckpt['config']['smoke']:raise ValueError('Cannot evaluate smoke as actual trained candidate')
            if ckpt['config']['base_model_sha256']!=base_hash:raise ValueError('Base graph mismatch')
            model=TemporalResidual();model.load_state_dict(ckpt['state_dict']);model.eval()
            transformed=classify_videos(model,videos)
            metadata={'epoch':ckpt['epoch'],'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'base_model_sha256':base_hash}
        measured=video_metrics(transformed)
        output={'model':name,'metadata':metadata,'scope':'Causal8frameTransformer residual on named visual graph predictions. Subject-heldback controlledspeech; sourcecliplabels are not expertperframe naturalfacialtruth. CAERcliplabeled numbers diagnostic only.',
                'target_sampling_fps':10,'crop_format':'png','videos':transformed,'metrics':measured}
        dest=args.output/(name+'.json')
        dest.write_text(json.dumps(output,indent=2),encoding='utf-8')
        subprocess.run([shutil.which('node'),str(ROOT/'scripts/evaluate_video_smoothing.cjs'),str(dest),str(dest.with_stem(name+'-display'))],
                       check=True,stdout=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        summary[name]={'metadata':metadata,'metrics':measured}
        print(json.dumps({'name':name,'epoch':metadata['epoch'],'metrics':{k:{n:m[n] for n in ('accuracy','macro_f1_present')} for k,m in measured.items()}}),flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
