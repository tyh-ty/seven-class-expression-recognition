"""Declare paired consistency experiment, smoke-test, append to live queue."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


def main():
    work=Path('training/noon-20260928')
    plan_path=work/'consistency-paired-plan.json'
    if plan_path.exists():raise RuntimeError('Plan already declared; inspect results, do not reset experiment')
    old=json.loads((work/'sam-paired-plan.json').read_text(encoding='utf-8'))
    base=old['baseline']
    common=old['jobs'][1]['args'][3:]  # Skip script, --output and old run directory.
    for flag,value in [('--batch-size','20'),('--seed','931')]:common[common.index(flag)+1]=value
    jobs=[];candidates=[base]
    for name,weight in [('jsd',12),('control',0)]:
        out=f'training/runs/vit-small-consistency-{name}-v1'
        jobs.append({'id':f'consistency-{name}4','args':['training/train_camera_consistency.py','--output',out,
                    '--jsd-weight',str(weight),*common],'require_before':[base],
                    'require_after':[out+'/epoch-004.pt']})
        candidates.extend(out+f'/epoch-{epoch:03d}.pt' for epoch in (1,2,4))
    existing=json.loads((work/'queue.json').read_text(encoding='utf-8'))
    photo_args=next(j for j in existing if j['id']=='paired-photos')['args']
    datasets=photo_args[photo_args.index('--dataset'):]
    # Retain only --dataset pairs, excluding prior output/runtime options.
    dataset_args=[]
    for i,v in enumerate(datasets):
        if v=='--dataset':dataset_args.extend(datasets[i:i+3])
    photo_output=str(work/'consistency-photos.json');assessment=str(work/'consistency-photo-assessment.json')
    jobs.extend([
        {'id':'consistency-photos','args':['training/compare_checkpoints.py','--candidates',*candidates,
         *dataset_args,'--output',photo_output,'--batch-size','40','--save-predictions','--inference-precision','float32'],
         'depends_on':['consistency-jsd4','consistency-control4'],'require_after':[photo_output]},
        {'id':'consistency-assessment','args':['training/assess_noon_photos.py','--report',photo_output,
         '--plan',str(plan_path),'--output',assessment],'depends_on':['consistency-photos'],'require_after':[assessment]}])
    plan={'declared_at':datetime.now(timezone.utc).isoformat(),'baseline':base,'baseline_onnx':old['baseline_onnx'],
          'baseline_onnx_sha256':old['baseline_onnx_sha256'],'deployment_gate':old['deployment_gate'],
          'paper':'https://arxiv.org/abs/1912.02781','code_commit':'9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4',
          'hypothesis':'Agreement across original, landmark-eyewear and mild camera views may improve occlusion robustness without adding inference cost.',
          'adaptation':'Three-view JSD from AugMix, not its mixed augmentation chains. Mean supervised CE across all three views; JSD only when baseline clean prediction matches original label.',
          'paired_control':'Same3views, seed931, baseline, data, 4epochs, last6blocks, LR3e-6/head1e-5, batch20. Only JSD12 vs0 differs.',
          'teacher':'Original released ViT baseline only. Rejected EmotiEff CNN not used.',
          'truth':'Original source labels unchanged. No teacher predictions used as truth. No new training data.',
          'selection':'Epoch1/2/4, repeated selection validation only. Opened final2610/180 excluded.',
          'limits':'Synthetic glasses do not establish real-glasses robustness. Occlusion can remove genuine class evidence; exact real-video/recall gates stay unchanged.',
          'jobs':jobs}
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','training/check_camera_consistency.py'],check=True)
    smoke=work/'consistency-cuda-smoke'
    with (work/'consistency-cuda-smoke.log').open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-X','utf8','-u','training/train_camera_consistency.py','--output',str(smoke),
                        '--jsd-weight','12',*common,'--epochs','1','--max-train-batches','2','--max-eval-batches','2'],
                       stdout=f,stderr=subprocess.STDOUT,check=True)
    initial=torch.load(base,map_location='cpu',weights_only=True)['model_state']
    result=torch.load(smoke/'epoch-001.pt',map_location='cpu',weights_only=True)
    changed=[n for n,v in initial.items() if not torch.equal(v,result['model_state'][n])]
    assert any(n.startswith('blocks.11.') for n in changed) and any(n.startswith('head.') for n in changed)
    assert not any(n.startswith(('blocks.0.','patch_embed.')) for n in changed)
    check={'changed_parameter_tensors':len(changed),'last_block_and_head_updated':True,'first_block_and_stem_fixed':True,
           'smoke_checkpoint_sha256':hashlib.sha256((smoke/'epoch-001.pt').read_bytes()).hexdigest()}
    (work/'consistency-cuda-check.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    # The existing runner owns status and GPU serialization. Never restart it here.
    existing=json.loads((work/'queue.json').read_text(encoding='utf-8'))
    if any(j['id'].startswith('consistency-') for j in existing):raise RuntimeError('Duplicate consistency job')
    temp=work/'queue.tmp';temp.write_text(json.dumps(existing+jobs,indent=2),encoding='utf-8');temp.replace(work/'queue.json')
    print(json.dumps({'queued':[j['id'] for j in jobs],'cuda_smoke':check}),flush=True)


if __name__=='__main__':main()
