"""Declare matched experiments before any outcome is observed."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

work=Path('training/noon-20260928')
base='training/runs/vit-small-expw-enterface-v1/weight-blends/enterface2-6-average-fp32/derived.pt'
common=['--data','data/seven-class/expw-enterface-v1','--initial-checkpoint',base,
        '--teacher-checkpoint',base,'--validation-data','data/ferplus/validation-dedup-v1',
        '--target-validation-data','data/seven-class/expw-original-eyewear-v3/expw-val',
        '--require-disjoint-validation','--epochs','4','--patience','4',
        '--batch-size','40','--workers','2','--learning-rate','0.000003',
        '--head-learning-rate','0.00001','--trainable-backbone-blocks','6',
        '--distillation-weight','1.0','--distillation-scope','base-and-correct-extra',
        '--new-source-fraction','0.25','--real-eyewear-multiplier','4',
        '--real-eyewear-class-weighting','inverse-sqrt',
        '--extra-source-class-weighting','inverse-sqrt','--primary-f1-tolerance','0.01',
        '--primary-f1-floor','0.792222','--no-evaluate-test','--save-all-epochs',
        '--device','cuda','--new-source-prefix','web_expw_',
        '--new-source-prefix','web_enterface_','--seed','817']
jobs=[]
for name,script,extra in [('sam','training/train_camera_sam.py',['--sam-rho','0.05']),
                          ('adamw','training/train_camera_robust.py',[])]:
    output=f'training/runs/vit-small-noon-{name}-v1'
    jobs.append({'id':f'{name}-paired4','args':[script,'--output',output,*extra,*common],
                 'require_before':[base], 'require_after':[output+'/epoch-004.pt']})
plan={'declared_at':datetime.now(timezone.utc).isoformat(),'paper':'https://arxiv.org/abs/2010.01412',
      'question':'Does SAM improve held-back-person/camera/real-eyewear validation over matched AdamW adaptation?',
      'baseline':base, 'baseline_onnx':'models/vit_emotion_seven_fp32.onnx',
      'baseline_onnx_sha256':'f9753ef6496707319576794704a0bae7674b46368305edbf366d538b2a7ce54f',
      'paired_control':'Same initialization/data/seed/sampler/augmentation/LR/epochs; SAM uses two passes per batch.',
      'training_count':2,'epochs_each':4,'selection':'Selection validation only; pre-existing final test remains excluded.',
      'evaluation_candidates':['baseline','sam epoch2','sam epoch4','adamw epoch2','adamw epoch4'],
      'deployment_gate':{'clean_fer_macro_f1_min':.792222,'expw_macro_f1_improvement_min':.005,
                        'real_eyewear_macro_f1_no_decrease':True,
                        'per_class_recall_max_drop':.05,
                        'video_fear_and_anger_and_each_eye_actor_no_decrease':True,
                        'actual_cpu_and_latency_required':True},
      'holdout_warning':'ExpW2610/eNTER180 final tests already viewed; never use for tuning or label fresh blind tests.',
      'jobs':jobs}
(work/'sam-paired-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(work/'queue.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
smoke=[sys.executable,'-X','utf8','-u','training/train_camera_sam.py','--output',str(work/'sam-cuda-smoke'),
       '--sam-rho','0.05',*common,'--epochs','1','--max-train-batches','2','--max-eval-batches','2']
with (work/'sam-cuda-smoke.log').open('w',encoding='utf-8') as f:
    subprocess.run(smoke,stdout=f,stderr=subprocess.STDOUT,check=True)
print('CUDA two-update smoke passed; declared paired plan ready',flush=True)
