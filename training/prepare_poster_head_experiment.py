"""Declare a fresh classifier trial before looking at any trained-head outcome."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

work=Path('training/noon-20260928')
validation=[('ferplus_clean','data/ferplus/validation-dedup-v1'),
            ('expw_original','data/seven-class/expw-original-eyewear-v3/expw-val'),
            ('real_eyeglasses_strict','data/expw/real-eyeglasses-val-v2'),
            ('real_eyeglasses_expanded','data/expw/real-eyeglasses-original-expanded-val-v1'),
            ('ravdess','data/ravdess/faces-seven/val'),
            ('enterface','data/seven-class/expw-enterface-v1/enterface-val'),
            ('enterface_eyewear','data/seven-class/expw-enterface-v1/enterface-eyewear-val')]
run='training/runs/poster-frozen-seven-v1'
args=['training/fit_poster_frozen_head.py','--output',run,'--epochs','30','--per-class','3000',
      '--device','cuda','--batch-size','16','--workers','2']
for name,path in validation:args+=['--dataset',name,path]
plan={'declared_at':datetime.now(timezone.utc).isoformat(),
      'hypothesis':'Landmark-guided regional frozen features may preserve eye/mouth evidence missing in gray whole-face ViT. Learn an explicit new7class head to avoid unknown original output mapping.',
      'source':'Official POSTER V2 code MIT + thirdparty pinned mirror weights. Original author-byte equivalence unknown; research only until provenance/deployment reviewed.',
      'head':'Discard original head entirely. Newly initialized768to7 linear, train-only normalization, CE smoothing.05, AdamW.003,30epochs CPU after GPUfeature extraction.',
      'data':'Existing training images only, SHA ordering up to3000 perclass, actual-eyewear training photos prioritized; no new dataset downloaded.',
      'candidates':['best_target','epoch10','epoch20','epoch30'],
      'selection':'Same repeated validation datasets as paired plan. No new blind accuracy claim; previous2610/180finaltest excluded.',
      'deployment_gates_from':'sam-paired-plan.json',
      'speed_warning':'PyTorchCPU feature-only389ms/face. No direct deployment; would require qualifying CPU ONNX/INT8 with accuracy preserved, or offline teacher path.',
      'job':{'id':'poster-frozen-head30','args':args,'require_after':[run+'/epoch-030.pt']}}
(work/'poster-head-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
# A tiny validation fixture is for execution/parity only, never reported as accuracy.
fixture=work/'poster-smoke-validation'
for directory in sorted(Path(validation[0][1]).iterdir()):
    if not directory.is_dir():continue
    source=next(p for p in sorted(directory.iterdir()) if p.suffix.lower() in ('.jpg','.png','.jpeg'))
    target=fixture/directory.name/source.name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
smoke=[sys.executable,'-X','utf8','-u','training/fit_poster_frozen_head.py',
       '--output',str(work/'poster-head-smoke'),'--epochs','1','--per-class','2','--batch-size','2',
       '--workers','0','--device','cuda','--smoke','--dataset','execution_only',str(fixture)]
with (work/'poster-head-smoke.log').open('w',encoding='utf-8') as f:
    subprocess.run(smoke,stdout=f,stderr=subprocess.STDOUT,check=True)
queue=json.loads((work/'queue.json').read_text(encoding='utf-8'))
if plan['job']['id'] not in {j['id'] for j in queue}:
    queue.append(plan['job'])
    temp=work/'queue.tmp';temp.write_text(json.dumps(queue,indent=2),encoding='utf-8');temp.replace(work/'queue.json')
print('CUDA feature/head smoke passed, new labeled-head experiment queued',flush=True)
