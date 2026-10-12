"""Append result assessment jobs to the existing noon worker, without restarting it."""
from datetime import datetime,timezone
import json
from pathlib import Path

work=Path('training/noon-20260928')
queue=json.loads((work/'queue.json').read_text(encoding='utf-8'))
plan_path=work/'regional-plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
status=json.loads((work/'queue-status.json').read_text(encoding='utf-8'))
if 'regional-train20' in status['jobs']:
    raise ValueError('Do not amend acceptance after full regional training starts')
plan['prospective_acceptance']['each_heldback_eyewear_subject_anger_no_drop']=True
plan['pretraining_clarification']={'time':datetime.now(timezone.utc).isoformat(),
                                  'detail':'Each heldback eyeglasses actor must also not lose anger clip/stable successes; declared before actual regional feature extraction or training.'}
plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
jobs=[{'id':'paired-photo-assessment','args':['training/assess_noon_photos.py'],
       'depends_on':['paired-photos'],'require_after':[str(work/'paired-photo-assessment.json')]}]
run='training/runs/regional-temporal-v1';videos=str(work/'regional-video')
jobs.extend([
 {'id':'regional-ravdess-stratification','args':['training/stratify_video_validation.py','--reports',videos,
    '--output',run+'/ravdess-intensity.json'],'depends_on':['regional-validation'],'require_after':[run+'/ravdess-intensity.json']},
 {'id':'regional-eyewear-stratification','args':['training/stratify_enterface_video.py','--reports',videos,
    '--output',run+'/enterface-eyewear.json'],'depends_on':['regional-validation'],'require_after':[run+'/enterface-eyewear.json']},
 {'id':'regional-acceptance','args':['training/assess_temporal_candidates.py','--run',run,'--videos',videos,
    '--plan',str(plan_path),'--baseline-name','released-fp32','--opened-historical-holdout'],
    'depends_on':['regional-ravdess-stratification','regional-eyewear-stratification'],
    'require_after':[run+'/video-acceptance.json']}
])
existing={j['id'] for j in queue}
for job in jobs:
    if job['id'] in existing:continue
    if job['id']=='paired-photo-assessment':
        index=next(i for i,j in enumerate(queue) if j['id']=='paired-photos')+1
        queue.insert(index,job)
    else:queue.append(job)
temporary=work/'queue.tmp';temporary.write_text(json.dumps(queue,indent=2),encoding='utf-8');temporary.replace(work/'queue.json')
print('Result assessment queued; no change to active training')
