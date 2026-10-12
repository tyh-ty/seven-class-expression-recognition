"""Append explicit sequential work without restarting a live GPU task."""
from datetime import datetime,timezone
import json
from pathlib import Path

work=Path('training/noon-20260928')
queue=json.loads((work/'queue.json').read_text(encoding='utf-8'))
base='training/runs/vit-small-expw-enterface-v1/weight-blends/enterface2-6-average-fp32/derived.pt'
candidates=[base]+[f'training/runs/vit-small-noon-{method}-v1/epoch-{epoch:03d}.pt' for method in ('sam','adamw') for epoch in (2,4)]
photo=['training/compare_checkpoints.py','--candidates',*candidates,'--output',str(work/'paired-photos.json'),
       '--batch-size','40','--save-predictions','--inference-precision','float32']
for name,path in [('ferplus_clean','data/ferplus/validation-dedup-v1'),
                  ('expw_original','data/seven-class/expw-original-eyewear-v3/expw-val'),
                  ('real_eyeglasses_strict','data/expw/real-eyeglasses-val-v2'),
                  ('real_eyeglasses_expanded','data/expw/real-eyeglasses-original-expanded-val-v1'),
                  ('ravdess','data/ravdess/faces-seven/val'),
                  ('enterface','data/seven-class/expw-enterface-v1/enterface-val'),
                  ('enterface_eyewear','data/seven-class/expw-enterface-v1/enterface-eyewear-val')]:
    photo+=['--dataset',name,path]
jobs=[{'id':'paired-photos','args':photo,'depends_on':['sam-paired4','adamw-paired4'],
       'require_after':[str(work/'paired-photos.json')]}]
for split,manifest,out in [('train','data/temporal-training-enterface2-fp32-v1/manifest.json','data/regional-temporal-train-v1'),
                           ('validation','data/video-validation-combined-10fps-v1/manifest.json','data/regional-temporal-validation-v1')]:
    jobs.append({'id':f'regional-{split}-features','args':['training/extract_regional_video_features.py',
                 '--manifest',manifest,'--split',split,'--checkpoint',base,'--graph','models/vit_emotion_seven_fp32.onnx',
                 '--output',out,'--device','cuda','--batch-size','40'],
                 'require_after':[out+'/manifest.json']})
run='training/runs/regional-temporal-v1'
jobs.append({'id':'regional-train20','args':['training/train_regional_temporal.py',
             '--train','data/regional-temporal-train-v1/manifest.json',
             '--validation','data/regional-temporal-validation-v1/manifest.json','--output',run,'--epochs','20'],
             'depends_on':['regional-train-features','regional-validation-features'],
             'require_after':[run+'/epoch-020.pt']})
jobs.append({'id':'regional-validation','args':['training/evaluate_regional_temporal.py',
             '--validation','data/regional-temporal-validation-v1/manifest.json','--checkpoint',run+'/best.pt',
             run+'/epoch-005.pt',run+'/epoch-010.pt',run+'/epoch-020.pt','--output',str(work/'regional-video')],
             'depends_on':['regional-train20'],'require_after':[str(work/'regional-video/summary.json')]})
plan={'declared_at':datetime.now(timezone.utc).isoformat(),
      'hypothesis':'Retain regional visual evidence lost in 7D probability history; learn causal change from controlled speech training videos.',
      'inspiration':'MAE-DFER ACM MM2023 regional/global and appearance/motion concepts; not a reproduction or use of its pretrained weights.',
      'visual':'Frozen released gray ViT-S/16; normalized CLS+upper/lower seven patch rows, 3x384 features per frame.',
      'temporal':'8 past/current frames, 3 regions, 64dim 4head one-layer Transformer. Bounded2tanh residual minus repeated-current residual; exact static invariance.',
      'data':'Existing1699 train-only clips; RAV832 actors1-16 + eNTER867 clips29actors. Validation208RAV +240eNTER remains subject disjoint; CAER90 diagnostic only.',
      'optimization':'CPU20epochs AdamW3e-4 batch128 CE smoothing.05 +KL.5; class/source/clip balanced, actual eyewear2x. No supervised feature-backbone update.',
      'baseline_graph_sha256':'f9753ef6496707319576794704a0bae7674b46368305edbf366d538b2a7ce54f',
      'validation_candidates':['best','epoch5','epoch10','epoch20'],
      'prospective_acceptance':{'both_RAV_and_enterface_clip_macro_f1_gain_min':.015,
          'both_RAV_and_enterface_halfsecond_stability_gain_min':.03,
          'per_class_drop_max_if_support20':.05,'RAV_normal_anger_halfsecond_gain_clips_min':2,
          'RAV_normal_fear_halfsecond_no_drop':True,'each_heldback_eyewear_subject_fear_correct_clips_gain_min':1,
          'each_heldback_eyewear_subject_fear_stable_halfsecond_gain_clips_min':1,
          'frozen_CAER_visibleface_agreement_drop_max':.02,'photos_exact_static_invariance':True,
          'runtime_classification_plus_temporal_singleface_p95_ms_max':150},
      'limits':'No new independent test yet. Controlled speech and weak clip labels cannot prove natural microexpression accuracy; previouslyopened2610/180excluded.',
      'smoke':'Exact static/baseline/no-future/no-face/IoU guards, real parameter update, CPUfeature/ONNX maxlogiterror7.25e-6; oneclip training only functional smoke.',
      'jobs':jobs}
(work/'regional-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
existing={j['id'] for j in queue}
queue.extend(j for j in jobs if j['id'] not in existing)
temporary=work/'queue.tmp';temporary.write_text(json.dumps(queue,indent=2),encoding='utf-8');temporary.replace(work/'queue.json')
state=Path('training/overnight-20260927/state.json')
s=json.loads(state.read_text(encoding='utf-8'))
s['last_updated']=datetime.now(timezone.utc).isoformat()
s['current_work']=['Continue authorized research/training to 12:00 Beijing. SAM paired4 running; matched AdamW4 next.',
                   'Regional visual temporal pipeline prepared and numerically checked, queued after paired runs.',
                   'Published morning FP32 source/service/sole ZIP remains unchanged pending evidence.']
s['next_actions']=['Monitor noon queue; no second GPU worker. Analyze paired FP32 photos; actual CPU/video required before replacement.',
                   'Complete regional extraction/train20/validation with subject isolation. Do not use old final test for tuning.',
                   'At12:00 stop new training, finalize honest report/package if qualified, pause heartbeat.']
s['noon_phase'].update(status='SAM paired4 running, queue includes matchedAdamW4, photos and regionaltemporal experiment',
                       queue_file=str(work/'queue.json'),queue_status=str(work/'queue-status.json'),
                       queue_pid=81980,training_pid=84464,
                       plans=[str(work/'sam-paired-plan.json'),str(work/'regional-plan.json')],
                       poster_mirror='Downloaded SHA verified; no source-author label-order confirmation, not yet evaluated or deployed',
                       mae_weights='Official Google Drive and usercontent both connection timeouts; no weights')
tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(state)
print('Appended',len(jobs),'jobs; preserved existing running queue')
