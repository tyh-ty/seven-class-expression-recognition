"""Official native-seven ONNX research benchmark; product stays a pure ViT."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

from dataset_contract import CLASSES
from fit_poster_frozen_head import measure

GRAPH=Path('models/research/emotieff-enet-b2-seven/model.onnx')
EXPECTED='7863032d8c0c0bc259aa0dd7b8ebf6b607af454e7a0ea82be2b37767914ecd0f'


def preprocess(image_bgr):
    rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    # Match pinned author's ONNX preprocessing: bilinear resize; normalization
    # initially float64, then NCHW float32. Its native7 label order matches ours.
    x=cv2.resize(rgb,(260,260))/255
    for i,(mean,std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):x[...,i]=(x[...,i]-mean)/std
    return x.transpose(2,0,1).astype(np.float32)[None]


def make_session():
    if hashlib.sha256(GRAPH.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Official graph bytes changed')
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(GRAPH),sess_options=opts,providers=['CPUExecutionProvider'])
    if session.get_outputs()[0].shape[-1]!=7:raise ValueError('Graph is not native seven classes')
    return session


def main():
    cv2.setNumThreads(1);torch.set_num_threads(2)
    work=Path('training/noon-20260928')
    plan={'declared_at':datetime.now(timezone.utc).isoformat(),'official_repository':'https://github.com/sb-ai-lab/EmotiEffLib',
          'commit':'520a051c64cd191521e5934655314e769a319684','graph_sha256':EXPECTED,
          'papers':['https://proceedings.mlr.press/v202/savchenko23a.html','https://ieeexplore.ieee.org/document/9815154'],
          'hypothesis':'Face-identity VGGFace2 initialization plus AffectNet FER learning may provide more transferable soft supervision than ImageNet-only initialization.',
          'method':'First benchmark exact native7 official pretrained CNN as possible OFFLINE TEACHER. Current product and model description remain pureViT.',
          'class_proof':'Pinned emotiefflib/facial_analysis.py lines81-90: Anger,Disgust,Fear,Happiness,Neutral,Sadness,Surprise.',
          'preprocess_proof':'Pinned code lines104-115,393-407: RGB,260x260bilinear,ImageNet normalization.',
          'teacher_gate':{'expw_macro_f1_gain_min':.02,'expanded_real_eyewear_macro_f1_gain_min':.03,
                          'expw_anger_fear_mean_recall_no_drop':True},
          'deployment':'Research teacher qualification does not authorize model replacement. All student photo/video/perclass gates still apply.',
          'scope':'Repeated selection validation. Original final2610/180 excluded. No newly downloaded images.'}
    (work/'emotieff-teacher-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    source=json.loads(Path('training/runs/poster-frozen-seven-v1/image-manifest.json').read_text(encoding='utf-8'))
    if source['smoke']:raise ValueError('Full validation only')
    session=make_session();reports={};timings=[];start=time.perf_counter()
    for name,rows in source['validation'].items():
        scores=[];records=[]
        for row in rows:
            raw=Path(row['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Frozen image changed')
            image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
            tensor=preprocess(image);t=time.perf_counter()
            logits=session.run(None,{session.get_inputs()[0].name:tensor})[0].reshape(7)
            timings.append((time.perf_counter()-t)*1000);scores.append(logits)
            p=np.exp(logits-logits.max());p/=p.sum()
            records.append({'file':row['file'],'label':row['label'],'predicted':CLASSES[int(p.argmax())],'probabilities':p.tolist()})
        measured=measure(torch.from_numpy(np.stack(scores)),torch.tensor([CLASSES.index(r['label']) for r in rows]))
        reports[name]={**measured,'records':records}
        print(json.dumps({'dataset':name,'accuracy':measured['accuracy'],'macro_f1':measured['macro_f1'],
                          'anger':measured['per_class']['anger'],'fear':measured['per_class']['fear'],
                          'elapsed_seconds':round(time.perf_counter()-start,1)}),flush=True)
    raw={'model':'official EmotiEffLib enet_b2_7','classes':CLASSES,'graph_sha256':EXPECTED,
         'preprocessing':plan['preprocess_proof'],'runtime':'Actual ONNX CPU batch1,2threads',
         'latency_median_ms':float(np.median(timings[10:])),'latency_p95_ms':float(np.quantile(timings[10:],.95)),
         'scope':plan['scope'],'reports':reports}
    (work/'emotieff-teacher-photos.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
    prior=json.loads((work/'paired-photos.json').read_text(encoding='utf-8'))
    key=next(k for k in prior['reports'] if Path(k).name=='derived.pt');baseline=prior['reports'][key]
    for n,rows in reports.items():
        if [(r['file'],r['label']) for r in rows['records']]!=[(r['file'],r['label']) for r in baseline[n]['records']]:
            raise ValueError('Baseline sample set differs')
    g=plan['teacher_gate']
    checks={'expw_macro_f1_gain':reports['expw_original']['macro_f1']-baseline['expw_original']['macro_f1']>=g['expw_macro_f1_gain_min'],
            'expanded_real_eyewear_gain':reports['real_eyeglasses_expanded']['macro_f1']-baseline['real_eyeglasses_expanded']['macro_f1']>=g['expanded_real_eyewear_macro_f1_gain_min'],
            'expw_anger_fear_mean_recall_no_drop':sum(reports['expw_original']['per_class'][c]['recall']-baseline['expw_original']['per_class'][c]['recall'] for c in ('anger','fear'))>=0}
    decision={'checks':checks,'qualified_as_research_teacher':all(checks.values()),'deployed':False,
              'note':'No direct replacement. If qualified, only existing original training labels remain truth; teacher distributions provide auxiliary training loss.'}
    (work/'emotieff-teacher-assessment.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
