"""Evaluate fixed global two-Transformer mixtures on saved, identical video frames."""
from collections import defaultdict
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CLASSES=('anger','disgust','fear','happiness','neutral','sadness','surprise')

def main():
    source=ROOT/'training/overnight-20260927/video-evaluation-v1'
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-report',type=Path,default=source/'vit-small-seven-glasses-v1-best_target-zoom1.json')
    parser.add_argument('--other-report',type=Path,default=source/'vit-base-expw-eyewear-v1-best_target-zoom1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'training/overnight-20260927/video-ensemble-validation')
    args=parser.parse_args()
    small=json.loads(args.base_report.read_text(encoding='utf-8'))
    large=json.loads(args.other_report.read_text(encoding='utf-8'))
    output=args.output;output.mkdir(exist_ok=True)
    summary={}
    for weight in (.25,.5,.75):
        result=copy.deepcopy(small)
        assert len(result['videos'])==len(large['videos'])
        groups=defaultdict(list)
        for clip,other in zip(result['videos'],large['videos']):
            assert (clip['id'],clip['label'],len(clip['frames']))==(other['id'],other['label'],len(other['frames']))
            for frame,other_frame in zip(clip['frames'],other['frames']):
                assert (frame['index'],frame['status'])==(other_frame['index'],other_frame['status'])
                assert frame.get('bbox') == other_frame.get('bbox'), 'Cannot combine predictions for different people/crops'
                if 'probabilities' in frame:
                    frame['probabilities']=((1-weight)*np.array(frame['probabilities'])+weight*np.array(other_frame['probabilities'])).tolist()
            active=[f for f in clip['frames'] if clip['known_label_window'][0]<=f['fraction']<=clip['known_label_window'][1]]
            detected=[f for f in active if 'probabilities' in f]
            clip['aggregate_prediction']=CLASSES[int(np.mean([f['probabilities'] for f in detected],axis=0).argmax())] if detected else None
            groups[clip['source']].append(clip)
        metrics={}
        for domain,clips in groups.items():
            matrix=np.zeros((7,8),dtype=int)
            for clip in clips:matrix[CLASSES.index(clip['label']),CLASSES.index(clip['aggregate_prediction']) if clip['aggregate_prediction'] else 7]+=1
            tp=matrix[:,:7].diagonal();support=matrix.sum(axis=1);precision=tp/np.maximum(1,matrix[:,:7].sum(axis=0));recall=tp/np.maximum(1,support);f1=2*precision*recall/np.maximum(1e-12,precision+recall)
            metrics[domain]={'clips':len(clips),'accuracy_including_detection_misses':float(tp.sum()/len(clips)),
                             'macro_f1':float(f1.mean()),'per_class':{label:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,label in enumerate(CLASSES)},'confusion_with_no_face_column':matrix.tolist()}
        result.update(model=f'Global probability ensemble: {small["model"]}*(1-{weight})+{large["model"]}*{weight}',metrics=metrics,
                      ensemble_sources=[str(args.base_report),str(args.other_report)],
                      ensemble_scope='Three fixed global weights, validation only, no per-class score adjustment. Would require both models in deployment; latency and quantization not verified.')
        result.pop('inference_batch_seconds',None)
        path=output/f'ensemble-{weight:g}.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        subprocess.run([shutil.which('node'),str(ROOT/'scripts/evaluate_video_smoothing.cjs'),str(path),str(path.with_stem(path.stem+'-display'))],check=True,stdout=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        summary[str(weight)]=metrics
        print(json.dumps({'second_model_weight':weight,'metrics':metrics}),flush=True)
    (output/'summary.json').write_text(json.dumps({'scope':'Validation-only ensemble and real frontend replay; source clip labels are not per-frame facial annotations. CAER is scene-label diagnostic only.','results':summary},indent=2),encoding='utf-8')

if __name__=='__main__':main()
