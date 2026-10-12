"""Compare camera capture sizes on fixed real-eyewear validation source images."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from screen_caer_pilot import camera_frame
import server

CLASSES = ('anger','disgust','fear','happiness','neutral','sadness','surprise')


def iou(first, second):
    x,y,w,h = first
    a,b,c,d = second
    intersection = max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
    return float(intersection/max(1,w*h+c*d-intersection))


def main():
    cv2.setNumThreads(1)
    model = server.EmotionRecognizer()
    source = Path('data/seven-class/expw-original-eyewear-v3/dataset.json')
    metadata = json.loads(source.read_text())
    names = {p.name for p in Path('data/expw/real-eyeglasses-original-expanded-val-v1').glob('*/*') if p.is_file()}
    rows = [r for r in metadata['images'] if r['split']=='expw-val' and Path(r['path']).name in names]
    assert len(rows)==196
    source_paths = {p.name:p for p in Path('D:/ExpW-source/original').rglob('*.jpg')}
    results = {512:[],768:[]}
    for index,row in enumerate(rows):
        original = cv2.imdecode(np.fromfile(source_paths[row['source']],dtype=np.uint8),cv2.IMREAD_COLOR)
        for side in ((512,768) if index%2==0 else (768,512)):
            frame = camera_frame(original,max_side=side)
            ratio = np.array([frame.shape[1]/original.shape[1],frame.shape[0]/original.shape[0]])
            left,top,right,bottom = row['annotation_bbox']
            reference = np.array([left,top,right-left,bottom-top])*np.tile(ratio,2)
            started=perf_counter();output=model.analyze(frame);elapsed=(perf_counter()-started)*1000
            faces=output['faces'];matched=max(faces,key=lambda f:iou(list(f['bbox'].values()),reference),default=None)
            overlap=iou(list(matched['bbox'].values()),reference) if matched else 0.
            detected=overlap>=.25
            results[side].append({'source':row['source'],'label':row['label'],'sha256':row['sha256'],
                'predicted':matched['emotion']['id'] if detected else None,
                'scores':matched['scores'] if detected else None, 'face_found':detected,
                'target_is_nearest':bool(detected and matched is faces[0]),'match_iou':overlap,
                'all_faces':len(faces),'latency_ms':elapsed,'timing_ms':output['timingMs'],
                'source_resolution':[original.shape[1],original.shape[0]],'capture_resolution':[frame.shape[1],frame.shape[0]]})
        if (index+1)%40==0:print('EYEWARE_CAPTURE',index+1,'/',len(rows),flush=True)
    summary={}
    for side,records in results.items():
        matrix=np.zeros((7,8),dtype=np.int64)
        for row in records:matrix[CLASSES.index(row['label']),CLASSES.index(row['predicted']) if row['predicted'] else 7]+=1
        tp=matrix[:,:7].diagonal();support=matrix.sum(1);recall=tp/np.maximum(1,support);precision=tp/np.maximum(1,matrix[:,:7].sum(0));f1=2*precision*recall/np.maximum(1e-12,precision+recall)
        singles=[r['latency_ms'] for r in records if r['all_faces']==1]
        summary[str(side)]={'examples':len(records),'matched_faces':sum(r['face_found'] for r in records),
            'target_nearest_count':sum(r['target_is_nearest'] for r in records),
            'accuracy_including_detection_misses':float(tp.sum()/len(records)),'macro_f1':float(f1.mean()),
            'per_class':{c:{'support':int(support[i]),'recall':float(recall[i]),'precision':float(precision[i]),'f1':float(f1[i])} for i,c in enumerate(CLASSES)},
            'all_faces_api_mean_ms':float(np.mean([r['latency_ms'] for r in records])),
            'single_face_median_ms':float(np.median(singles)),'single_face_p95_ms':float(np.percentile(singles,95)),
            'confusion_with_no_face_column':matrix.tolist(),'records':records}
    report={'scope':'Fixed196ExpW ordinary-eyewear validation sourceimages. Actual server.analyze CPU graph and detection, targetmatched by sourceannotation IoU>=.25 among atmost4returnedfaces; not alltargets are the nearest person. Alternating resolutionorder. Same originalclasslabels, noholdout. Runtime measured while training/preparation run, not fullbrowser capture or labPC timing.',
            'model_sha256':hashlib.sha256(server.TRANSFORMER_MODEL_PATH.read_bytes()).hexdigest(),
            'source_metadata_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reports':summary}
    Path('training/overnight-20260927/capture-resolution-eyewear.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in summary.items()}),flush=True)


if __name__=='__main__':
    main()
