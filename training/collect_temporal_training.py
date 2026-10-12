"""Collect causal-training inputs from train actors using the unchanged deployed API model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server
from screen_caer_pilot import camera_frame

EMOTIONS={'01':'neutral','03':'happiness','04':'sadness','05':'anger','06':'fear','07':'disgust','08':'surprise'}
CLASSES=tuple(e[0] for e in server.EMOTIONS)
EYE_SUBJECTS={8,19,20,22,23,35,42}


def training_records():
    rows=[]
    for actor in range(1,17):
        archive=Path(f'D:/RAVDESS/Video_Speech_Actor_{actor:02d}.zip')
        with ZipFile(archive) as z:
            for entry in sorted(z.namelist()):
                stem=Path(entry).stem
                parts=stem.split('-')
                if len(parts)!=7 or parts[:2]!=['01','01'] or parts[2] not in EMOTIONS:
                    continue
                assert int(parts[-1])==actor
                rows.append({'id':'ravdess-'+stem,'source':'RAVDESS-train','split':'train','label':EMOTIONS[parts[2]],
                             'actor':actor,'subject_key':f'ravdess-{actor}','archive':str(archive),'entry':entry,
                             'intensity':parts[3],'eyewear':False,'known_label_window':[.35,.75]})
    manifest=Path('D:/eNTERFACE-source/manifest-stratified-v2.json')
    data=json.loads(manifest.read_text(encoding='utf-8'))
    for r in data['videos']:
        if r['split']!='train' or r['sentence'] is None:
            continue
        assert r['subject'] not in {9,13}
        rows.append({'id':'enterface-'+hashlib.sha256(r['source_entry'].encode()).hexdigest()[:16],
                     'source':'eNTERFACE-train','split':'train','label':r['label'],'subject':r['subject'],
                     'subject_key':f'enterface-{r["subject"]}','path':r['file'],'source_entry':r['source_entry'],
                     'source_sha256':r['sha256'],'eyewear':r['subject'] in EYE_SUBJECTS,'known_label_window':[.2,.8]})
    val=json.loads((ROOT/'data/video-validation-combined-10fps-v1/manifest.json').read_text(encoding='utf-8'))
    assert not {r['id'] for r in rows}&{r['id'] for r in val['videos']}
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('data/temporal-training-int8-v1'))
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    model=server.EmotionRecognizer()
    model_hash=hashlib.sha256(server.TRANSFORMER_MODEL_PATH.read_bytes()).hexdigest()
    policy={'sampling_fps':10,'camera_max_side':512,'jpeg_quality':85,'model_sha256':model_hash,
            'server_sha256':hashlib.sha256((ROOT/'server.py').read_bytes()).hexdigest()}
    records=training_records()
    if args.limit:
        records=records[:args.limit]
    result=[]
    start=time.perf_counter()
    processed=0
    for index,row in enumerate(records,1):
        dest=args.output/(row['id']+'.json')
        if dest.exists():
            saved=json.loads(dest.read_text(encoding='utf-8'))
            if saved['policy']!=policy:
                raise ValueError('Model/preprocessing changed; use new output directory')
            result.append(saved)
            continue
        temporary=None
        if 'archive' in row:
            temporary=args.output/(row['id']+'.source.mp4')
            with ZipFile(row['archive']) as z:
                raw=z.read(row['entry'])
            row['source_sha256']=hashlib.sha256(raw).hexdigest()
            temporary.write_bytes(raw)
            path=temporary
        else:
            path=Path(row['path'])
        cap=cv2.VideoCapture(str(path))
        frames=[]
        try:
            fps=float(cap.get(cv2.CAP_PROP_FPS))
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps<=0 or count<=0:
                raise ValueError(f'Cannot decode {path}')
            # Read fixed complete clips; supervise only existing central label windows later.
            wanted={min(count-1,round(t*fps/10)) for t in range(int(count/fps*10+.99999))}
            frame_index=0
            while cap.grab():
                if frame_index in wanted:
                    ok,original=cap.retrieve()
                    if not ok:
                        raise ValueError(f'Decode failed {path}:{frame_index}')
                    image=camera_frame(original,max_side=512)
                    output=model.analyze(image)
                    face=output['faces'][0] if output['faces'] else None
                    item={'index':frame_index,'seconds':round(frame_index/fps,4),
                          'fraction':frame_index/max(1,count-1),'status':'detected' if face else 'no_face',
                          'width':output['width'],'height':output['height'],'face_count':len(output['faces'])}
                    if face:
                        scores={s['id']:s['score'] for s in face['scores']}
                        item.update(probabilities=[scores[c] for c in CLASSES],bbox=face['bbox'],face_score=face['faceConfidence'])
                    frames.append(item)
                    processed+=1
                frame_index+=1
            saved={**row,'policy':policy,'fps':fps,'source_frames':count,'sampled_frames':len(frames),'frames':frames}
            dest.write_text(json.dumps(saved,indent=2),encoding='utf-8')
            result.append(saved)
        finally:
            cap.release()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if index%10==0 or index==len(records):
            print(json.dumps({'videos':index,'total_videos':len(records),'new_frames':processed,
                              'seconds':round(time.perf_counter()-start,1)}),flush=True)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'policy':policy,
            'scope':'Training actors only: RAVDESS1-16 and29eNTERFACEv2train subjects. Source clip labels, central windows only; not spontaneous/micro-expression facial truth. Existing validation and finalholdout excluded.',
            'partial_smoke_only':args.limit is not None,'classes':CLASSES,'videos':result,
            'counts':dict(Counter(r['source']+'/'+r['label'] for r in result))}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPLETE',len(result),flush=True)


if __name__=='__main__':
    main()
