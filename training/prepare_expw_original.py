"""Build camera-shaped crops from original ExpW face boxes, with grouped source/face splits."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import server
from screen_caer_pilot import camera_frame

CLASSES=('anger','disgust','fear','happiness','neutral','sadness','surprise')
SOURCE_CLASSES=('anger','disgust','fear','happiness','sadness','surprise','neutral')
BASE=Path('D:/ExpW-source')
DEADLINE=datetime.fromisoformat('2026-09-28T08:00:00+08:00')
LOCAL=threading.local()

def objects():
    if not hasattr(LOCAL,'detector'):
        LOCAL.detector=cv2.FaceDetectorYN.create(str(server.EmotionRecognizer._opencv_safe_path(server.FACE_DETECTOR_PATH)),'',(320,320),.5,.3,5000)
        LOCAL.identity=cv2.FaceRecognizerSF.create(str(server.EmotionRecognizer._opencv_safe_path(ROOT/'models/face_recognition_sface_2021dec.onnx')),'')
    return LOCAL.detector,LOCAL.identity

def overlap(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);r=min(a[0]+a[2],b[0]+b[2]);bottom=min(a[1]+a[3],b[1]+b[3])
    area=max(0,r-x)*max(0,bottom-y)
    return area/max(a[2]*a[3]+b[2]*b[3]-area,1)

def process(item):
    row,path=item
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:return None,'decode_failure'
    original_h,original_w=image.shape[:2]
    frame=camera_frame(image);height,width=frame.shape[:2]
    left,top,right,bottom=row['annotation_bbox']
    reference=np.asarray([left,top,right-left,bottom-top],dtype=np.float32)
    reference*=np.asarray([width/original_w,height/original_h,width/original_w,height/original_h])
    if min(reference[2:])<28:return None,'tiny_annotated_face'
    detector,identity=objects();detector.setInputSize((width,height));_,faces=detector.detect(frame)
    selected=max(faces,key=lambda f:overlap(f[:4],reference),default=None) if faces is not None else None
    found=selected is not None and overlap(selected[:4],reference)>=.25
    box=selected[:4] if found else reference
    x,y,w,h=[int(round(float(v))) for v in box]
    l,t,cw,ch=server.EmotionRecognizer._square_face_box(x,y,w,h,width,height)
    crop=frame[t:t+ch,l:l+cw]
    if not crop.size:return None,'empty_crop'
    crop=cv2.resize(crop,(224,224),interpolation=cv2.INTER_AREA if min(crop.shape[:2])>=224 else cv2.INTER_LINEAR)
    ok,encoded=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
    if not ok:return None,'encode_failure'
    feature=None;points=None
    if found:
        aligned=identity.alignCrop(frame,selected)
        feature=identity.feature(aligned).reshape(-1).astype(np.float32)
        feature/=max(float(np.linalg.norm(feature)),1e-10)
        points=(selected[4:14].reshape(5,2)-np.asarray([l,t]))*np.asarray([224/cw,224/ch])
    raw=encoded.tobytes();digest=hashlib.sha256(raw).hexdigest()
    return ({**row,'sha256':digest,'camera_face_detected':bool(found),'crop_bbox':[l,t,cw,ch],
             'landmarks':points.tolist() if points is not None else None,
             'laplacian_variance':round(float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()),3)},raw,feature),'ok'

def deduplicate_crops(rows,features):
    """Exclude contradictory pixels, preserving all source aliases for split grouping."""
    hashes=defaultdict(list)
    for i,row in enumerate(rows):hashes[row['sha256']].append(i)
    retained=[];retained_features=[];conflicts=[];duplicates=0
    for digest,members in hashes.items():
        labels={rows[i]['label'] for i in members}
        if len(labels)>1:
            conflicts.append({'sha256':digest,'annotations':[
                {k:rows[i][k] for k in ('source','face_id','label')} for i in members]})
            continue
        representative=next((i for i in members if features[i] is not None),members[0])
        row=dict(rows[representative])
        row['source_aliases']=sorted({rows[i]['source'] for i in members})
        row['real_eyewear']=any(rows[i]['real_eyewear'] for i in members)
        retained.append(row);retained_features.append(features[representative])
        duplicates+=len(members)-1
    return retained,retained_features,conflicts,duplicates

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-extraction',action='store_true');parser.add_argument('--class-cap',type=int,default=6000);args=parser.parse_args()
    cv2.setNumThreads(1)
    while not (BASE/'extraction-report.json').exists():
        if not args.wait_extraction:raise FileNotFoundError('ExpW extraction is not verified yet')
        if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('Deadline reached')
        time.sleep(30)
    output=ROOT/'data/seven-class/expw-original-v1'
    if output.exists():raise FileExistsError(output)
    temp=output.with_name(output.name+'-building');temp.mkdir(exist_ok=True)
    crop_cache=BASE/'camera-crop-cache-v1';crop_cache.mkdir(exist_ok=True)
    index={p.name:p for p in (BASE/'original').rglob('*.jpg')}
    subset=json.loads((ROOT/'data/seven-class/expw-subset-v1/dataset.json').read_text())
    fixed={r['source']:r for r in subset['images']}
    eyewear=json.loads((ROOT/'training/overnight-20260927/expw-eyewear-review.json').read_text())
    glasses={r['source'] for r in eyewear['images'] if r['eyewear']=='clear_eyeglasses'}
    by_label=defaultdict(list)
    for line in (BASE/'label.lst').read_text(encoding='utf-8-sig').splitlines():
        v=line.split()
        if not v:continue
        filename=v[0];label=SOURCE_CLASSES[int(v[-1])]
        if filename not in index:raise FileNotFoundError(filename)
        row={'source':filename,'face_id':int(v[1]),'label':label,'annotation_bbox':[int(v[3]),int(v[2]),int(v[4]),int(v[5])],
             'source_annotation_score':float(v[6]),'real_eyewear':filename in glasses}
        by_label[label].append(row)
    selected=[]
    for label in CLASSES:
        # Keep all previously split subset names, then deterministic unseen additions.
        records=sorted(by_label[label],key=lambda r:(r['source'] not in fixed,hashlib.sha256((r['source']+':'+str(r['face_id'])).encode()).hexdigest()))
        selected.extend(records[:args.class_cap])
    rows=[];features=[];counts=Counter()
    manifest_path=crop_cache/'manifest.jsonl'
    with manifest_path.open('w',encoding='utf-8') as log,ThreadPoolExecutor(max_workers=4) as pool:
        items=((row,index[row['source']]) for row in selected)
        for i,(result,status) in enumerate(pool.map(process,items),1):
            counts[status]+=1
            if result:
                row,raw,feature=result
                path=crop_cache/(row['sha256']+'.jpg');path.write_bytes(raw)
                row['cache_path']=str(path)
                rows.append(row);features.append(feature)
                log.write(json.dumps(row)+'\n')
            if i%500==0:log.flush();print('camera_crops',i,'/',len(selected),dict(counts),flush=True)
    rows,features,conflicts,duplicate_count=deduplicate_crops(rows,features)
    counts['exact_same_label_duplicates_removed']=duplicate_count
    counts['conflicting_crop_hashes_excluded']=len(conflicts)
    counts['conflicting_annotations_excluded']=sum(len(c['annotations']) for c in conflicts)
    (crop_cache/'conflicting-labels-excluded.json').write_text(json.dumps(conflicts,indent=2),encoding='utf-8')
    parents=list(range(len(rows)))
    def find(x):
        while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    source_groups=defaultdict(list)
    for i,row in enumerate(rows):
        for source in row['source_aliases']:source_groups[source].append(i)
    for members in source_groups.values():
        for i in members[1:]:union(members[0],i)
    # Preserve the previous subset's grouped splits, even if crop geometry changed.
    prior_groups=defaultdict(list)
    for i,row in enumerate(rows):
        for source in row['source_aliases']:
            if source in fixed:prior_groups[fixed[source]['group']].append(i)
    for members in prior_groups.values():
        for i in members[1:]:union(members[0],i)
    valid=[i for i,f in enumerate(features) if f is not None]
    matrix=np.stack([features[i] for i in valid])
    # Chunked comparison caps memory; automatic similarity is split hygiene, not ground-truth identity.
    for start in range(0,len(valid),256):
        similarity=matrix[start:start+256]@matrix.T
        a,b=np.where(similarity>=.55)
        for x,y in zip(a+start,b):
            if x<y:union(valid[int(x)],valid[int(y)])
        if start%4096==0:print('identity_grouping',start,'/',len(valid),flush=True)
    groups=defaultdict(list)
    for i in range(len(rows)):groups[find(i)].append(i)
    for split in ('train','val','expw-val','expw-holdout','real-eyeglasses-val'):
        for label in CLASSES:(temp/split/label).mkdir(parents=True,exist_ok=True)
    base=ROOT/'data/seven-class/glasses-adaptation-v1'
    for split in ('train','val'):
        for label in CLASSES:
            for file in (base/split/label).glob('*.jpg'):
                os.link(file,temp/split/label/file.name);counts[f'base/{split}/{label}']+=1
    for members in groups.values():
        group_id=min(rows[i]['sha256'] for i in members)
        prior={fixed[source]['split'] for i in members for source in rows[i]['source_aliases'] if source in fixed}
        # Never move previously held-out sources into training, including via matched face groups.
        if 'expw-holdout' in prior:split='expw-holdout'
        elif 'expw-val' in prior:split='expw-val'
        elif 'train' in prior:split='train'
        else:
            bucket=int(group_id[:8],16)%20
            split='expw-holdout' if bucket<2 else 'expw-val' if bucket<4 else 'train'
            if any(features[i] is None for i in members):split='train'
        for i in members:
            row=rows[i];row.update(split=split,group=group_id,index=i)
            dest=temp/split/row['label']/f"web_expw_full_{row['sha256'][:24]}.jpg"
            # Cache is on D:, dataset on C:, so copying is intentional.
            dest.write_bytes(Path(row['cache_path']).read_bytes());row['path']=str(dest.relative_to(temp))
            counts[f'new/{split}/{row["label"]}']+=1
            if split=='expw-val' and row['real_eyewear']:
                os.link(dest,temp/'real-eyeglasses-val'/row['label']/dest.name)
    report={'source':'https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html','mirror':'https://www.kaggle.com/datasets/shahzadabbas/expression-in-the-wild-expw-dataset',
            'labels':'Original annotation class ids and face boxes, not pseudo-labels. No predicted expression score used to select samples.',
            'counts':dict(counts),'groups':len(groups),'largest_group':max(map(len,groups.values())),
            'preprocessing':'Browser-like <=512 edge/JPEG85, match YuNet detection to annotated face by IoU>=.25, same product square1.45 crop; annotation fallback flagged.',
            'deduplication':'Exact crop pixels with contradictory labels are excluded. Equal-label duplicates retain all source aliases for prior split and group preservation.',
            'split':'Source-image groups plus SFace>=.55 connected components; prior subset split preserved with holdout/validation priority. Automatic groups can merge/miss identities. Prior-dataset cross-source identity overlaps not exhaustively excluded.',
            'images':rows}
    (temp/'dataset.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    output.parent.mkdir(parents=True,exist_ok=True);temp.replace(output)
    print('PREPARE_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='images'}),flush=True)

if __name__=='__main__':main()
