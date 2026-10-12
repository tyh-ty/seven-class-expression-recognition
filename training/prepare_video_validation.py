"""Cache fixed videos through the webcam pipeline; the caller fixes split and sampling.

The default entrypoint contains validation videos only. Other callers must retain
their subject split and may request fixed fractional samples for training.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from screen_caer_pilot import create_detector,camera_frame,detect_faces
from server import EmotionRecognizer

OUT=ROOT/'data/video-validation-20260927'
SAMPLE_FPS=4.0
CROP_FORMAT='jpg'
CAMERA_MAX_SIDE=512
EMOTIONS={'01':'neutral','03':'happiness','04':'sadness','05':'anger','06':'fear','07':'disgust','08':'surprise'}

def process(record):
    directory=OUT/record['id'];directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'clip.json'
    fractions=record.get('sample_fractions')
    expected_rate=None if fractions is not None else SAMPLE_FPS
    if fractions is not None and (not fractions or any(not 0 <= f <= 1 for f in fractions)):
        raise ValueError('Fixed sample fractions must lie within[0,1]')
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        if saved.get('sampling_fps',4.0)!=expected_rate or saved.get('sample_fractions')!=fractions:
            raise ValueError('Cache already contains a different sampling rate; use a new output directory')
        if saved.get('crop_format','jpg')!=CROP_FORMAT:
            raise ValueError('Cache already contains a different crop encoding; use a new output directory')
        if saved.get('camera_max_side',512)!=CAMERA_MAX_SIDE:
            raise ValueError('Cache already contains a different capture size; use a new output directory')
        return saved
    detector=create_detector()
    temporary=None
    if 'archive' in record:
        temporary=directory/'source.mp4'
        with ZipFile(record['archive']) as z:temporary.write_bytes(z.read(record['entry']))
        path=temporary
    else:path=Path(record['path'])
    capture=cv2.VideoCapture(str(path))
    frames=[]
    try:
        fps=float(capture.get(cv2.CAP_PROP_FPS));count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps<=0 or count<=0:raise ValueError('Cannot decode validation clip')
        if fractions is not None:
            wanted={int(round(f*(count-1))) for f in fractions}
        elif SAMPLE_FPS==4.0:
            wanted=set(range(0,count,max(1,round(fps/4))))
        else:
            wanted={min(count-1,int(round(t*fps/SAMPLE_FPS))) for t in range(int(np.ceil(count/fps*SAMPLE_FPS)))}
        index=0
        while True:
            ok= capture.grab()
            if not ok:break
            if index not in wanted:
                index+=1;continue
            ok,original=capture.retrieve()
            if not ok:
                frames.append({'index':index,'status':'decode_failed','seconds':index/fps});index+=1;continue
            image=camera_frame(original,max_side=CAMERA_MAX_SIDE);height,width=image.shape[:2]
            faces=detect_faces(detector,image)
            row={'index':index,'seconds':round(index/fps,4),'fraction':index/max(1,count-1),
                 'width':width,'height':height,'face_count':len(faces),'status':'no_face'}
            if faces:
                x,y,w,h,score,landmarks=max(faces,key=lambda f:f[2]*f[3])
                left,top,cw,ch=EmotionRecognizer._square_face_box(x,y,w,h,width,height)
                crop=image[top:top+ch,left:left+cw]
                if crop.size:
                    crop=cv2.resize(crop,(224,224),interpolation=cv2.INTER_AREA if min(crop.shape[:2])>=224 else cv2.INTER_LINEAR)
                    dest=directory/f'frame-{index:06d}.{CROP_FORMAT}'
                    options=[cv2.IMWRITE_JPEG_QUALITY,95] if CROP_FORMAT=='jpg' else [cv2.IMWRITE_PNG_COMPRESSION,1]
                    success,encoded=cv2.imencode('.'+CROP_FORMAT,crop,options)
                    if not success:raise ValueError('Frame encode failure')
                    encoded.tofile(dest)
                    row.update(status='detected',crop=str(dest.relative_to(OUT)),bbox={'x':x,'y':y,'width':w,'height':h},face_score=score)
                    row['landmarks'] = ((np.asarray(landmarks).reshape(5,2)-np.asarray([left,top]))*np.asarray([224/cw,224/ch])).tolist()
            frames.append(row);index+=1
        result={**record,'fps':fps,'source_frames':count,'sampled_frames':len(frames),'frames':frames,
                'sampling_fps':expected_rate,
                'crop_format':CROP_FORMAT,
                'camera_max_side':CAMERA_MAX_SIDE,
                'scope':'Clip-level source labels; not independently annotated facial expression at each frame.'}
        manifest.write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    finally:
        capture.release()
        if temporary is not None:temporary.unlink(missing_ok=True)

def main():
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    jobs=[]
    for actor in range(17,21):
        archive=Path(f'D:/RAVDESS/Video_Speech_Actor_{actor:02d}.zip')
        with ZipFile(archive) as z:
            for entry in sorted(z.namelist()):
                name=Path(entry).name;parts=Path(name).stem.split('-')
                if len(parts)!=7 or parts[:2]!=['01','01'] or parts[2] not in EMOTIONS:continue
                jobs.append({'id':'ravdess-'+Path(name).stem,'source':'RAVDESS-val','label':EMOTIONS[parts[2]],'actor':actor,'archive':str(archive),'entry':entry,'intensity':parts[3],
                             'known_label_window':[.35,.75]})
    source=Path('D:/CAER-source/pilot-v1');manifest=json.loads((source/'manifest.json').read_text())
    for row in manifest['videos']:
        if row['split']!='val':continue
        jobs.append({'id':'caer-'+row['label']+'-'+Path(row['file']).stem,'source':'CAER-val','label':row['label'],'path':str(source/row['file']),
                     'source_sha256':row['sha256'],'known_label_window':[0,1]})
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(process,jobs),1):
            results.append(result)
            if i%10==0:print('validation_video_cache',i,'/',len(jobs),flush=True)
    report={'split':'RAVDESS actors17-20 and all90 original CAER pilot validation clips; no training clips used.',
            'scope':'RAVDESS acted speech central35-75% window plus full sequence; CAER scene label is only diagnostic and can differ from individual face. Not webcam ground truth.',
            'preprocessing':f'{SAMPLE_FPS:g}fps, longest side{CAMERA_MAX_SIDE}/JPEG85, YuNet and largest detected face, square1.45; classification is separate.',
            'target_sampling_fps':SAMPLE_FPS,
            'crop_format':CROP_FORMAT,
            'camera_max_side':CAMERA_MAX_SIDE,
            'counts':dict(Counter(r['source']+'/'+r['label'] for r in results)),'videos':results}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('VIDEO_CACHE_COMPLETE',len(results),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sample-fps',type=float,default=4.0)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--crop-format',choices=('jpg','png'),default='jpg')
    parser.add_argument('--camera-max-side',type=int,choices=(512,768),default=512)
    options=parser.parse_args()
    if not 0<options.sample_fps<=60:raise ValueError('Sampling rate must be within(0,60]')
    SAMPLE_FPS=options.sample_fps;OUT=options.output;CROP_FORMAT=options.crop_format;CAMERA_MAX_SIDE=options.camera_max_side
    main()
