"""Reuse frozen train/validation frame selections, preserving source provenance."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server import EmotionRecognizer
from emotion_preprocessing import emotion_tensor_bgr
from screen_caer_pilot import camera_frame
from regional_temporal import RegionalVisual


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_server(old,expected):
    if sha(old)!=expected: raise ValueError('Archived source does not match capture provenance')
    def normalized(path):
        tree=ast.parse(Path(path).read_text(encoding='utf-8'))
        # Explicitly permit only the released model filename/display name changes.
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                if node.value in ('vit_emotion_seven_int8.onnx','vit_emotion_seven_fp32.onnx'):
                    node.value='MODEL_FILE'
                if node.value in ('Custom ViT-S/16 Seven-Class INT8','Custom ViT-S/16 Seven-Class FP32 (20260928)'):
                    node.value='MODEL_DISPLAY_NAME'
        return ast.dump(tree,include_attributes=False)
    before,after=normalized(old),normalized(ROOT/'server.py')
    if before!=after: raise ValueError('Product detector/crop or other logic changed beyond approved model names')
    return hashlib.sha256(after.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--split',choices=('train','validation'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--batch-size',type=int,default=40)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=json.loads(args.manifest.read_text(encoding='utf-8'))
    if source.get('partial_smoke_only'):raise ValueError('Incomplete source cache')
    archived=Path('training/overnight-20260927/before-final-release/server.py')
    expected=source.get('policy',{}).get('server_sha256','19ce3bb22807f5948b2d474c08728b566ec50c36222fa72a14a5fafde6132796')
    fingerprint=verify_server(archived,expected)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    base=timm.create_model(checkpoint['model_name'],pretrained=False,num_classes=7,**checkpoint.get('model_kwargs',{}))
    base.load_state_dict(checkpoint['model_state']);model=RegionalVisual(base).to(args.device).eval()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.graph),sess_options=options,providers=['CPUExecutionProvider'])
    policy={'source_manifest_sha256':sha(args.manifest),'checkpoint_sha256':sha(args.checkpoint),
            'graph_sha256':sha(args.graph),'capture_logic_ast_sha256':fingerprint,
            'current_server_sha256':sha(ROOT/'server.py'),'split':args.split,
            'precision':'FP32, TF32 disabled','regions':['CLS','top7patchrows','bottom7patchrows'],
            'feature_dim':384,'grayscale':checkpoint.get('grayscale',True)}
    args.output.mkdir(parents=True,exist_ok=True)
    rows=source['videos'][:args.limit] if args.limit else source['videos']
    result=[];parity=[];start=time.perf_counter()
    for vi,original in enumerate(rows,1):
        if args.split=='train' and original.get('split')!='train':raise ValueError('Nontrain video in training')
        if args.split=='validation' and original.get('source') not in ('RAVDESS-val','eNTERFACE-val','CAER-val'):
            raise ValueError('Unexpected validation source')
        dest=args.output/(original['id']+'.json')
        if dest.exists():
            cached=json.loads(dest.read_text(encoding='utf-8'))
            if cached['feature_policy']!=policy or sha(cached['feature_file'])!=cached['feature_sha256']:
                raise ValueError('Stale/corrupt feature cache')
            result.append(cached);continue
        video=deepcopy(original);temporary=None;capture=None
        selected=[f for f in video['frames'] if f.get('status')=='detected']
        features=[];logits_all=[];inputs=[];refs=[]

        def flush():
            if not inputs:return
            x=np.stack(inputs)
            with torch.inference_mode():
                logits,regional=model(torch.from_numpy(x).to(args.device))
                logits=logits.cpu().numpy();regional=regional.cpu().numpy()
            for i in range(min(len(x),40-len(parity))):
                deployed=session.run(None,{session.get_inputs()[0].name:x[i:i+1]})[0].reshape(7)
                error=float(np.abs(deployed-logits[i]).max())
                parity.append(error)
                if error>5e-4:raise ValueError('Regional visual logits disagree with deployed ONNX')
            probabilities=np.exp(logits-logits.max(1,keepdims=True));probabilities/=probabilities.sum(1,keepdims=True)
            offset=len(features)
            for i,(f,scores) in enumerate(zip(refs,probabilities)):
                f['probabilities']=scores.tolist();f['feature_index']=offset+i
            features.extend(regional);logits_all.extend(logits);inputs.clear();refs.clear()

        def add(frame,image):
            inputs.append(emotion_tensor_bgr(image,grayscale=policy['grayscale']))
            refs.append(frame)
            if len(inputs)==args.batch_size:flush()

        try:
            if args.split=='validation':
                for frame in selected:
                    image=cv2.imdecode(np.fromfile(frame['crop'],dtype=np.uint8),cv2.IMREAD_COLOR)
                    if image is None:raise ValueError('Missing validation crop')
                    add(frame,image)
            else:
                if 'archive' in video:
                    with ZipFile(video['archive']) as z:raw=z.read(video['entry'])
                    if hashlib.sha256(raw).hexdigest()!=video['source_sha256']:raise ValueError('Changed training video')
                    temporary=args.output/(video['id']+'.source.mp4');temporary.write_bytes(raw);path=temporary
                else:
                    path=Path(video['path'])
                    if sha(path)!=video['source_sha256']:raise ValueError('Changed training video')
                capture=cv2.VideoCapture(str(path))
                if int(capture.get(cv2.CAP_PROP_FRAME_COUNT))!=video['source_frames']:raise ValueError('Frame count differs')
                wanted={f['index']:f for f in selected};seen=set();index=0
                while capture.grab():
                    if index in wanted:
                        ok,raw=capture.retrieve()
                        if not ok:raise ValueError('Frame decoding failed')
                        image=camera_frame(raw,max_side=512);frame=wanted[index];h,w=image.shape[:2]
                        if (w,h)!=(frame['width'],frame['height']):raise ValueError('Capture size differs')
                        b=frame['bbox'];x,y,cw,ch=EmotionRecognizer._square_face_box(b['x'],b['y'],b['width'],b['height'],w,h)
                        add(frame,image[y:y+ch,x:x+cw]);seen.add(index)
                    index+=1
                if seen!=set(wanted):raise ValueError('Selected frames missing')
            flush()
        finally:
            if capture is not None:capture.release()
            if temporary is not None:temporary.unlink(missing_ok=True)
        arrays=args.output/(video['id']+'.npz')
        np.savez_compressed(arrays,features=np.asarray(features,dtype=np.float32).reshape(-1,3,384),
                            logits=np.asarray(logits_all,dtype=np.float32).reshape(-1,7))
        video.update(feature_file=str(arrays.resolve()),feature_sha256=sha(arrays),feature_policy=policy)
        dest.write_text(json.dumps(video,indent=2),encoding='utf-8');result.append(video)
        if vi%20==0 or vi==len(rows):
            print(json.dumps({'videos':vi,'total':len(rows),'elapsed_seconds':round(time.perf_counter()-start,1)}),flush=True)
    report={'policy':policy,'videos':result,'classes':checkpoint['classes'],'partial_smoke_only':bool(args.limit),
            'created_at':datetime.now(timezone.utc).isoformat(),'parity_logit_max_errors':parity,
            'scope':'Frozen regional visual features from fixed frames; subject-disjoint research selections, not new blind test.'}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPLETE',len(result),flush=True)


if __name__=='__main__':main()
