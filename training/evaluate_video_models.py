"""Replay fixed video crops through candidate checkpoints; aggregate by entire clip."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
import timm
import torch
from emotion_preprocessing import emotion_tensor_path

ROOT=Path(__file__).resolve().parents[1]
CLASSES=('anger','disgust','fear','happiness','neutral','sadness','surprise')

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--zoom',type=float,default=1.0);p.add_argument('--cache',type=Path,default=ROOT/'data/video-validation-20260927');p.add_argument('--inference-precision',choices=('float32','bfloat16'),default='float32');args=p.parse_args()
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    cache=args.cache;manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    summary={'scope':manifest['scope'],'checkpoint_reports':{}}
    for checkpoint_path in args.checkpoint:
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if tuple(checkpoint['classes'])!=CLASSES:raise ValueError('Unverified checkpoint label order')
        grayscale=checkpoint.get('grayscale',True)
        model=timm.create_model(checkpoint['model_name'],pretrained=False,num_classes=7,**checkpoint.get('model_kwargs',{})).to(device).eval()
        model.load_state_dict(checkpoint['model_state'])
        videos=json.loads(json.dumps(manifest['videos']))
        references=[frame for clip in videos for frame in clip['frames'] if frame['status']=='detected']
        start=perf_counter()
        with torch.inference_mode():
            for offset in range(0,len(references),args.batch_size):
                batch=references[offset:offset+args.batch_size]
                tensor=np.stack([emotion_tensor_path(cache/frame['crop'],grayscale=grayscale,evaluation_zoom=args.zoom) for frame in batch])
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and args.inference_precision=='bfloat16'):
                    logits=model(torch.from_numpy(tensor).to(device))
                probabilities=torch.softmax(logits.float(),dim=-1).cpu().numpy()
                for frame,scores in zip(batch,probabilities):frame['probabilities']=scores.tolist()
        groups=defaultdict(list)
        for clip in videos:
            active=[frame for frame in clip['frames'] if clip['known_label_window'][0]<=frame['fraction']<=clip['known_label_window'][1]]
            detected=[frame for frame in active if 'probabilities' in frame]
            average=np.mean([frame['probabilities'] for frame in detected],axis=0) if detected else None
            label=CLASSES[int(np.argmax(average))] if average is not None else None
            clip['aggregate_prediction']=label
            clip['detection_rate']=len(detected)/max(1,len(active))
            groups[clip['source']].append(clip)
        metrics={}
        for domain,clips in groups.items():
            matrix=np.zeros((7,8),dtype=np.int64)
            for clip in clips:matrix[CLASSES.index(clip['label']),CLASSES.index(clip['aggregate_prediction']) if clip['aggregate_prediction'] else 7]+=1
            tp=np.diag(matrix[:,:7]);support=matrix.sum(axis=1);predicted=matrix[:,:7].sum(axis=0)
            precision=tp/np.maximum(1,predicted);recall=tp/np.maximum(1,support);f1=2*precision*recall/np.maximum(precision+recall,1e-10)
            metrics[domain]={'clips':len(clips),'accuracy_including_detection_misses':float(tp.sum()/len(clips)),
                             'macro_f1':float(f1.mean()),'mean_detection_rate':float(np.mean([c['detection_rate'] for c in clips])),
                             'per_class':{label:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,label in enumerate(CLASSES)},'confusion_with_no_face_column':matrix.tolist()}
        key=checkpoint_path.parent.name+'-'+checkpoint_path.stem+f'-zoom{args.zoom:g}'
        report={'model':str(checkpoint_path),'scope':manifest['scope'],'grayscale':grayscale,'zoom':args.zoom,
                'device':str(device),'inference_precision':args.inference_precision if device.type=='cuda' else 'float32',
                'target_sampling_fps':manifest.get('target_sampling_fps',4),'crop_format':manifest.get('crop_format','jpg'),
                'preprocessing':manifest['preprocessing']+('; lossless PNG crops' if manifest.get('crop_format')=='png' else '; cached crops re-encoded JPEG95 before classification'),
                'inference_batch_seconds':round(perf_counter()-start,2),'metrics':metrics,'videos':videos}
        raw=args.output/(key+'.json');raw.write_text(json.dumps(report,indent=2),encoding='utf-8')
        node=shutil.which('node')
        if not node:raise RuntimeError('Node is required to evaluate actual frontend smoothing')
        subprocess.run([node,str(ROOT/'scripts/evaluate_video_smoothing.cjs'),str(raw),str(args.output/(key+'-display.json'))],check=True,stdout=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        summary['checkpoint_reports'][key]=metrics
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps({'candidate':key,'metrics':metrics}),flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()

if __name__=='__main__':main()
