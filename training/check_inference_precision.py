"""Verify GPU FP32 comparison against exact exported CPU graphs on fixed real images."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch

from emotion_preprocessing import emotion_tensor_path


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('GPU is required for this sequential queue check')
    photos=[]
    for root in (Path('data/ferplus/validation-dedup-v1'),Path('data/expw/real-eyeglasses-original-expanded-val-v1')):
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            photos.extend(sorted((p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'}),
                                 key=lambda p:hashlib.sha256(p.name.encode()).hexdigest())[:10])
    inputs=np.stack([emotion_tensor_path(p,grayscale=True) for p in photos])
    runs=[('selective8','training/runs/vit-small-expw-selective-retention-v1/best_target.pt',
           'training/runs/vit-small-expw-selective-retention-v1/onnx/model-fp32.onnx')]
    reports=[]
    for name,checkpoint,graph in runs:
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=timm.create_model(data['model_name'],pretrained=False,num_classes=7,**data.get('model_kwargs',{}))
        model.load_state_dict(data['model_state']);model=model.cuda().eval()
        logits={}
        with torch.inference_mode():
            for precision in ('float32','bfloat16'):
                chunks=[]
                for start in range(0,len(inputs),32):
                    x=torch.from_numpy(inputs[start:start+32]).cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bfloat16'):
                        value=model(x)
                    chunks.append(value.float().cpu().numpy())
                logits[precision]=np.concatenate(chunks)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(graph,sess_options=options,providers=['CPUExecutionProvider'])
        cpu=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in inputs])
        result={'name':name,'checkpoint':checkpoint,'graph':graph,'graph_sha256':hashlib.sha256(Path(graph).read_bytes()).hexdigest(),
                'examples':len(inputs),'comparison':{}}
        for precision,value in logits.items():
            result['comparison'][precision]={'max_logit_error':float(np.abs(value-cpu).max()),
                'mean_logit_error':float(np.abs(value-cpu).mean()),'changed_top1':int(np.sum(value.argmax(1)!=cpu.argmax(1)))}
        reports.append(result)
        del model;torch.cuda.empty_cache()
    report={'scope':'Fixed SHAordered upto10photos perclass fromtwoexistingvalidationcohorts; nofinalholdout. Numerical verification only, not additional independentaccuracy estimate.',
            'photos':[str(p) for p in photos],'reports':reports}
    out=Path('training/overnight-20260927/gpu-fp32-onnx-photo-parity.json')
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(reports),flush=True)
    if any(r['comparison']['float32']['max_logit_error']>1e-4 or r['comparison']['float32']['changed_top1'] for r in reports):
        raise RuntimeError('GPUfloat32/CPUONNX parity exceeded tolerance; inspect before comparisons')


if __name__=='__main__':
    main()
