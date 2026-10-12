"""Export only the newly supervised head; never infer original logit labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from poster_research import build_features,SHA
from emotion_preprocessing import emotion_tensor_path
from dataset_contract import CLASSES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-smoke',action='store_true')
    p.add_argument('--quantize',action='store_true')
    args=p.parse_args();torch.set_num_threads(2)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ckpt['config']['smoke'] and not args.allow_smoke:raise ValueError('Smoke cannot be exported as a real trained candidate')
    if tuple(ckpt['config']['classes'])!=CLASSES or ckpt['config']['feature_weights_sha256']!=SHA:
        raise ValueError('Feature/class contract mismatch')
    data=json.loads(args.manifest.read_text(encoding='utf-8'))
    if data['smoke']!=ckpt['config']['smoke']:raise ValueError('Manifest/checkpoint purpose mismatch')
    model=build_features()
    head=nn.Linear(768,7)
    with torch.no_grad():
        head.weight.copy_(ckpt['head_state']['weight']/ckpt['std'][None])
        head.bias.copy_(ckpt['head_state']['bias']-(head.weight*ckpt['mean'][None]).sum(1))
    model.VIT.head=head;model.eval()
    # All parity/calibration input images are training images; no held-out tuning.
    rows=data['train']
    samples=[r for label in CLASSES for r in sorted((r for r in rows if r['label']==label),key=lambda r:r['sha256'])[:8]]
    tensors=[emotion_tensor_path(Path(r['file']),grayscale=False)[None] for r in samples]
    args.output.mkdir(parents=True,exist_ok=True)
    graph=args.output/'model-fp32.onnx'
    torch.onnx.export(model,torch.from_numpy(tensors[0]),str(graph),opset_version=17,
                       input_names=['pixel_values'],output_names=['logits'],dynamo=False)
    onnx.checker.check_model(str(graph))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
    differences=[];predictions=[];times=[]
    with torch.inference_mode():
        for i,x in enumerate(tensors[:14]):
            reference=model(torch.from_numpy(x)).numpy()
            start=time.perf_counter();actual=session.run(None,{'pixel_values':x})[0];times.append((time.perf_counter()-start)*1000)
            differences.append(float(np.max(np.abs(actual-reference))))
            predictions.append(bool(actual.argmax()==reference.argmax()))
    if max(differences)>5e-4 or not all(predictions):raise ValueError('ONNX/PyTorch mismatch')
    report={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'smoke_only':ckpt['config']['smoke'],'classes':CLASSES,'grayscale':False,'epoch':ckpt['epoch'],
            'original_head_reused':False,'preprocessing':data['preprocessing'],'format':'FP32',
            'sha256':hashlib.sha256(graph.read_bytes()).hexdigest(),'pytorch_onnx_max_logit_error':max(differences),
            'top1_parity_all':all(predictions),'training_images_measured':len(times),
            'warm_median_ms':float(np.median(times[2:])), 'warm_p95_ms':float(np.quantile(times[2:],.95)),
            'scope':'Training image format/timing check, not validation accuracy or physical webcam performance.'}
    if args.quantize:
        from onnxruntime.quantization import quantize_static,QuantFormat,QuantType,CalibrationDataReader
        class Reader(CalibrationDataReader):
            def __init__(self):self.items=iter([{'pixel_values':x} for x in tensors])
            def get_next(self):return next(self.items,None)
        target=args.output/'model-qdq-int8.onnx'
        quantize_static(str(graph),str(target),Reader(),quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,per_channel=True,
                        op_types_to_quantize=['Conv','MatMul','Gemm'],extra_options={'WeightSymmetric':True})
        quant=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
        qt=[];kl=[];same=[]
        softmax=lambda a:np.exp(a-a.max())/np.exp(a-a.max()).sum()
        for x in tensors:
            reference=session.run(None,{'pixel_values':x})[0]
            start=time.perf_counter();actual=quant.run(None,{'pixel_values':x})[0];qt.append((time.perf_counter()-start)*1000)
            a,b=softmax(reference).clip(1e-9),softmax(actual).clip(1e-9)
            kl.append(float((a*(np.log(a)-np.log(b))).sum()));same.append(bool(a.argmax()==b.argmax()))
        report['quantization']={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'calibration_train_images':len(tensors),
            'train_kl_mean':float(np.mean(kl)),'train_top1_agreement':float(np.mean(same)),
            'warm_median_ms':float(np.median(qt[2:])),'warm_p95_ms':float(np.quantile(qt[2:],.95)),
            'note':'Calibration/train parity is not held-out accuracy; actual CPU all-validation still required.'}
    (args.output/'export-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
