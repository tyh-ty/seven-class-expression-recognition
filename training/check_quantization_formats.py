"""Choose a faster graph format by numerical fidelity on training images only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from emotion_preprocessing import emotion_tensor_path

CLASSES = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.input.with_suffix('.json').read_text(encoding='utf-8'))
    if not metadata.get('grayscale', True):
        raise ValueError('This bounded format experiment specifies a grayscale visual model')
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    paths = []
    for label in CLASSES:
        choices = list((args.dataset/'train'/label).glob('*.jpg'))
        paths += sorted(choices, key=lambda path: hashlib.sha256(path.name.encode()).hexdigest())[:20]
    if len(paths) != 140:
        raise ValueError('Expected twenty training images per class')
    tensors = [emotion_tensor_path(path, grayscale=True)[None] for path in paths]
    reference = ort.InferenceSession(str(args.input), sess_options=options, providers=['CPUExecutionProvider'])
    expected = np.concatenate([reference.run(None, {reference.get_inputs()[0].name: x})[0] for x in tensors])
    expected_probs = np.exp(expected-expected.max(axis=1, keepdims=True))
    expected_probs /= expected_probs.sum(axis=1, keepdims=True)
    variants = [('per-tensor-int8', False, False), ('per-channel-int8', True, False),
                ('per-channel-int7', True, True)]
    results = []
    for name, per_channel, reduce_range in variants:
        path = args.output/(name+'.onnx')
        quantize_dynamic(str(args.input), str(path), weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul', 'Gemm'], per_channel=per_channel,
                         reduce_range=reduce_range)
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        measured, timings = [], []
        for tensor in tensors:
            start = time.perf_counter()
            measured.append(session.run(None, {session.get_inputs()[0].name: tensor})[0])
            timings.append((time.perf_counter()-start)*1000)
        measured = np.concatenate(measured)
        probs = np.exp(measured-measured.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        result = {'name': name, 'graph': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'per_channel': per_channel, 'reduce_range': reduce_range,
                  'top1_matches': int(np.sum(expected.argmax(1) == measured.argmax(1))),
                  'images': len(paths), 'mean_logit_error': float(np.abs(expected-measured).mean()),
                  'max_logit_error': float(np.abs(expected-measured).max()),
                  'mean_probability_kl': float(np.mean(np.sum(expected_probs*(np.log(expected_probs.clip(1e-9))-np.log(probs.clip(1e-9))), axis=1))),
                  'p50_ms': float(np.percentile(timings[3:], 50)), 'p95_ms': float(np.percentile(timings[3:], 95))}
        results.append(result)
        source_verification_keys = ('max_absolute_difference', 'mean_absolute_difference', 'same_top_class')
        quantized_metadata = {key: value for key, value in metadata.items() if key not in source_verification_keys}
        quantized_metadata.update(onnx=str(path), quantization=result,
            reference_fp32_export_verification={key: metadata[key] for key in source_verification_keys if key in metadata},
            scope='Format fidelity on 140 training samples, not expression accuracy.')
        path.with_suffix('.json').write_text(json.dumps(quantized_metadata, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    selected = min(results, key=lambda row: (row['mean_probability_kl'], -row['top1_matches']))
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'reference': str(args.input),
              'reference_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'selection': 'Lowest probability KL to FP32 on fixed training crops; evaluate selected graph with unchanged photo/video gates before use.',
              'scope': 'No validation or holdout labels used. Numerical agreement is not classification accuracy. Timing under other research load.',
              'samples': [str(path) for path in paths], 'variants': results, 'selected': selected}
    (args.output/'format-selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
