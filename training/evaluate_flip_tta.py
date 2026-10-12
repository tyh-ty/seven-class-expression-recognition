"""Fixed horizontal-mirror inference experiment on validation photos, CPU ONNX only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from emotion_preprocessing import emotion_tensor_bgr
from evaluate_onnx_ensemble import CLASSES, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--dataset', nargs=2, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=3)
    args = p.parse_args()
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape[1:] == [3, 224, 224]
    assert session.get_outputs()[0].shape[-1] == 7
    input_name = session.get_inputs()[0].name
    names = ['original-int8', 'mirror-int8', 'mirror-average-int8']
    reports = {name: {} for name in names}
    cache, timing = {}, defaultdict(list)
    started = perf_counter()
    for domain, directory in args.dataset:
        directory = Path(directory)
        matrices = {name: np.zeros((7, 7), dtype=np.int64) for name in names}
        records = {name: [] for name in names}
        for label_index, label in enumerate(CLASSES):
            for path in sorted((directory / label).glob('*')):
                if path.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
                    continue
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest not in cache:
                    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError(f'Cannot decode {path}')
                    tensor = emotion_tensor_bgr(bgr, grayscale=True, add_batch_dimension=True)
                    scores = []
                    total_start = perf_counter()
                    for name, value in zip(names[:2], [tensor, np.ascontiguousarray(tensor[:, :, :, ::-1])]):
                        inference_start = perf_counter()
                        logits = session.run(None, {input_name: value})[0].reshape(7).astype(np.float64)
                        timing[name].append((perf_counter()-inference_start)*1000)
                        probs = np.exp(logits-logits.max())
                        probs /= probs.sum()
                        assert np.isfinite(probs).all()
                        scores.append(probs)
                    timing[names[2]].append((perf_counter()-total_start)*1000)
                    cache[digest] = scores
                    if len(cache) % 500 == 0:
                        print(json.dumps({'unique_images':len(cache),'seconds':round(perf_counter()-started,1)}),flush=True)
                original, mirror = cache[digest]
                for name, probs in zip(names, [original, mirror, (original+mirror)/2]):
                    predicted = int(probs.argmax())
                    matrices[name][label_index, predicted] += 1
                    records[name].append({'file':str(path),'label':label,'predicted':CLASSES[predicted],'probabilities':probs.tolist()})
        for name in names:
            reports[name][domain] = dict(metrics(matrices[name]),records=records[name])
            assert reports[name][domain]['examples'] > 0, f'Empty domain {domain}'
        output = {
            'scope':'Fixed same-model horizontal flip and equal probability mean. Actual CPU batch1 INT8, unchanged grayscale preprocessing/class order. Research only, no deployment or holdout; validation reused for model selection.',
            'source':{'path':str(args.model),'sha256':hashlib.sha256(args.model.read_bytes()).hexdigest()},
            'baseline_key':names[0], 'unique_images':len(cache),
            'classification_latency':{name:{'n':len(values),'median_ms':float(np.median(values[2:])),
                                           'p95_ms':float(np.percentile(values[2:],95))} for name,values in timing.items()},
            'reports':reports,
        }
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
        print(json.dumps({'domain':domain,'f1':{name:reports[name][domain]['macro_f1'] for name in names}}),flush=True)


if __name__ == '__main__':
    main()
