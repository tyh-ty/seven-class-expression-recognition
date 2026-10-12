"""Actual batch-one CPU validation of two ONNX graphs and fixed probability means."""
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

CLASSES = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')


def metrics(matrix):
    tp = np.diag(matrix)
    support = matrix.sum(1)
    precision = tp / np.maximum(1, matrix.sum(0))
    recall = tp / np.maximum(1, support)
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    return {'examples': int(matrix.sum()), 'accuracy': round(float(tp.sum()/max(1, matrix.sum())), 6),
            'macro_f1': round(float(f1.mean()), 6), 'confusion': matrix.tolist(),
            'per_class': {label: {'precision': round(float(precision[i]), 6),
                'recall': round(float(recall[i]), 6), 'f1': round(float(f1[i]), 6), 'support': int(support[i])}
                          for i, label in enumerate(CLASSES)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--other', type=Path, required=True)
    p.add_argument('--dataset', nargs=2, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--other-rgb', action='store_true', help='Use RGB normalization for the second graph only.')
    args = p.parse_args()
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sessions = [ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                for path in (args.base, args.other)]
    for session in sessions:
        assert session.get_inputs()[0].shape[1:] == [3, 224, 224]
        assert session.get_outputs()[0].shape[-1] == 7
    names = [str(args.base), str(args.other), 'onnx-ensemble-0.5', 'onnx-ensemble-0.75']
    reports = {name: {} for name in names}
    cache = {}
    elapsed = defaultdict(list)
    started = perf_counter()
    sources = [{'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in (args.base, args.other)]
    for source, grayscale in zip(sources, (True, not args.other_rgb)):
        source['grayscale'] = grayscale
        metadata_path = Path(source['file']).with_suffix('.json')
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if 'grayscale' in metadata and metadata['grayscale'] != grayscale:
                raise ValueError('Requested color preprocessing differs from export metadata')
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
                    other_tensor = emotion_tensor_bgr(bgr, grayscale=False, add_batch_dimension=True) if args.other_rgb else tensor
                    probabilities = []
                    total_start = perf_counter()
                    for index, session in enumerate(sessions):
                        model_start = perf_counter()
                        logits = session.run(None, {session.get_inputs()[0].name: tensor if index == 0 else other_tensor})[0].reshape(7)
                        elapsed[names[index]].append((perf_counter()-model_start)*1000)
                        scores = np.exp(logits.astype(np.float64)-logits.max())
                        scores /= scores.sum()
                        assert np.isfinite(scores).all()
                        probabilities.append(scores)
                    elapsed['two_models_total'].append((perf_counter()-total_start)*1000)
                    cache[digest] = probabilities
                    if len(cache) % 500 == 0:
                        print(json.dumps({'unique_images': len(cache), 'elapsed_seconds': round(perf_counter()-started, 1)}), flush=True)
                first, second = cache[digest]
                choices = [first, second, (first+second)/2, first*.25+second*.75]
                for name, scores in zip(names, choices):
                    prediction = int(scores.argmax())
                    matrices[name][label_index, prediction] += 1
                    records[name].append({'file': str(path), 'label': label, 'predicted': CLASSES[prediction],
                                          'probabilities': scores.tolist()})
        for name in names:
            reports[name][domain] = dict(metrics(matrices[name]), records=records[name])
        timing = {name: {'observations': len(values), 'mean_ms': float(np.mean(values[2:])),
                         'p50_ms': float(np.percentile(values[2:], 50)), 'p95_ms': float(np.percentile(values[2:], 95))}
                  for name, values in elapsed.items()}
        output = {'scope': 'CPU batch1 actual ONNX graphs identified by file, SHA256 and explicit per-graph color normalization; fixed global mixture weights .5/.75. Exact byte duplicates computed once but all domain records retained. No holdout data.',
                  'sources': sources, 'baseline_key': names[0], 'unique_images': len(cache),
                  'classification_latency': timing, 'timing_scope': 'Warmed actual single-face serial inference, excludes detection/capture/display. Concurrent GPU training.',
                  'reports': reports}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
        print(json.dumps({'domain': domain, 'macro_f1': {name: reports[name][domain]['macro_f1'] for name in names}}), flush=True)


if __name__ == '__main__':
    main()
