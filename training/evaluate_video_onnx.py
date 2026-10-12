"""Replay video crops through real batch-one CPU ONNX deployment graphs."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
import onnxruntime as ort

from emotion_preprocessing import emotion_tensor_path
from evaluate_onnx_ensemble import CLASSES, metrics

ROOT = Path(__file__).resolve().parents[1]


def cached_base_probabilities(paths, base_hash, base_name, manifest, cache):
    records = {}
    provenance = []
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['sources'][0]['sha256'] != base_hash or data['model'] != base_name or data['second_model_weight'] != 0:
            raise ValueError('Cached baseline graph/model selection does not match')
        if not data['sources'][0].get('grayscale', data.get('grayscale', True)):
            raise ValueError('Cached baseline must use deployed grayscale preprocessing')
        if data.get('target_sampling_fps') != manifest.get('target_sampling_fps') or data.get('crop_format') != manifest.get('crop_format'):
            raise ValueError('Cached baseline frame cadence/encoding does not match')
        for video in data['videos']:
            for frame in video['frames']:
                key = (video['id'], frame['index'])
                if key in records:
                    raise ValueError('Duplicate cached baseline frame')
                records[key] = (frame, Path(data['cache']))
        provenance.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    selected = {}
    for video in manifest['videos']:
        for frame in video['frames']:
            key = (video['id'], frame['index'])
            if key not in records:
                raise ValueError('Baseline cache missing a requested exact frame')
            old, old_cache = records[key]
            for field in ('status', 'seconds', 'fraction', 'width', 'height', 'bbox', 'face_count'):
                if frame.get(field) != old.get(field):
                    raise ValueError(f'Cached baseline frame metadata differs: {key}, {field}')
            if frame['status'] == 'detected':
                if (cache / frame['crop']).resolve() != (old_cache / old['crop']).resolve():
                    raise ValueError('Cached baseline must reference the identical preserved crop file')
                probabilities = np.asarray(old['probabilities'], dtype=np.float64)
                if probabilities.shape != (7,) or not np.isfinite(probabilities).all():
                    raise ValueError('Invalid cached baseline probabilities')
                selected[key] = probabilities
    return selected, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--other', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--other-rgb', action='store_true', help='Use RGB normalization for the second graph only.')
    p.add_argument('--wait-cache-seconds', type=int, default=0)
    p.add_argument('--base-name', default='baseline-int8')
    p.add_argument('--other-name', default='expw-int8')
    p.add_argument('--skip-mixtures', action='store_true')
    p.add_argument('--base-report', type=Path, action='append', default=[],
                   help='Reuse exact prior baseline probabilities after graph hash, frame metadata, cadence and crop-path verification.')
    args = p.parse_args()
    if args.base_name == args.other_name or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', name)
                                               for name in (args.base_name, args.other_name)):
        raise ValueError('Model names must be distinct safe filename components')
    deadline = time.monotonic() + args.wait_cache_seconds
    manifest_path = args.cache / 'manifest.json'
    while not manifest_path.is_file() and time.monotonic() < deadline:
        time.sleep(5)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    paths = [args.base, args.other]
    sources = [{'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    for source, grayscale in zip(sources, (True, not args.other_rgb)):
        source['grayscale'] = grayscale
        metadata_path = Path(source['file']).with_suffix('.json')
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if 'grayscale' in metadata and metadata['grayscale'] != grayscale:
                raise ValueError('Requested color preprocessing differs from export metadata')
    cached, cache_provenance = cached_base_probabilities(args.base_report, sources[0]['sha256'], args.base_name, manifest, args.cache) if args.base_report else ({}, [])
    sessions = [ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider']) for path in (paths[1:] if args.base_report else paths)]
    for session in sessions:
        assert session.get_inputs()[0].shape[1:] == [3, 224, 224]
        assert session.get_outputs()[0].shape[-1] == 7
    scores_by_frame = {}
    timings = []
    started = time.perf_counter()
    for video in manifest['videos']:
        for frame in video['frames']:
            if frame['status'] != 'detected':
                continue
            tensor = emotion_tensor_path(args.cache / frame['crop'], grayscale=True)[None]
            other_tensor = emotion_tensor_path(args.cache / frame['crop'], grayscale=False)[None] if args.other_rgb else tensor
            probabilities = [cached[video['id'], frame['index']]] if args.base_report else []
            inference_start = time.perf_counter()
            for session_index, session in enumerate(sessions):
                input_tensor = other_tensor if args.base_report or session_index == 1 else tensor
                logits = session.run(None, {session.get_inputs()[0].name: input_tensor})[0].reshape(7)
                scores = np.exp(logits.astype(np.float64) - logits.max())
                scores /= scores.sum()
                assert np.isfinite(scores).all()
                probabilities.append(scores)
            timings.append((time.perf_counter()-inference_start)*1000)
            scores_by_frame[video['id'], frame['index']] = probabilities
        if len(scores_by_frame) // 500 > (len(scores_by_frame)-video['sampled_frames']) // 500:
            print(json.dumps({'frames': len(scores_by_frame), 'elapsed_seconds': round(time.perf_counter()-started, 1)}), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    variants = [(args.base_name, 0.), (args.other_name, 1.)]
    if not args.skip_mixtures:
        variants.extend([('ensemble-0.5', .5), ('ensemble-0.75', .75)])
    for name, second_weight in variants:
        videos = copy.deepcopy(manifest['videos'])
        domains = defaultdict(list)
        for video in videos:
            for frame in video['frames']:
                if frame['status'] == 'detected':
                    first, second = scores_by_frame[video['id'], frame['index']]
                    frame['probabilities'] = ((1-second_weight)*first + second_weight*second).tolist()
            active = [f for f in video['frames'] if video['known_label_window'][0] <= f['fraction'] <= video['known_label_window'][1]]
            detected = [f for f in active if 'probabilities' in f]
            video['aggregate_prediction'] = CLASSES[int(np.mean([f['probabilities'] for f in detected], axis=0).argmax())] if detected else None
            video['detection_rate'] = len(detected)/max(1, len(active))
            domains[video['source']].append(video)
        measured = {}
        for domain, clips in domains.items():
            matrix = np.zeros((7, 8), dtype=np.int64)
            for video in clips:
                predicted = CLASSES.index(video['aggregate_prediction']) if video['aggregate_prediction'] is not None else 7
                matrix[CLASSES.index(video['label']), predicted] += 1
            metric = metrics(matrix[:, :7])
            tp = matrix[:, :7].diagonal()
            support = matrix.sum(1)
            precision = tp / np.maximum(1, matrix[:, :7].sum(0))
            recall = tp / np.maximum(1, support)
            f1 = 2 * precision * recall / np.maximum(1e-12, precision+recall)
            metric['per_class'] = {label: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                           'f1': float(f1[i]), 'support': int(support[i])}
                                   for i, label in enumerate(CLASSES)}
            metric['macro_f1'] = float(f1.mean())
            metric['confusion_with_no_face_column'] = matrix.tolist()
            metric['examples'] = len(clips)
            metric['clips'] = len(clips)
            metric['no_face_clips'] = int(matrix[:, 7].sum())
            metric['accuracy_including_detection_misses'] = float(np.trace(matrix)/max(1,len(clips)))
            metric['accuracy'] = metric['accuracy_including_detection_misses']
            measured[domain] = metric
        result = {'model': name, 'sources': sources, 'second_model_weight': second_weight,
                  'scope': manifest['scope'], 'cache': str(args.cache),
                  'grayscale': True if second_weight == 0 else (not args.other_rgb if second_weight == 1 else (True if not args.other_rgb else None)),
                  'color_preprocessing': [s['grayscale'] for s in sources],
                  'target_sampling_fps': manifest.get('target_sampling_fps', 4), 'crop_format': manifest.get('crop_format', 'jpg'),
                  'preprocessing': manifest['preprocessing'],
                  'runtime': 'Actual ONNX CPU batch1. Nominal camera sampling cadence; transport/service backlog is not simulated.',
                  'classification_graphs_measured_per_frame': len(sessions),
                  'baseline_cache_reports': cache_provenance,
                  ('other_model_classification_latency' if args.base_report else 'two_model_classification_latency'): {'mean_ms': float(np.mean(timings[2:])),
                    'p50_ms': float(np.percentile(timings[2:],50)), 'p95_ms': float(np.percentile(timings[2:],95))},
                  'metrics': measured, 'videos': videos}
        path = args.output / (name+'.json')
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        subprocess.run([shutil.which('node'), str(ROOT/'scripts/evaluate_video_smoothing.cjs'), str(path), str(path.with_stem(name+'-display'))],
                       check=True, stdout=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        summary[name] = measured
        print(json.dumps({'model': name, 'metrics': measured}), flush=True)
    (args.output/'summary.json').write_text(json.dumps({'scope': 'Batch1 CPU ONNX video validation; consult source hashes for exact graph versions. Not independent natural webcam ground truth.', 'reports': summary}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
