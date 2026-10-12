"""CPU latency and cache parity for conditional face verification on fixed videos."""
from collections import defaultdict
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from threading import Lock

import cv2
import numpy as np

from screen_caer_pilot import camera_frame, create_detector, detect_faces
import prepare_video_reverified as experiment
from server import EmotionRecognizer

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'training/overnight-20260927/caer-face-review'


def probes():
    annotations = json.loads((WORK / 'annotations-v1.json').read_text(encoding='utf-8'))
    cache = json.loads((ROOT / 'data/video-validation-20260927/manifest.json').read_text(encoding='utf-8'))
    videos = {v['id']: v for v in cache['videos']}
    wanted = defaultdict(list)
    for row in annotations['rows']:
        wanted[row['video_id']].append(row)
    result = []
    for video_id, rows in wanted.items():
        capture = cv2.VideoCapture(videos[video_id]['path'])
        by_index = {r['frame_index']: r for r in rows}
        try:
            for index in range(max(by_index) + 1):
                if not capture.grab():
                    raise RuntimeError(f'Cannot decode {video_id} at {index}')
                if index in by_index:
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise RuntimeError(f'Cannot retrieve {video_id} at {index}')
                    result.append((by_index[index], camera_frame(frame)))
        finally:
            capture.release()
    if len(result) != len(annotations['rows']):
        raise RuntimeError('Missing fixed probes')
    return result


def stats(values):
    return {key: float(value) for key, value in zip(
        ('mean_ms', 'p50_ms', 'p95_ms', 'max_ms'),
        (np.mean(values), np.percentile(values, 50), np.percentile(values, 95), np.max(values)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('competing-low', 'competing-larger'), default='competing-low')
    p.add_argument('--product', action='store_true')
    args = p.parse_args()
    if args.product and args.mode != 'competing-larger':
        raise ValueError('Product implementation uses competing-larger')
    cv2.setNumThreads(1)
    experiment.VERIFY_MODE = args.mode
    detector = create_detector()
    functions = {'original': detect_faces, 'conditional': experiment.verified_faces}
    if args.product:
        recognizer = EmotionRecognizer.__new__(EmotionRecognizer)
        recognizer.face_detector = detector
        recognizer.inference_lock = Lock()
        functions['product'] = lambda detector, image: recognizer._detect_faces(image)
    cache_name = 'competing' if args.mode == 'competing-low' else 'larger'
    expected = json.loads((ROOT / f'data/video-validation-{cache_name}-reverified-20260928/manifest.json').read_text(encoding='utf-8'))
    expected_frames = {(v['id'], f['index']): f for v in expected['videos'] for f in v['frames']}
    rows = []
    for probe, frame in probes():
        record = {'video_id': probe['video_id'], 'frame_index': probe['frame_index'],
                  'review_index': probe['review_index'], 'times_ms': defaultdict(list)}
        for repeat in range(3):
            # Alternate order; exclude each detector's first pass on this frame.
            order = list(functions) if repeat % 2 else list(reversed(functions))
            for name in order:
                started = perf_counter()
                faces = functions[name](detector, frame)
                elapsed = (perf_counter() - started) * 1000
                if repeat:
                    record['times_ms'][name].append(elapsed)
                record[name + '_faces'] = len(faces)
                if name in ('conditional', 'product'):
                    selected = max(faces, key=lambda f: f[2] * f[3]) if faces else None
                    box = dict(zip(('x', 'y', 'width', 'height'), selected[:4])) if selected else None
                    if box != expected_frames[probe['video_id'], probe['frame_index']].get('bbox'):
                        raise AssertionError(f'Conditional cache mismatch: {probe}')
        rows.append(record)
    cohorts = {}
    for cohort, members in {
        'all': rows,
        'original_single_face': [r for r in rows if r['original_faces'] == 1],
        'original_multiple_faces': [r for r in rows if r['original_faces'] > 1],
    }.items():
        cohorts[cohort] = {'frames': len(members), **{
            name: stats([t for r in members for t in r['times_ms'][name]]) for name in functions
        }}
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Actual CPU YuNet latency on all270 frozen CAER review frames, one OpenCV thread; concurrent GPU training may affect host speed. Detector only, not end-to-end camera latency.',
              'cache_parity': 'All270 selected boxes exactly match conditional cache.',
              'cohorts': cohorts, 'records': rows}
    report['mode'] = args.mode
    report['product_checked'] = args.product
    filename = 'product-detection-latency.json' if args.product else 'conditional-detection-latency.json'
    (WORK / filename).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
