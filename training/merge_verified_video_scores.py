"""Validate CPU/GPU parity, then assess fixed mixtures on identical validation frames."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import onnxruntime as ort

from emotion_preprocessing import emotion_tensor_path
from evaluate_frozen_system import confusion_report
from evaluate_video_onnx import cached_base_probabilities, CLASSES, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-report', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--base-report', type=Path, action='append', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-graph', type=Path, default=Path('models/vit_emotion_seven_int8.onnx'))
    p.add_argument('--base-source-index', type=int, choices=(0, 1), default=0)
    p.add_argument('--weights', type=float, nargs='+', default=[1., .5, .75])
    args = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    gpu = read(args.gpu_report)
    if gpu['inference_precision'] != 'float32' or gpu['grayscale'] or gpu['zoom'] != 1.0:
        raise ValueError('Expected pure FP32 RGB scores at original crop scale')
    export = read(args.graph.with_suffix('.json'))
    if Path(export['checkpoint']).resolve() != Path(gpu['model']).resolve() or export['grayscale']:
        raise ValueError('Graph and checkpoint preprocessing differ')
    manifest = read(args.cache/'manifest.json')
    base_path = args.base_graph
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if args.base_source_index == 0:
        base, provenance = cached_base_probabilities(args.base_report, digest(base_path), 'baseline-int8', manifest, args.cache)
    else:
        base, provenance = {}, []
        expected = {(v['id'],f['index']):f for v in manifest['videos'] for f in v['frames']}
        for path in args.base_report:
            cached = read(path)
            if cached['second_model_weight'] != 1 or cached['sources'][1]['sha256'] != digest(base_path) or not cached['sources'][1]['grayscale']:
                raise ValueError('Alternate base is not the specified raw grayscale CPU graph')
            if cached['target_sampling_fps'] != 10 or cached['crop_format'] != 'png':
                raise ValueError('Alternate base cadence or encoding differs')
            for v in cached['videos']:
                for f in v['frames']:
                    key = v['id'], f['index']
                    current = expected[key]
                    for field in ('seconds','fraction','width','height','bbox','status'):
                        if f.get(field) != current.get(field):
                            raise ValueError('Alternate base frame differs')
                    if f['status'] == 'detected':
                        if (Path(cached['cache'])/f['crop']).resolve() != (args.cache/current['crop']).resolve() or key in base:
                            raise ValueError('Alternate base crop changed or duplicated')
                        base[key] = np.asarray(f['probabilities'])
            provenance.append(dict(path=str(path), sha256=digest(path)))
        if len(base) != sum(f['status']=='detected' for f in expected.values()):
            raise ValueError('Alternate base does not cover all requested frames')
    new = {(v['id'], f['index']): (v, f) for v in gpu['videos'] for f in v['frames']}
    references = []
    for v in manifest['videos']:
        for f in v['frames']:
            ov, of = new[v['id'], f['index']]
            for field in ('seconds', 'fraction', 'width', 'height', 'bbox', 'status', 'crop'):
                if f.get(field) != of.get(field):
                    raise ValueError('Frame identity changed: '+field)
            if v['label'] != ov['label'] or v['known_label_window'] != ov['known_label_window']:
                raise ValueError('Label or window changed')
            if f['status'] == 'detected':
                references.append((v['id'], f))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.graph), sess_options=options, providers=['CPUExecutionProvider'])
    checks = []
    for index in np.linspace(0, len(references)-1, 128, dtype=int):
        video_id, frame = references[index]
        tensor = emotion_tensor_path(args.cache/frame['crop'], grayscale=False)[None]
        logits = session.run(None, {session.get_inputs()[0].name:tensor})[0].reshape(7).astype(np.float64)
        scores = np.exp(logits-logits.max()); scores /= scores.sum()
        prior = np.asarray(new[video_id, frame['index']][1]['probabilities'])
        error = float(np.max(np.abs(scores-prior)))
        checks.append(dict(video=video_id, index=frame['index'], max_probability_difference=error,
                           same_top1=bool(scores.argmax() == prior.argmax())))
    if max(r['max_probability_difference'] for r in checks) > 1e-4 or not all(r['same_top1'] for r in checks):
        raise ValueError('CPU graph parity failed; wait for full CPU evaluation')
    args.output.mkdir(parents=True, exist_ok=True)
    parity = dict(checks=checks, max_probability_difference=max(r['max_probability_difference'] for r in checks),
                  graph_sha256=digest(args.graph), checkpoint_sha256=digest(Path(gpu['model'])),
                  scope='128 evenly spaced exact validation frames. Remaining GPU scores are a numerically checked proxy, not measured CPU inference.')
    (args.output/'parity.json').write_text(json.dumps(parity, indent=2), encoding='utf-8')
    for weight in args.weights:
        if not 0 <= weight <= 1:
            raise ValueError('Invalid mixture weight')
        name = 'dinov2-best-fp32-proxy' if weight == 1 else f'ensemble-{weight:g}'
        videos = copy.deepcopy(manifest['videos'])
        groups = defaultdict(list)
        for v in videos:
            for f in v['frames']:
                if f['status'] == 'detected':
                    a = base[v['id'], f['index']]
                    b = np.asarray(new[v['id'], f['index']][1]['probabilities'])
                    f['probabilities'] = ((1-weight)*a+weight*b).tolist()
            detected = [f for f in v['frames'] if v['known_label_window'][0] <= f['fraction'] <= v['known_label_window'][1] and 'probabilities' in f]
            v['aggregate_prediction'] = CLASSES[int(np.mean([f['probabilities'] for f in detected], axis=0).argmax())] if detected else None
            groups[v['source']].append(v)
        measured = {}
        for domain, clips in groups.items():
            m = confusion_report([v['label'] for v in clips], [v['aggregate_prediction'] for v in clips])
            m.update(clips=len(clips), accuracy_including_detection_misses=m['accuracy'])
            measured[domain] = m
        report = dict(model=name, second_model_weight=weight, target_sampling_fps=10, crop_format='png',
                      cache=str(args.cache), videos=videos, metrics=measured,
                      sources=[dict(file=str(base_path), sha256=digest(base_path), grayscale=True),
                               dict(file=str(args.graph), sha256=digest(args.graph), grayscale=False)],
                      scope='Validation only. CPU baseline plus FP32 GPU new scores after 128-frame ONNX parity. Full CPU validation remains in progress.',
                      parity=parity, baseline_cache_reports=provenance)
        path = args.output/(name+'.json')
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        subprocess.run([shutil.which('node'), str(ROOT/'scripts/evaluate_video_smoothing.cjs'), str(path), str(path.with_stem(name+'-display'))], check=True, stdout=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        print(name, {d:round(v['accuracy'], 4) for d, v in measured.items()}, flush=True)


if __name__ == '__main__':
    main()
