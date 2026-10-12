"""Replay the frozen validation frames through two actual local CPU services."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from urllib.request import ProxyHandler, Request, build_opener

import cv2

from benchmark_face_reverification import probes, stats

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'training/overnight-20260927/caer-face-review'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-port', type=int, default=8765)
    p.add_argument('--candidate-port', type=int, default=8790)
    args = p.parse_args()
    opener = build_opener(ProxyHandler({}))
    ports = {'baseline': args.baseline_port, 'candidate': args.candidate_port}
    caches = {}
    for name, cache in {'baseline': 'video-validation-20260927',
                         'candidate': 'video-validation-larger-reverified-20260928'}.items():
        data = json.loads((ROOT / 'data' / cache / 'manifest.json').read_text())
        caches[name] = {(v['id'], f['index']): f for v in data['videos'] for f in v['frames']}
    health = {}
    for name, port in ports.items():
        with opener.open(f'http://127.0.0.1:{port}/api/health', timeout=5) as response:
            health[name] = json.load(response)
        assert health[name]['ready']
    records = []
    for index, (annotation, frame) in enumerate(probes()):
        ok, encoded = cv2.imencode('.png', frame)
        assert ok
        payload = encoded.tobytes()
        record = {'review_index': annotation['review_index'], 'video': annotation['video_id'],
                  'frame_index': annotation['frame_index']}
        order = list(ports) if index % 2 else list(reversed(ports))
        for name in order:
            request = Request(f'http://127.0.0.1:{ports[name]}/api/analyze?model=transformer',
                              data=payload, headers={'Content-Type': 'image/png'})
            started = perf_counter()
            with opener.open(request, timeout=10) as response:
                result = json.load(response)
            elapsed = (perf_counter() - started) * 1000
            selected = result['faces'][0] if result['faces'] else None
            expected = caches[name][annotation['video_id'], annotation['frame_index']].get('bbox')
            if (selected['bbox'] if selected else None) != expected:
                raise AssertionError(f'{name} deployed API/cache mismatch: {annotation["review_index"]}')
            record[name] = {'http_ms': elapsed, 'latency_ms': result['latencyMs'],
                            'timing_ms': result['timingMs'], 'faces': len(result['faces']),
                            'primary': selected}
        records.append(record)
    summary = {name: {'http': stats([r[name]['http_ms'] for r in records][2:]),
                       'service': stats([r[name]['latency_ms'] for r in records][2:]),
                       'single_face_service': stats([r[name]['latency_ms'] for r in records[2:] if r[name]['faces'] == 1])}
               for name in ports}
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Actual deployed Python/ONNX CPU service on270 frozen video frames. Lossless PNG transports already512/JPEG85-emulated frames, so no second JPEG compression. This measures service/HTTP response, not capture intervals or UI smoothing delay. Concurrent GPU training; exclude first two frames for timing warmup.',
              'health': health, 'cache_parity': 'Both original and revised actual APIs match selected face boxes on all270 frames.',
              'summary': summary, 'records': records}
    (WORK / 'product-api-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'cache_parity': report['cache_parity'], 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
