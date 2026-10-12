"""Rescore preserved train-only frame/box selections with a verified visual checkpoint."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import EmotionRecognizer
from screen_caer_pilot import camera_frame
from emotion_preprocessing import emotion_tensor_bgr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=Path('data/temporal-training-int8-v1/manifest.json'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--batch-size', type=int, default=40)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    if source.get('partial_smoke_only'):
        raise ValueError('Source training cache incomplete')
    if source['policy']['server_sha256'] != hashlib.sha256((ROOT/'server.py').read_bytes()).hexdigest():
        raise ValueError('Detector/crop source changed')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert tuple(checkpoint['classes']) == tuple(source['classes'])
    model = timm.create_model(checkpoint['model_name'], pretrained=False, num_classes=7,
                              **checkpoint.get('model_kwargs', {})).to(args.device).eval()
    model.load_state_dict(checkpoint['model_state'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.graph), sess_options=options, providers=['CPUExecutionProvider'])
    policy = {**source['policy'], 'model_sha256': hashlib.sha256(args.graph.read_bytes()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'rescore_precision': 'float32, matmul/cuDNN TF32 disabled',
              'original_selection_manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    records = source['videos'][:args.limit] if args.limit else source['videos']
    results = []
    parity = []
    started = time.perf_counter()
    for vi, original in enumerate(records, 1):
        if original['split'] != 'train':
            raise ValueError('Nontraining video')
        dest = args.output/(original['id']+'.json')
        if dest.exists():
            old = json.loads(dest.read_text())
            if old['policy'] != policy:
                raise ValueError('Existing rescore has different graph/source')
            results.append(old)
            continue
        video = deepcopy(original)
        temporary = None
        if 'archive' in video:
            with ZipFile(video['archive']) as archive:
                raw = archive.read(video['entry'])
            if hashlib.sha256(raw).hexdigest() != video['source_sha256']:
                raise ValueError('Source video bytes changed')
            temporary = args.output/(video['id']+'.source.mp4')
            temporary.write_bytes(raw)
            path = temporary
        else:
            path = Path(video['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != video['source_sha256']:
                raise ValueError('Source video bytes changed')
        capture = cv2.VideoCapture(str(path))
        wanted = {f['index']: f for f in video['frames'] if f['status'] == 'detected'}
        tensors, refs = [], []
        seen = set()

        def flush():
            if not tensors:
                return
            x = np.stack(tensors)
            with torch.inference_mode():
                logits = model(torch.from_numpy(x).to(args.device)).cpu().numpy()
            # Verify actual deployment graph agreement on the first 40 train frames.
            for i in range(min(len(x), 40-len(parity))):
                deployed = session.run(None, {session.get_inputs()[0].name: x[i:i+1]})[0].reshape(7)
                error = float(np.abs(deployed-logits[i]).max())
                equal = int(deployed.argmax()) == int(logits[i].argmax())
                parity.append({'max_logit_error': error, 'top1_equal': equal})
                if error > 5e-4 or not equal:
                    raise ValueError('Research scorer and actual CPU graph disagree')
            scores = np.exp(logits-logits.max(axis=1, keepdims=True))
            scores /= scores.sum(axis=1, keepdims=True)
            for frame, values in zip(refs, scores):
                frame['probabilities'] = [round(float(v), 6) for v in values]
            tensors.clear()
            refs.clear()

        try:
            if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != video['source_frames']:
                raise ValueError('Source frame count changed')
            index = 0
            while capture.grab():
                if index in wanted:
                    ok, raw_frame = capture.retrieve()
                    if not ok:
                        raise ValueError('Video decode failed')
                    image = camera_frame(raw_frame, max_side=512)
                    frame = wanted[index]
                    h, w = image.shape[:2]
                    if (w, h) != (frame['width'], frame['height']):
                        raise ValueError('Capture dimensions changed')
                    b = frame['bbox']
                    x, y, width, height = EmotionRecognizer._square_face_box(b['x'], b['y'], b['width'], b['height'], w, h)
                    crop = image[y:y+height, x:x+width]
                    tensors.append(emotion_tensor_bgr(crop, grayscale=checkpoint.get('grayscale', True)))
                    refs.append(frame)
                    seen.add(index)
                    if len(tensors) == args.batch_size:
                        flush()
                index += 1
            flush()
            if seen != set(wanted):
                raise ValueError('Missing selected training frames')
        finally:
            capture.release()
            if temporary:
                temporary.unlink(missing_ok=True)
        video['policy'] = policy
        dest.write_text(json.dumps(video, indent=2), encoding='utf-8')
        results.append(video)
        if vi % 20 == 0 or vi == len(records):
            print(json.dumps({'videos': vi, 'total': len(records), 'seconds': round(time.perf_counter()-started, 1)}), flush=True)
    result = {**source, 'policy': policy, 'videos': results,
              'created_at': datetime.now(timezone.utc).isoformat(), 'partial_smoke_only': args.limit is not None,
              'rescore_scope': 'Exact prior train-only frame selections and boxes, source SHA verified. Batched FP32 logits checked against actual CPU graph on 40 train frames. No validation/holdout training.',
              'parity': parity}
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()
