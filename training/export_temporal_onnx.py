"""Export a bounded temporal research model to the existing lightweight CPU runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from temporal_residual import TemporalResidual


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['config']['smoke']:
        raise ValueError('Do not export a one-step smoke checkpoint as a trained candidate')
    model = TemporalResidual().eval()
    model.load_state_dict(checkpoint['state_dict'])
    rng = np.random.default_rng(20260928)
    example = rng.dirichlet(np.ones(7), size=(1, 8)).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.from_numpy(example), args.output, input_names=['probabilities'],
                      output_names=['logits'], opset_version=18, dynamo=False,
                      do_constant_folding=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    max_error = 0.
    max_static_error = 0.
    durations = []
    for i in range(128):
        values = rng.dirichlet(np.ones(7), size=(1, 8)).astype(np.float32)
        if i % 4 == 0:
            values[:] = values[:, -1:]
        with torch.inference_mode():
            reference = model(torch.from_numpy(values)).numpy()
        start = time.perf_counter()
        actual = session.run(None, {'probabilities': values})[0]
        durations.append((time.perf_counter()-start)*1000)
        max_error = max(max_error, float(np.abs(reference-actual).max()))
        if i % 4 == 0:
            probs = np.exp(actual-actual.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            max_static_error = max(max_static_error, float(np.abs(probs-values[:, -1]).max()))
    if max_error > 1e-4 or max_static_error > 1e-6:
        raise ValueError('Export changed temporal numerics/static invariance')
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'graph': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'base_model_sha256': checkpoint['config']['base_model_sha256'],
              'classes': checkpoint['config']['classes'], 'epoch': checkpoint['epoch'],
              'input': [1, 8, 7], 'output': [1, 7], 'window_seconds': .85,
              'gap_reset_seconds': .3, 'box_iou_reset_below': .3, 'probability_decimals': 6,
              'numerical_sequences': 128, 'max_logit_error': max_error,
              'max_static_probability_error': max_static_error,
              'latency_p50_ms': float(np.percentile(durations[3:], 50)),
              'latency_p95_ms': float(np.percentile(durations[3:], 95)),
              'scope': 'Conversion/numerical tests only, not deployment acceptance. Visual graph fingerprint must match. No training or validation accuracy inferred.'}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
