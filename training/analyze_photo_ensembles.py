"""Compare three fixed global averages of saved model probabilities, never labels."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from train_transformer import CLASSES, metrics_from_confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--other', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    normalize = lambda path: str(path).replace('\\', '/')
    reports = json.loads(args.report.read_text(encoding='utf-8'))['reports']
    base_key = next(k for k in reports if normalize(k) == normalize(args.base))
    other_key = next(k for k in reports if normalize(k) == normalize(args.other))
    base, other = reports[base_key], reports[other_key]
    assert base.keys() == other.keys()
    results = {base_key: base, other_key: other}
    for weight in (.25, .5, .75):
        candidate = {}
        for domain in base:
            first, second = base[domain]['records'], other[domain]['records']
            assert len(first) == len(second)
            matrix = torch.zeros((7, 7), dtype=torch.int64)
            records = []
            for a, b in zip(first, second):
                assert normalize(a['file']) == normalize(b['file']) and a['label'] == b['label']
                scores = (1-weight)*np.asarray(a['probabilities']) + weight*np.asarray(b['probabilities'])
                assert abs(scores.sum() - 1) < 1e-5 and np.isfinite(scores).all()
                prediction = int(scores.argmax())
                matrix[CLASSES.index(a['label']), prediction] += 1
                records.append(dict(a, predicted=CLASSES[prediction], probabilities=scores.tolist()))
            metric = asdict(metrics_from_confusion(matrix, 0.))
            candidate[domain] = {k: metric[k] for k in ('accuracy', 'macro_f1', 'per_class')}
            candidate[domain].update(examples=len(records), confusion=matrix.tolist(), records=records)
        results[f'global-ensemble-{weight:g}'] = candidate
    output = {'scope': 'Three global fixed probability mixtures, not class-specific boosts. Validation only; requires two actual models and final deployment latency/quantization checks.',
              'sources': [base_key, other_key], 'reports': results}
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({key: {domain: value['macro_f1'] for domain, value in candidate.items()
                           if domain in ('ferplus_clean','expw_original','real_eyeglasses_expanded')}
                     for key, candidate in results.items()}), flush=True)


if __name__ == '__main__':
    main()
