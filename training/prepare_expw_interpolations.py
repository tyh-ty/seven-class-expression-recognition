"""Fixed coarse weight interpolation after an adaptation run; no test-set access."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    history = json.loads((args.run / 'history.json').read_text(encoding='utf-8'))
    chosen = max(history, key=lambda row: (row['target_validation']['metrics']['macro_f1'],
                                           row['validation']['macro_f1'], -row['epoch']))
    source = args.run / f"epoch-{chosen['epoch']:03d}.pt"
    output = args.run / 'interpolated'
    output.mkdir(exist_ok=False)
    raw = output / 'target-validation-best.pt'
    shutil.copyfile(source, raw)
    for alpha in (.25, .5, .75):
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('interpolate_checkpoints.py')),
            '--base', str(args.base), '--adapted', str(raw), '--alpha', str(alpha),
            '--output', str(output / f'alpha-{alpha:g}.pt')], check=True)
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
        'selection': 'Maximum original ExpW validation macro F1, then primary validation F1, then earliest epoch. This is an unconstrained research candidate, not an approved replacement.',
        'source_epoch': chosen['epoch'], 'source_checkpoint': str(source),
        'fixed_alphas': [.25, .5, .75], 'holdout_used': False,
        'purpose': 'Check whether partial parameter adaptation retains baseline domains while gaining on ExpW. All parameters use one global coefficient; no per-class probability edits.',
        'acceptance': 'Existing multidomain selection-policy.json remains unchanged; all blended models require fresh validation and final export/quantization checks.'}
    (output / 'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
