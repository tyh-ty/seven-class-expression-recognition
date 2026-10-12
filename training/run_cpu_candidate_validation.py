"""Evaluate a named research graph on the fixed photos and exact camera frames."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'training/overnight-20260927'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--skip-photos', action='store_true')
    parser.add_argument('--other-rgb', action='store_true')
    args = parser.parse_args()
    if not args.name.replace('-', '').isalnum():
        raise ValueError('Use a safe output name')
    photo = WORK / (args.name + '-photos.json')
    video = WORK / ('video-' + args.name + '-10fps')
    jobs = []
    if not args.skip_photos:
        old = json.loads((WORK / 'selective-fp32-photos.json').read_text())
        datasets = next(iter(old['reports'].values()))
        command = ['training/evaluate_onnx_ensemble.py', '--base',
                   'models/vit_emotion_seven_int8.onnx', '--other', args.model,
                   '--threads', '2', '--output', str(photo)]
        for domain, row in datasets.items():
            command += ['--dataset', domain, str(Path(row['records'][0]['file']).parent.parent)]
        jobs += [('photos', command), ('photo-rubric', ['training/assess_multidomain_candidates.py',
                 '--report', str(photo), '--output', str(WORK / (args.name + '-photos-rubric.json'))])]
    jobs += [('video', ['training/evaluate_video_onnx.py', '--base',
              'models/vit_emotion_seven_int8.onnx', '--other', args.model,
              '--cache', 'data/video-validation-combined-10fps-v1', '--output', str(video),
              '--other-name', args.name, '--threads', '2',
              '--base-report', str(WORK / 'video-int8-10fps/baseline-int8.json'),
              '--base-report', str(WORK / 'video-enterface-initial-int8/baseline-int8.json')]),
             ('intensity', ['training/stratify_video_validation.py', '--reports', str(video),
              '--output', str(WORK / (args.name + '-intensity.json'))]),
             ('eyewear', ['training/stratify_enterface_video.py', '--reports', str(video),
              '--output', str(WORK / (args.name + '-eyewear.json'))])]
    state = {'name': args.name, 'model': args.model, 'jobs': {}, 'status': 'running',
             'started_at': datetime.now(timezone.utc).isoformat()}
    status_path = WORK / (args.name + '-pipeline.json')
    for key, command in jobs:
        if args.other_rgb and key in ('photos', 'video'):
            command += ['--other-rgb']
        state['current'] = key
        status_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        with (WORK / (args.name + '-' + key + '.log')).open('w', encoding='utf-8') as out:
            result = subprocess.run([sys.executable, '-X', 'utf8', '-u', *command],
                                    cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        state['jobs'][key] = {'exit_code': result.returncode,
                              'finished_at': datetime.now(timezone.utc).isoformat()}
        if result.returncode:
            state['status'] = 'failed'
            status_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
            raise RuntimeError(f'{args.name}: {key} failed; inspect its log')
    state['status'] = 'completed-research-only'
    state['current'] = None
    status_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
