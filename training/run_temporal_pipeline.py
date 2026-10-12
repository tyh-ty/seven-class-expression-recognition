"""Wait for complete train-only input cache, then run bounded CPU research jobs."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'training/overnight-20260927'
RUN='training/runs/temporal-residual-v1'
VIDEOS='training/overnight-20260927/video-temporal-residual-v1'
REPORTS=['training/overnight-20260927/video-int8-10fps/baseline-int8.json',
         'training/overnight-20260927/video-enterface-initial-int8/baseline-int8.json']


def record(**fields):
    path=OUT/'temporal-pipeline-status.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'updated_at':datetime.now(timezone.utc).isoformat(),**fields},indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    global RUN, VIDEOS, REPORTS, record
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-manifest',type=Path,default=ROOT/'data/temporal-training-int8-v1/manifest.json')
    parser.add_argument('--run',default=RUN)
    parser.add_argument('--videos',default=VIDEOS)
    parser.add_argument('--validation-report',action='append')
    parser.add_argument('--base-name',default='baseline-int8')
    parser.add_argument('--name',default='temporal')
    args=parser.parse_args()
    if not args.name.replace('-','').isalnum():raise ValueError('Unsafe pipeline name')
    RUN,VIDEOS=args.run,args.videos
    REPORTS=args.validation_report or REPORTS
    def save_status(**fields):
        path=OUT/(args.name+'-pipeline-status.json')
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'updated_at':datetime.now(timezone.utc).isoformat(),**fields},indent=2),encoding='utf-8')
        temporary.replace(path)
    record=save_status
    manifest=args.train_manifest
    deadline=datetime.fromisoformat('2026-09-28T07:10:00+08:00').timestamp()
    while time.time()<deadline:
        if manifest.is_file() and all(Path(p).is_file() for p in REPORTS):
            try:
                data=json.loads(manifest.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                # The collector publishes its final manifest after the last clip.
                time.sleep(1)
                continue
            if not data.get('partial_smoke_only'):
                if len(data['videos'])!=1699:raise ValueError('Unexpected train corpus size')
                break
        record(status='waiting-complete-inputs',manifest=str(manifest),cpu_only=True)
        time.sleep(15)
    else:
        record(status='not-started-before-time-cutoff',cpu_only=True)
        return
    inputs=[arg for p in REPORTS for arg in ['--validation-report',p]]
    jobs=[
      ('train',['training/train_temporal_residual.py','--train-data',str(manifest),*inputs,'--output',RUN,
                '--epochs','30','--batch-size','256','--threads','3','--base-name',args.base_name]),
      ('video',['training/evaluate_temporal_residual.py','--checkpoint',*(RUN+'/'+p for p in ['epoch-010.pt','epoch-020.pt','epoch-030.pt','best.pt']),
                *inputs,'--output',VIDEOS,'--threads','3','--base-name',args.base_name]),
      ('ravdess',['training/stratify_video_validation.py','--reports',VIDEOS,'--output',RUN+'/ravdess-intensity.json']),
      ('eyewear',['training/stratify_enterface_video.py','--reports',VIDEOS,'--output',RUN+'/enterface-eyewear.json']),
    ]
    for name,job_args in jobs:
        with (OUT/f'{args.name}-{name}.log').open('wb') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-u',*job_args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            record(status='running',job=name,child_pid=child.pid,cpu_only=True,log=str(log.name))
            code=child.wait()
            if code:
                record(status='failed',job=name,exit_code=code,log=str(log.name),cpu_only=True)
                raise RuntimeError(f'Temporal {name} failed: {code}')
    record(status='completed-research-only',run=RUN,video=VIDEOS,cpu_only=True,deployed=False)


if __name__=='__main__':
    main()
