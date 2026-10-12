"""Open the six reserved actors only after validating an immutable final selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

from evaluate_frozen_system import read_frozen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    read_frozen(args.decision)
    fingerprint = hashlib.sha256(args.decision.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    start = args.output/'preparation-started.json'
    if start.exists():
        if json.loads(start.read_text())['decision_sha256'] != fingerprint:
            raise ValueError('Reserved actors were opened for a different selection')
    else:
        start.write_text(json.dumps({'decision_sha256':fingerprint,'opened_at':datetime.now(timezone.utc).isoformat()},indent=2))
    manifest_path = Path('D:/eNTERFACE-source/manifest-stratified-v2.json')
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    archive_path = Path('D:/eNTERFACE-source/enterface05-pinned.zip')
    with archive_path.open('rb') as handle:
        if hashlib.file_digest(handle,'sha256').hexdigest() != source['archive_sha256']:
            raise ValueError('Pinned original video archive changed')
    rows = [r for r in source['videos'] if r['split']=='holdout' and r['sentence'] is not None]
    if len(rows) != 180 or len({r['subject'] for r in rows}) != 6:
        raise ValueError('Reserved actor cohort changed')
    previous = {r['sha256'] for r in source['videos'] if r['split']!='holdout' and 'sha256' in r}
    with ZipFile(archive_path) as archive:
        for row in rows:
            info = archive.getinfo(row['source_entry'])
            if info.file_size != row['bytes'] or info.CRC != row['crc32']:
                raise ValueError('Reserved archive entry changed')
            raw = archive.read(info)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in previous:
                raise ValueError('Identical video bytes cross training/validation and final actors')
            destination = args.output/'sources'/(hashlib.sha256(row['source_entry'].encode()).hexdigest()[:24]+'.avi')
            if not destination.resolve().is_relative_to(args.output.resolve()):
                raise ValueError('Source extraction escaped output')
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(raw)
            row.update(file=str(destination.resolve()),sha256=digest)
    source.update(holdout_videos_extracted=True,final_selection_sha256=fingerprint,
                  prior_manifest=str(manifest_path),prior_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    updated = args.output/'source-manifest.json'
    updated.write_text(json.dumps(source,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('prepare_enterface.py')),
        '--source',str(updated),'--split','holdout','--output',str(args.output/'frames'),
        '--frozen-selection',str(args.decision)],check=True)


if __name__=='__main__':
    main()
