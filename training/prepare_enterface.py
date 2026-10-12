"""Subject-isolated camera-shaped eNTERFACE video caches, with sealed test actors."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2

import prepare_video_validation as video
import prepare_video_reverified as verified

FRACTIONS = [.2, .35, .5, .65, .8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('D:/eNTERFACE-source/manifest.json'))
    parser.add_argument('--split', choices=('train', 'val', 'holdout'), required=True)
    parser.add_argument('--frozen-selection', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.split == 'holdout':
        if args.frozen_selection is None:
            raise ValueError('Holdout preparation requires a frozen final system')
        from evaluate_frozen_system import read_frozen
        read_frozen(args.frozen_selection)
    cv2.setNumThreads(1)
    source = json.loads(args.source.read_text(encoding='utf-8'))
    split_rows = [r for r in source['videos'] if r['split'] == args.split]
    # Subject6 is stored as long unsegmented recordings. Do not assume its
    # arbitrary central frames inherit the carefully segmented clip labels.
    excluded = [r['source_entry'] for r in split_rows if r['sentence'] is None]
    rows = [r for r in split_rows if r['sentence'] is not None]
    all_hash_splits = {}
    for row in source['videos']:
        if 'sha256' in row:
            all_hash_splits.setdefault(row['sha256'], set()).add(row['split'])
    if any(len(splits) > 1 for splits in all_hash_splits.values()):
        raise ValueError('Identical video bytes cross train/validation subjects; audit before preparation')
    if args.limit:
        rows = rows[:args.limit]
    video.OUT = args.output
    video.OUT.mkdir(parents=True, exist_ok=True)
    video.SAMPLE_FPS = 10.
    video.CROP_FORMAT = 'png'
    verified.VERIFY_MODE = 'competing-larger'
    video.detect_faces = verified.verified_faces
    records = []
    for row in rows:
        record = {'id': 'enterface-'+hashlib.sha256(row['source_entry'].encode()).hexdigest()[:16],
                  'source': 'eNTERFACE-'+args.split, 'split': args.split,
                  'label': row['label'], 'subject': row['subject'], 'sentence': row['sentence'],
                  'path': row['file'], 'source_entry': row['source_entry'], 'source_sha256': row['sha256'],
                  'known_label_window': [.2, .8]}
        if args.split == 'train':
            record['sample_fractions'] = FRACTIONS
        records.append(record)
    results = []
    with ThreadPoolExecutor(2) as pool:
        for i, result in enumerate(pool.map(video.process, records), 1):
            results.append(result)
            if i % 20 == 0 or i == len(records):
                print('PREPARE', args.split, i, '/', len(records), flush=True)
    report = {'scope': 'Controlled elicited emotional speech from public research dataset. Source clip label is not per-frame expert facial truth. Subject split fixed before any model outputs. No neutral class. '+('Reserved test actors opened only after final system frozen.' if args.split == 'holdout' else 'No test actors opened.'),
              'split': args.split, 'source_manifest': str(args.source),
              'source_manifest_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'preprocessing': 'Browser<=512/JPEG85; production competing-larger YuNet verification; largest face square1.45;224pxlosslessPNG',
              'target_sampling_fps': 10 if args.split != 'train' else None, 'crop_format': 'png',
              'sample_fractions': FRACTIONS if args.split == 'train' else None,
              'partial_smoke_only': args.limit is not None,
              'excluded_unsegmented_recordings': excluded,
              'counts': dict(Counter(r['label'] for r in results)),
              'subjects': sorted(set(r['subject'] for r in results)), 'videos': results}
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('COMPLETE', json.dumps({k: v for k, v in report.items() if k != 'videos'}), flush=True)


if __name__ == '__main__':
    main()
