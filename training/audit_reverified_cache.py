"""Check fixed original-video and real-eyeglasses retention, without reading holdout images."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_caer_review import box_iou
from prepare_expw_original import overlap
from screen_caer_pilot import camera_frame, create_detector, detect_faces
import prepare_video_reverified as experiment

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'training/overnight-20260927/caer-face-review'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--mode', choices=('competing-low', 'competing-larger', 'nearest-competitor'), required=True)
    p.add_argument('--prefix', required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    experiment.VERIFY_MODE = args.mode
    original = json.loads((ROOT / 'data/video-validation-20260927/manifest.json').read_text())
    updated = json.loads((args.cache / 'manifest.json').read_text())
    old_frames = {(v['id'], f['index']): f for v in original['videos'] for f in v['frames']}
    new_frames = {(v['id'], f['index']): f for v in updated['videos'] for f in v['frames']}
    assert old_frames.keys() == new_frames.keys()
    annotations = json.loads((WORK / 'annotations-v1.json').read_text())
    changes = []
    for row in annotations['rows']:
        key = (row['video_id'], row['frame_index'])
        iou = box_iou(old_frames[key].get('bbox'), new_frames[key].get('bbox'))
        if iou < .5:
            changes.append(dict(row, new_frame=new_frames[key], iou=iou))
    audit = {'old_counts': dict(Counter(f['status'] for f in old_frames.values())),
             'new_counts': dict(Counter(f['status'] for f in new_frames.values())),
             'selected_face_changed_all_frames': sum(box_iou(old_frames[k].get('bbox'), f.get('bbox')) < .5
                                                     for k, f in new_frames.items() if old_frames[k].get('bbox') or f.get('bbox')),
             'review_frame_changed_counts': dict(Counter(r['visible_expression'] for r in changes)),
             'reverification': {k: v for k, v in updated['face_reverification'].items() if k != 'candidates'},
             'changed_review_rows': changes}
    (WORK / f'{args.prefix}-cache-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

    metadata = json.loads((ROOT / 'data/seven-class/expw-original-eyewear-v3/dataset.json').read_text())
    originals = {p.name: p for p in Path('D:/ExpW-source/original').rglob('*.jpg')}
    rows = [r for r in metadata['images'] if r['split'] == 'expw-val' and r.get('real_eyewear')]
    assert len(rows) == 196
    detector = create_detector()
    records = []
    for row in rows:
        source = cv2.imdecode(np.fromfile(originals[row['source']], dtype=np.uint8), cv2.IMREAD_COLOR)
        source_h, source_w = source.shape[:2]
        frame = camera_frame(source)
        height, width = frame.shape[:2]
        left, top, right, bottom = row['annotation_bbox']
        reference = np.asarray([left, top, right-left, bottom-top], dtype=np.float32)
        reference *= np.asarray([width/source_w, height/source_h, width/source_w, height/source_h])
        before = detect_faces(detector, frame)
        after = experiment.verified_faces(detector, frame)
        original_detected = any(overlap(f[:4], reference) >= .25 for f in before)
        updated_detected = any(overlap(f[:4], reference) >= .25 for f in after)
        records.append({'sha256': row['sha256'], 'source': row['source'], 'label': row['label'],
                        'original_detected': original_detected, 'reverified_detected': updated_detected,
                        'original_candidate_count': len(before), 'updated_candidate_count': len(after)})
    report = {'scope': f'{args.mode} face retention on all196 confirmed eyewear validation originals512/JPEG85, annotation IoU>=.25. Not expression accuracy.',
              'examples': len(records), 'original_detected': sum(r['original_detected'] for r in records),
              'reverified_detected': sum(r['reverified_detected'] for r in records),
              'lost': sum(r['original_detected'] and not r['reverified_detected'] for r in records),
              'records': records}
    (WORK / f'{args.prefix}-eyewear-retention.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'video': {k: v for k, v in audit.items() if k != 'changed_review_rows'},
                      'eyewear': {k: v for k, v in report.items() if k != 'records'}}), flush=True)


if __name__ == '__main__':
    main()
