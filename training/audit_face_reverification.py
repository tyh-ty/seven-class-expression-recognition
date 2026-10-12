"""Diagnostic only: second YuNet pass on a crop, preserving original product files."""
from collections import Counter
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from screen_caer_pilot import create_detector, detect_faces


def main():
    cv2.setNumThreads(1)
    work = ROOT / 'training/overnight-20260927/caer-face-review'
    cache = ROOT / 'data/video-validation-20260927'
    annotations = json.loads((work / 'annotations-v1.json').read_text(encoding='utf-8'))
    videos = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))['videos']
    frames = {(v['id'], f['index']): f for v in videos for f in v['frames']}
    detector = create_detector()
    rows = []
    for row in annotations['rows']:
        frame = frames[row['video_id'], row['frame_index']]
        rows.append({'cohort': 'video_no_visible_face' if row['visible_expression'] == 'no_face'
                     else 'video_uncertain' if row['visible_expression'] == 'uncertain' else 'video_clear_expression',
                     'review_index': row['review_index'], 'file': str(cache / row['crop']) if row['crop'] else None,
                     'original_score': frame.get('face_score', 0.)})
    eyewear = ROOT / 'data/expw/real-eyeglasses-original-expanded-val-v1'
    for path in sorted(eyewear.glob('*/*.jpg')):
        rows.append({'cohort': 'verified_photo_eyeglasses', 'file': str(path),
                     'original_score': None})
    for row in rows:
        if row['file'] is None:
            row['secondary_score'] = 0.
            continue
        image = cv2.imdecode(np.fromfile(row['file'], dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        faces = detect_faces(detector, image)
        # The padded original target should remain central and occupy useful area.
        central = [f for f in faces if abs((f[0] + f[2] / 2) / w - .5) < .3
                   and abs((f[1] + f[3] / 2) / h - .5) < .3 and f[2] * f[3] / (w * h) >= .08]
        row['secondary_score'] = max((f[4] for f in central), default=0.)
    summary = {}
    for threshold in (.8, .85, .9):
        counts = {}
        for cohort in sorted({r['cohort'] for r in rows}):
            members = [r for r in rows if r['cohort'] == cohort]
            counts[cohort] = {'total': len(members),
                'accepted_secondary_only': sum(r['secondary_score'] >= threshold for r in members),
                'accepted_original_or_secondary': sum((r['original_score'] or 0.) >= threshold or r['secondary_score'] >= threshold for r in members)}
        summary[str(threshold)] = counts
    report = {'scope': 'Exploratory second-detector-pass diagnostic on fixed validation crops. No product or cache altered. Photo cohort lacks original detector confidence, so its acceptance is a conservative secondary-only estimate. Appearance annotations are assistant labels, not expert truth.',
              'thresholds': summary, 'records': rows}
    (work / 'face-reverification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
