"""Experimental validation cache with low-confidence face reverification, not product code."""
import argparse
import json
from pathlib import Path
import cv2
import prepare_video_validation as original

RAW_DETECT = original.detect_faces
AUDIT = []
VERIFY_MODE = 'all-low'


def verified_faces(detector, image):
    candidates = RAW_DETECT(detector, image)
    # A stricter check is optional only when another convincing face competes
    # with a weak larger box. Single-person sensitivity can remain unchanged.
    if VERIFY_MODE in ('competing-low', 'competing-larger', 'nearest-competitor') and (len(candidates) < 2 or max(f[4] for f in candidates) < .9):
        return candidates
    largest_confident_area = max((f[2] * f[3] for f in candidates if f[4] >= .9), default=0)
    height, width = image.shape[:2]
    accepted = []
    for face in candidates:
        x, y, w, h, score, points = face
        if score >= .9:
            accepted.append(face)
            continue
        if VERIFY_MODE == 'competing-larger' and w * h < 1.5 * largest_confident_area:
            accepted.append(face)
            continue
        if VERIFY_MODE == 'nearest-competitor' and w * h < largest_confident_area:
            accepted.append(face)
            continue
        left, top, crop_w, crop_h = original.EmotionRecognizer._square_face_box(x, y, w, h, width, height)
        crop = image[top:top + crop_h, left:left + crop_w]
        secondary_score = 0.
        if crop.size:
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA if min(crop.shape[:2]) >= 224 else cv2.INTER_LINEAR)
            for sx, sy, sw, sh, confidence, _ in RAW_DETECT(detector, resized):
                if abs((sx + sw / 2) / 224 - .5) < .3 and abs((sy + sh / 2) / 224 - .5) < .3 and sw * sh / 224**2 >= .08:
                    secondary_score = max(secondary_score, confidence)
        keep = secondary_score >= .85
        AUDIT.append({'original_score': score, 'secondary_score': secondary_score, 'accepted': keep})
        if keep:
            accepted.append(face)
    return accepted


def main():
    global VERIFY_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('all-low', 'competing-low', 'competing-larger', 'nearest-competitor'), default='all-low')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-fps', type=float, default=4.0)
    parser.add_argument('--crop-format', choices=('jpg', 'png'), default='jpg')
    parser.add_argument('--camera-max-side', type=int, choices=(512,768), default=512)
    args = parser.parse_args()
    VERIFY_MODE = args.mode
    if not 0 < args.sample_fps <= 60:
        raise ValueError('Sampling rate must be within(0,60]')
    original.SAMPLE_FPS = args.sample_fps
    original.CROP_FORMAT = args.crop_format
    original.CAMERA_MAX_SIDE = args.camera_max_side
    original.OUT = args.output or (original.ROOT / 'data/video-validation-reverified-20260928')
    existing_manifest = original.OUT / 'manifest.json'
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding='utf-8'))
        if (existing.get('target_sampling_fps', 4.0) != args.sample_fps
                or existing.get('crop_format', 'jpg') != args.crop_format
                or existing.get('camera_max_side',512) != args.camera_max_side
                or existing.get('face_reverification', {}).get('mode') != args.mode):
            raise ValueError('Completed cache uses a different preprocessing policy; choose a new output directory')
        print('CACHE_ALREADY_COMPLETE', str(original.OUT), flush=True)
        return
    original.detect_faces = verified_faces
    original.main()
    path = original.OUT / 'manifest.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata['preprocessing'] += f'; experimental {VERIFY_MODE} face reverification: original>=.9 OR central crop second-pass>=.85; not yet deployed'
    metadata['face_reverification'] = {'original_bypass': .9, 'secondary_threshold': .85,
        'mode': VERIFY_MODE,
        'minimum_competing_area_ratio': 1.5 if VERIFY_MODE == 'competing-larger' else 1.0 if VERIFY_MODE == 'nearest-competitor' else None,
        'center_tolerance_fraction': .3, 'minimum_secondary_box_fraction': .08,
        'checked_candidates': len(AUDIT), 'accepted_after_check': sum(r['accepted'] for r in AUDIT),
        'rejected_after_check': sum(not r['accepted'] for r in AUDIT),
        'audit_scope': 'Secondary checks performed in this invocation; previously completed clip caches are not re-counted.',
        'candidates': AUDIT}
    path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
