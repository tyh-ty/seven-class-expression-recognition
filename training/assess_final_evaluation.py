"""Apply the frozen final-test policy without selecting another candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def assess(photos, videos, rules):
    b, s = photos['results']['baseline'], photos['results']['selected']
    bv, sv = videos['results']['baseline'], videos['results']['selected']
    eps = 1e-9
    photo_drops = {c: r['recall'] - b['per_class'][c]['recall']
                   for c, r in s['per_class'].items() if r['support'] >= 20}
    video_drops = {c: r['recall'] - bv['per_class'][c]['recall']
                   for c, r in sv['per_class'].items() if r['support'] >= 20}
    checks = {
        'ExpW_macro_f1_no_drop': s['macro_f1'] + eps >= b['macro_f1'],
        'ExpW_mean_anger_fear_recall_drop_max': sum(s['per_class'][c]['recall'] - b['per_class'][c]['recall'] for c in ('anger', 'fear')) / 2 + eps >= -rules['ExpW_mean_anger_fear_recall_drop_max'],
        'ExpW_single_class_recall_drop_max_support20': all(d + eps >= -rules['ExpW_single_class_recall_drop_max_support20'] for d in photo_drops.values()),
        'enterface_present_class_macro_f1_no_drop': sv['macro_f1_present'] + eps >= bv['macro_f1_present'],
        'enterface_halfsecond_stability_no_drop': sv['stable_correct_half_second_clip_rate'] + eps >= bv['stable_correct_half_second_clip_rate'],
        'enterface_single_class_recall_drop_max_support20': all(d + eps >= -rules['enterface_single_class_recall_drop_max_support20'] for d in video_drops.values()),
    }
    return dict(checks=checks, passed=all(checks.values()),
                failed=[k for k, v in checks.items() if not v],
                photo_recall_changes=photo_drops, video_recall_changes=video_drops,
                release_system='selected' if all(checks.values()) else 'baseline')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=Path('training/overnight-20260927/final-evaluation-protocol.json'))
    args = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    photos, videos, protocol = read(args.directory/'photos-final.json'), read(args.directory/'videos-final.json'), read(args.protocol)
    fingerprint = hashlib.sha256(args.decision.read_bytes()).hexdigest()
    if any(r['decision_sha256'] != fingerprint for r in (photos, videos)):
        raise ValueError('Final reports use a different frozen decision')
    result = assess(photos, videos, protocol['final_checks'])
    result.update(assessed_at=datetime.now(timezone.utc).isoformat(), decision_sha256=fingerprint,
                  policy_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                  scope='One selected system against baseline, previously reserved data. Failure retains baseline; no alternate candidate or tuning after final results.')
    (args.directory/'acceptance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
