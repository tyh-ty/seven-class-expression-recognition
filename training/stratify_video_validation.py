"""Separate normal/strong acted expressions and actual frontend recognition streaks."""
import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')


def summarize(videos, displayed):
    duration_available = all('stableCorrectAtLeastHalfSecond' in displayed[v['id']] for v in videos)
    result = {'clips': len(videos),
              'clip_aggregate_accuracy': sum(v['aggregate_prediction'] == v['label'] for v in videos) / max(1, len(videos)),
              'stable_correct_clip_rate': sum(displayed[v['id']]['stableCorrectAtLeastThreeFrames'] for v in videos) / max(1, len(videos)),
              'stable_correct_half_second_clip_rate': sum(displayed[v['id']]['stableCorrectAtLeastHalfSecond'] for v in videos) / max(1, len(videos)) if duration_available else None,
              'per_class': {}}
    for label in CLASSES:
        subset = [v for v in videos if v['label'] == label]
        count = len(subset)
        result['per_class'][label] = {'support_clips': count,
            'clip_aggregate_recall': sum(v['aggregate_prediction'] == label for v in subset) / count if count else None,
            'stable_correct_clip_rate': sum(displayed[v['id']]['stableCorrectAtLeastThreeFrames'] for v in subset) / count if count else None,
            'stable_correct_half_second_clip_rate': sum(displayed[v['id']]['stableCorrectAtLeastHalfSecond'] for v in subset) / count if count and duration_available else None,
            'mean_correct_display_fraction': float(np.mean([displayed[v['id']]['correctDisplayFraction'] for v in subset])) if count else None}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reports', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results = {}
    for directory in args.reports:
        for path in sorted(directory.glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if 'videos' not in data:
                continue
            display = json.loads(path.with_stem(path.stem+'-display').read_text(encoding='utf-8'))
            displayed = {clip['id']: clip for clip in display['clips']}
            clips = [v for v in data['videos'] if v['source'] == 'RAVDESS-val']
            results[str(path)] = {'model': data['model'],
                'target_sampling_fps': data.get('target_sampling_fps', 4),
                'crop_format': data.get('crop_format', 'jpg'),
                'all': summarize(clips, displayed),
                'normal_intensity': summarize([v for v in clips if v['intensity'] == '01'], displayed),
                'strong_intensity': summarize([v for v in clips if v['intensity'] == '02'], displayed)}
    report = {'scope': 'RAVDESS actors17–20, existing central35–75% window. Normal acted intensity is not spontaneous micro-expression. Source labels are clip-level, not verified per-frame facial labels; 4 actors/within-actor correlated clips, no real-eyeglasses guarantee. Some public initialization models report RAVDESS pretraining. Legacy stability means three consecutive sampled display frames, whose duration varies with sampling rate. The separate half-second metric requires continuously correct display for at least0.5s and is not first-frame latency. Nominal replay does not model transport backlog.',
              'reports': results}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({Path(k).name: {'normal_accuracy': v['normal_intensity']['clip_aggregate_accuracy'],
                                   'normal_stable': v['normal_intensity']['stable_correct_clip_rate'],
                                   'normal_half_second_stable': v['normal_intensity']['stable_correct_half_second_clip_rate'],
                                   'normal_fear': v['normal_intensity']['per_class']['fear'],
                                   'normal_anger': v['normal_intensity']['per_class']['anger']}
                      for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    main()
