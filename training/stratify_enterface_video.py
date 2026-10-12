"""Report clip and stable-display diagnostics by held-back subject and eyewear."""
import argparse
import json
from pathlib import Path

from stratify_video_validation import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(Path('training/overnight-20260927/enterface-review/eyewear-review-v1.json').read_text())
    eyewear = set(review['clear_eyeglasses_subjects'])
    results = {}
    for directory in args.reports:
        for path in sorted(directory.glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if 'videos' not in data:
                continue
            videos = [v for v in data['videos'] if v['source'] == 'eNTERFACE-val']
            if not videos:
                continue
            display = json.loads(path.with_stem(path.stem+'-display').read_text(encoding='utf-8'))
            displayed = {clip['id']: clip for clip in display['clips']}
            subjects = sorted({v['subject'] for v in videos})
            groups = {'all': videos, 'eyewear': [v for v in videos if v['subject'] in eyewear],
                      'no_eyewear': [v for v in videos if v['subject'] not in eyewear]}
            results[str(path)] = {'model': data['model'], 'subjects': subjects,
                'eyewear_subjects': [s for s in subjects if s in eyewear],
                'groups': {name: summarize(clips, displayed) for name, clips in groups.items()},
                'per_subject': {str(s): summarize([v for v in videos if v['subject']==s], displayed) for s in subjects}}
    report = {'scope': 'Six-class controlled elicited speech clips, withoutneutral. Original clip labels can differ from instantaneous facial expression. Actor-stratified validation,8subjects/2clear-eyeglasses subjects: clips/frames are correlated, not independentwebcam accuracy. Eyewear is sampled actor-level appearance review. No final test actors read.',
              'reports': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({Path(k).name: {'aggregate': v['groups']['all']['clip_aggregate_accuracy'],
        'stable_half_second': v['groups']['all']['stable_correct_half_second_clip_rate'],
        'eyewear': v['groups']['eyewear']} for k,v in results.items()}), flush=True)


if __name__ == '__main__':
    main()
