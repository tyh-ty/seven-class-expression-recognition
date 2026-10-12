"""Add subject-separated eNTERFACE training to unchanged replay; no holdout reads."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

CLASSES = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')
BASE = Path('data/seven-class/expw-original-eyewear-v3')
OUT = Path('data/seven-class/expw-enterface-v1')


def main():
    workspace = Path(__file__).resolve().parents[1]
    if not OUT.resolve().is_relative_to(workspace):
        raise ValueError('Output must stay within this workspace')
    if OUT.exists():
        raise FileExistsError(OUT)
    temp = OUT.with_name(OUT.name+'-building')
    if not temp.resolve().is_relative_to(workspace):
        raise ValueError('Temporary output must stay within this workspace')
    if temp.exists():
        raise FileExistsError(temp)
    for split in ('train', 'val', 'enterface-val', 'enterface-eyewear-val'):
        for label in CLASSES:
            (temp/split/label).mkdir(parents=True, exist_ok=True)
    review_path = Path('training/overnight-20260927/enterface-review/eyewear-review-v1.json')
    review = json.loads(review_path.read_text())
    clear = set(review['clear_eyeglasses_subjects'])
    tinted = set(review['tinted_eyeglasses_subjects'])
    baseline = json.loads((BASE/'dataset.json').read_text())
    rows = [dict(row) for row in baseline['images'] if row['split'] == 'train']
    for split, source in [('train', BASE/'train'), ('val', Path('data/ferplus/validation-dedup-v1'))]:
        for label in CLASSES:
            for file in (source/label).iterdir():
                if file.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
                    os.link(file, temp/split/label/file.name)
    new_rows = []
    subjects = {}
    for split, root in [('train', Path('data/enterface/train-frames-v2')),
                        ('enterface-val', Path('data/video-validation-enterface-10fps-v2'))]:
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['partial_smoke_only']:
            raise ValueError('Cannot use a smoke-only cache for training')
        subjects[split] = set(manifest['subjects'])
        for video in manifest['videos']:
            detected = [f for f in video['frames'] if f['status'] == 'detected']
            if split != 'train' and detected:
                # Fixed fractions only, no model-based selection. Frames in one
                # video and all videos of one actor remain correlated.
                selected = {min(detected, key=lambda f: abs(f['fraction']-t))['index'] for t in [.2,.35,.5,.65,.8]}
                detected = [f for f in detected if f['index'] in selected]
            for frame in detected:
                source = root/frame['crop']
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                name = f"web_enterface_s{video['subject']:02d}_{video['id'].split('-')[-1]}_{frame['index']:06d}.png"
                row = {'split': split, 'label': video['label'], 'subject': video['subject'],
                       'video_id': video['id'], 'source': video['source_entry'], 'source_sha256': video['source_sha256'],
                       'frame_index': frame['index'], 'fraction': frame['fraction'], 'group': 'enterface-subject-'+str(video['subject']),
                       'sha256': digest, 'cache_path': str(source), 'path': str(Path(split)/video['label']/name),
                       'real_eyewear': video['subject'] in clear, 'eyewear_possible': video['subject'] in clear | tinted,
                       'eyewear_tag_scope': 'Actor-level review on fixed6emotion samples; not independently verified for every frame.',
                       'landmarks': frame['landmarks'], 'label_scope': 'Source clip-level elicited emotion; individual frames may be ambiguous.'}
                new_rows.append(row)
    if subjects['train'] & subjects['enterface-val']:
        raise ValueError('Training/validation subject overlap')
    by_hash = defaultdict(list)
    for row in new_rows:
        by_hash[row['sha256']].append(row)
    discarded = []
    kept = []
    for digest, group in by_hash.items():
        if len({r['split'] for r in group}) > 1 or len({r['label'] for r in group}) > 1:
            discarded.extend(group)
        else:
            kept.append(group[0])
    for row in kept:
        target = temp/row['path']
        os.link(row['cache_path'], target)
        if row['split'] == 'enterface-val' and row['real_eyewear']:
            os.link(target, temp/'enterface-eyewear-val'/row['label']/target.name)
    rows.extend(kept)
    counts = {f'{split}/{label}': len(list((temp/split/label).iterdir()))
              for split in ('train', 'val', 'enterface-val', 'enterface-eyewear-val') for label in CLASSES}
    report = {'source_dataset': str(BASE), 'new_video_source': 'D:/eNTERFACE-source/manifest-stratified-v2.json',
              'counts': counts, 'images': rows, 'new_unique_images': len(kept),
              'new_train_images': sum(r['split']=='train' for r in kept),
              'new_eyewear_train_images': sum(r['split']=='train' and r['real_eyewear'] for r in kept),
              'new_eyewear_fear_train_images': sum(r['split']=='train' and r['real_eyewear'] and r['label']=='fear' for r in kept),
              'new_subjects': {k: sorted(v) for k,v in subjects.items()},
              'eyewear_review': str(review_path), 'conflicting_or_cross_split_frames_excluded': len(discarded),
              'same_label_duplicate_frames_collapsed': len(new_rows)-len(discarded)-len(kept),
              'primary_validation': 'Unchanged deduplicated FER validation2922 images.',
              'target_validation': 'Use unchanged originalExpW2917 externally. Additional eNTERFACE val is six classes, correlated videos and actor-level splits, not independent frame accuracy.',
              'holdout_read': False, 'limitations': 'Controlled acted emotional speech with noisy per-frame labels, not natural/microexpression ground truth. Maintain neutral/base replay and existing acceptance rules.'}
    (temp/'dataset.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (temp/'excluded-new-frames.json').write_text(json.dumps(discarded, indent=2), encoding='utf-8')
    temp.replace(OUT)
    print(json.dumps({k: v for k,v in report.items() if k not in ('images','counts')}), flush=True)


if __name__ == '__main__':
    main()
