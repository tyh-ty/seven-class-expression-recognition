"""Compare cached predictions with frozen exploratory face annotations; never retrain."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')


def box_iou(first, second):
    if not first or not second:
        return 0.
    overlap = max(0, min(first['x'] + first['width'], second['x'] + second['width']) - max(first['x'], second['x'])) * max(0, min(first['y'] + first['height'], second['y'] + second['height']) - max(first['y'], second['y']))
    return overlap / max(1, first['width'] * first['height'] + second['width'] * second['height'] - overlap)


def summarize(rows, field):
    matrix = np.zeros((7, 8), dtype=np.int64)
    for row in rows:
        if row['visible_expression'] not in CLASSES or row.get('selected_face_changed'):
            continue
        predicted = row[field]
        matrix[CLASSES.index(row['visible_expression']),
               CLASSES.index(predicted) if predicted in CLASSES else 7] += 1
    tp = np.diag(matrix[:, :7])
    support = matrix.sum(axis=1)
    precision = tp / np.maximum(1, matrix[:, :7].sum(axis=0))
    recall = tp / np.maximum(1, support)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    valid = support > 0
    return {'annotated_frames': int(support.sum()),
            'agreement_including_abstentions': float(tp.sum() / max(1, support.sum())),
            'macro_f1_supported_classes': float(f1[valid].mean()) if valid.any() else None,
            'classes_with_no_annotations': [name for i, name in enumerate(CLASSES) if not valid[i]],
            'per_class': {name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                 'f1': float(f1[i]), 'support': int(support[i])}
                          for i, name in enumerate(CLASSES)},
            'confusion_with_abstain_column': matrix.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = ROOT / 'training/overnight-20260927/caer-face-review'
    annotations_path = work / 'annotations-v1.json'
    checksum = hashlib.sha256(annotations_path.read_bytes()).hexdigest()
    assert checksum == (work / 'annotations-v1.sha256').read_text(encoding='utf-8').strip()
    annotations = json.loads(annotations_path.read_text(encoding='utf-8'))
    original_cache = json.loads((ROOT / 'data/video-validation-20260927/manifest.json').read_text(encoding='utf-8'))
    original_frames = {(video['id'], frame['index']): frame for video in original_cache['videos'] for frame in video['frames']}
    candidates = {}
    for directory in args.reports:
        for path in sorted(directory.glob('*.json')):
            report = json.loads(path.read_text(encoding='utf-8'))
            if 'videos' not in report:
                continue
            display_path = path.with_name(path.stem + '-display.json')
            display = json.loads(display_path.read_text(encoding='utf-8'))
            frames = {(video['id'], frame['index']): frame for video in report['videos'] for frame in video['frames']}
            displayed = {(video['id'], frame['index']): frame['displayed'] for video in display['clips'] for frame in video['frames']}
            rows = []
            for annotation in annotations['rows']:
                key = (annotation['video_id'], annotation['frame_index'])
                frame = frames[key]
                scores = frame.get('probabilities')
                rows.append(dict(annotation, raw_prediction=CLASSES[int(np.argmax(scores))] if scores else None,
                                 displayed_prediction=displayed[key],
                                 selected_face_changed=box_iou(frame.get('bbox'), original_frames[key].get('bbox')) < .5))
            no_face = [row for row in rows if row['visible_expression'] == 'no_face' and not row['selected_face_changed']]
            candidates[str(path)] = {
                'model': report['model'], 'raw': summarize(rows, 'raw_prediction'),
                'displayed': summarize(rows, 'displayed_prediction'),
                'eyeglasses_raw': summarize([r for r in rows if r['real_eyeglasses']], 'raw_prediction'),
                'eyeglasses_displayed': summarize([r for r in rows if r['real_eyeglasses']], 'displayed_prediction'),
                'no_visible_face_crops': len(no_face),
                'changed_selected_face_frames_excluded': sum(r['selected_face_changed'] for r in rows),
                'emotion_displayed_on_no_visible_face_crop': sum(r['displayed_prediction'] is not None for r in no_face),
                'records': rows}
    output = {'scope': annotations['scope'], 'limits': annotations['limits'],
              'annotation_sha256': checksum, 'annotation_counts': annotations['counts'],
              'coverage': sum(r['visible_expression'] in CLASSES for r in annotations['rows']) / len(annotations['rows']),
              'reports': candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'coverage': output['coverage'], 'annotation_counts': output['annotation_counts'],
        'candidates': {k: {mode: v[mode]['agreement_including_abstentions']
                           for mode in ('raw', 'displayed', 'eyeglasses_raw')} for k, v in candidates.items()}}))


if __name__ == '__main__':
    main()
