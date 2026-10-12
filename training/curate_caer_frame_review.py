"""Freeze exploratory visible-expression annotations before reading frame predictions."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["neutral","neutral","no_face","uncertain","uncertain","neutral","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","neutral","neutral","neutral","neutral","neutral","happiness","neutral","neutral","uncertain","uncertain","disgust","disgust","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","neutral","neutral","neutral","no_face","uncertain","uncertain","uncertain","uncertain","uncertain","anger","no_face","uncertain","happiness","happiness","happiness","no_face","no_face","happiness","uncertain","uncertain","uncertain","neutral","neutral","happiness","no_face","uncertain","uncertain","uncertain","uncertain","uncertain","happiness","neutral","happiness","uncertain","uncertain","uncertain","anger","anger","anger","uncertain","uncertain","disgust","uncertain","uncertain","surprise","neutral","uncertain","neutral","neutral","neutral","neutral","neutral","neutral","neutral","happiness","happiness","uncertain","happiness","happiness","happiness","no_face","no_face","no_face","uncertain","disgust","disgust","neutral","neutral","neutral","uncertain","uncertain","uncertain","neutral","neutral","neutral","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","no_face","no_face","no_face","happiness","happiness","happiness","uncertain","uncertain","uncertain","neutral","neutral","neutral","sadness","sadness","sadness","no_face","happiness","happiness","uncertain","uncertain","uncertain","no_face","no_face","no_face","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","uncertain","no_face","uncertain","disgust","happiness","happiness","neutral","disgust","disgust","disgust","neutral","neutral","neutral","neutral","neutral","neutral","uncertain","neutral","neutral","uncertain","uncertain","uncertain","happiness","happiness","happiness","surprise","surprise","surprise","neutral","neutral","neutral","happiness","happiness","happiness","uncertain","uncertain","uncertain","happiness","neutral","neutral","uncertain","uncertain","uncertain","neutral","neutral","neutral","neutral","uncertain","uncertain","neutral","neutral","neutral","uncertain","uncertain","uncertain","neutral","neutral","neutral","neutral","no_face","neutral","happiness","happiness","happiness","neutral","uncertain","uncertain","uncertain","uncertain","uncertain","neutral","happiness","happiness","neutral","happiness","happiness","anger","anger","anger","anger","anger","disgust","happiness","happiness","happiness","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","uncertain","uncertain","uncertain","happiness","happiness","happiness","uncertain","uncertain","uncertain","surprise","uncertain","surprise","anger","disgust","disgust","neutral","neutral","neutral","uncertain","uncertain","uncertain","uncertain","uncertain","neutral"]
EYEGLASSES = set([18,19,20,54,55,56,105,106,107,117,118,119,148,149,246,247,248,249,250,251])


def main():
    work = ROOT / 'training/overnight-20260927/caer-face-review'
    manifest = json.loads((work / 'manifest.json').read_text(encoding='utf-8'))
    assert len(LABELS) == len(manifest['rows']) == 270
    rows = [dict(row, visible_expression=label, real_eyeglasses=row['review_index'] in EYEGLASSES)
            for row, label in zip(manifest['rows'], LABELS)]
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Assistant visual appearance review at224px per face of three fixed frames from each of90 validation clips. No source emotion labels or frame predictions displayed during review; earlier aggregate CAER scores were known. Frozen before per-frame prediction comparison.',
        'annotation_policy': 'Seven visible-expression categories, uncertain for ambiguous/mixed/speaking/occluded faces, and no_face when selected crop contains body/back-of-head/background rather than readable face. Fear and surprise are not forced apart without sufficient visible evidence. Eyeglasses flagged only when ordinary frames visible.',
        'limits': 'Exploratory assistant annotation, NOT independent expert ground truth. Sparse temporal samples, correlated adjacent frames, few identities, selected clear faces may be easier. No confirmed fear frames. Report agreement and coverage, not natural-scene seven-class accuracy. Never train on these validation frames.',
        'counts': dict(Counter(LABELS)), 'rows': rows}
    path = work / 'annotations-v1.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / 'annotations-v1.sha256').write_text(checksum + '\n', encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'sha256': checksum}))


if __name__ == '__main__':
    main()
