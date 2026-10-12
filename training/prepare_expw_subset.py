"""Verify ExpW subset labels against original single-face annotations and group by face similarity."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server

CLASSES = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')
ORIGINAL = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')
MIRROR = {'angry': 'anger', 'disgust': 'disgust', 'fear': 'fear', 'happy': 'happiness',
          'sad': 'sadness', 'surprise': 'surprise', 'neutral': 'neutral'}

def main():
    cv2.setNumThreads(2)
    output = ROOT / 'data/seven-class/expw-subset-v1'
    if output.exists():
        raise FileExistsError(output)
    temp = output.with_name(output.name + '-building')
    temp.mkdir(exist_ok=True)
    report_dir = ROOT / 'training/overnight-20260927/expw-review'
    report_dir.mkdir(exist_ok=True)
    source = Path('D:/ExpW-source')
    labels = defaultdict(list)
    for line in (source / 'label.lst').read_text(encoding='utf-8-sig').splitlines():
        tokens = line.split()
        if tokens:
            labels[tokens[0]].append(tokens)
    detector = cv2.FaceDetectorYN.create(str(server.EmotionRecognizer._opencv_safe_path(server.FACE_DETECTOR_PATH)), '', (224, 224), .6, .3, 5000)
    identity = cv2.FaceRecognizerSF.create(str(server.EmotionRecognizer._opencv_safe_path(ROOT / 'models/face_recognition_sface_2021dec.onnx')), '')
    rows, images, features = [], [], []
    excluded = Counter()
    seen = set()
    with ZipFile(source / 'expw-subset-224.zip') as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.jpg'):
                continue
            filename = Path(name).name
            records = labels[filename]
            label = MIRROR[Path(name).parent.name]
            if len(records) != 1:
                excluded['ambiguous_multiple_original_faces'] += 1
                continue
            if ORIGINAL[int(records[0][-1])] != label:
                raise ValueError('Original label mismatch')
            raw = archive.read(name)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                excluded['exact_duplicate'] += 1
                continue
            seen.add(digest)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Corrupt image')
            detector.setInputSize((image.shape[1], image.shape[0]))
            _, faces = detector.detect(image)
            feature = None
            face = None
            if faces is not None and len(faces):
                face = max(faces, key=lambda f: float(f[2]*f[3]))
                aligned = identity.alignCrop(image, face)
                feature = identity.feature(aligned).reshape(-1)
                feature = feature / max(float(np.linalg.norm(feature)), 1e-10)
            rows.append({'source': filename, 'archive_entry': name, 'label': label, 'sha256': digest,
                         'face_feature_available': feature is not None,
                         'landmarks': face[4:14].reshape(5,2).tolist() if face is not None else None,
                         'source_face_annotation': records[0]})
            images.append(raw)
            features.append(feature)
            if len(rows) % 250 == 0:
                print('verified_and_group_features', len(rows), flush=True)
    parents = list(range(len(rows)))
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a,b)] = min(a,b)
    valid = [i for i, f in enumerate(features) if f is not None]
    matrix = np.stack([features[i] for i in valid]).astype(np.float32)
    # Conservative grouping for split hygiene, not an identity assertion.
    for start in range(0, len(valid), 256):
        similarity = matrix[start:start+256] @ matrix.T
        a, b = np.where(similarity >= .5)
        for x, y in zip(a+start, b):
            if x < y:
                join(valid[int(x)], valid[int(y)])
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    counts = Counter()
    for split in ('train', 'val', 'expw-val', 'expw-holdout'):
        for label in CLASSES:
            (temp / split / label).mkdir(parents=True, exist_ok=True)
    base = ROOT / 'data/seven-class/glasses-adaptation-v1'
    for split in ('train', 'val'):
        for label in CLASSES:
            for file in sorted((base / split / label).glob('*.jpg')):
                os.link(file, temp / split / label / file.name)
                counts[f'base/{split}/{label}'] += 1
    for members in groups.values():
        group_id = min(rows[i]['sha256'] for i in members)
        bucket = int(group_id[:8],16) % 20
        split = 'expw-holdout' if bucket < 3 else 'expw-val' if bucket < 6 else 'train'
        # No reliable identity grouping for missed faces: training only.
        if any(features[i] is None for i in members):
            split = 'train'
        for i in members:
            row = rows[i]
            row.update(group=group_id, split=split, index=i)
            filename = f"web_expw_{row['sha256'][:24]}.jpg"
            (temp / split / row['label'] / filename).write_bytes(images[i])
            row['path'] = str(Path(split) / row['label'] / filename)
            counts[f'new/{split}/{row["label"]}'] += 1
    report = {'source': 'https://hf-mirror.com/datasets/Mengyuh/ExpW_preprocessed',
              'original_source': 'https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html',
              'label_verification': 'Only uniquely annotated original source faces, exact original class matches.',
              'grouping': 'SFace cosine >=0.5 connected components; 70/15/15 group hash split; misses train-only. Automatic grouping can miss/merge identities; cross-source identity overlap with prior datasets not yet ruled out.',
              'groups': len(groups), 'largest_group': max(map(len, groups.values())),
              'counts': dict(counts), 'excluded': dict(excluded), 'images': rows}
    (temp / 'dataset.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    temp.replace(output)
    # Review only training/validation; do not view final holdout during development.
    review_rows = [r for r in rows if r['split'] != 'expw-holdout']
    for page in range((len(review_rows)+99)//100):
        sheet = Image.new('RGB',(1000,1200),'white')
        draw = ImageDraw.Draw(sheet)
        for n,row in enumerate(review_rows[page*100:(page+1)*100]):
            image = Image.open(output/row['path']).convert('RGB');image.thumbnail((96,96))
            x,y=(n%10)*100,(n//10)*120
            sheet.paste(image,(x,y))
            draw.text((x,y+96),f"{row['index']} {row['label'][:3]} {'V' if row['split']=='expw-val' else 'T'}",fill='black')
        sheet.save(report_dir/f'contact-{page:02d}.jpg',quality=88)
    print(json.dumps({k:v for k,v in report.items() if k!='images'}),flush=True)

if __name__ == '__main__':
    main()
