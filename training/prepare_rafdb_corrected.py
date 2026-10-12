"""Build RAF basic from independently checked 1..7 labels; exclude corrupt mirror labels."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')
MAPPING = {1: 'surprise', 2: 'fear', 3: 'disgust', 4: 'happiness', 5: 'sadness', 6: 'anger', 7: 'neutral'}

def main():
    output = ROOT / 'data/seven-class/rafdb-corrected-v1'
    if output.exists():
        raise FileExistsError(output)
    building = output.with_name(output.name + '-building')
    building.mkdir()
    base = ROOT / 'data/seven-class/glasses-adaptation-v1'
    archive = Path('D:/RAFDB-source/canonical-audit.zip')
    counts = Counter()
    rows = []
    for split in ('train', 'val', 'rafdb-val', 'rafdb-test'):
        for label in CLASSES:
            (building / split / label).mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        for label in CLASSES:
            for source in sorted((base / split / label).glob('*.jpg')):
                os.link(source, building / split / label / source.name)
                counts[f'base/{split}/{label}'] += 1
    # Use the existing training manifest to exclude exact cross-source leakage.
    manifest = ROOT / 'training/runs/vit-small-seven-caer-reviewed-v1/dataset-manifest.jsonl'
    base_hashes = {json.loads(line)['sha256'] for line in manifest.read_text(encoding='utf-8').splitlines()}
    seen = set()
    with ZipFile(archive) as z:
        labels = {}
        for split in ('train', 'test'):
            raw = z.read(f'{split}_labels.csv')
            parsed = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
            for row in parsed:
                labels[row['image']] = MAPPING[int(row['label'])]
        assert len(labels) == 15339
        for name in sorted(z.namelist()):
            if not name.endswith('.jpg'):
                continue
            filename = Path(name).name
            label = labels[filename]
            assert MAPPING[int(Path(name).parent.name)] == label
            raw = z.read(name)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen or digest in base_hashes:
                counts['excluded/exact_duplicate'] += 1
                continue
            seen.add(digest)
            if filename.startswith('test_'):
                split = 'rafdb-test'
            else:
                split = 'rafdb-val' if int.from_bytes(hashlib.sha256(filename.encode()).digest()[:4], 'big') % 10 == 0 else 'train'
            destination = building / split / label / f'rafdb_corrected_{digest[:24]}.jpg'
            destination.write_bytes(raw)
            counts[f'rafdb/{split}/{label}'] += 1
            rows.append({'source_entry': name, 'label': label, 'split': split, 'sha256': digest,
                         'path': str(destination.relative_to(building))})
    for split in ('train', 'rafdb-val', 'rafdb-test'):
        assert all(counts[f'rafdb/{split}/{label}'] > 0 for label in CLASSES)
    report = {'source': 'https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset',
              'original_source': 'http://www.whdeng.cn/RAF/model1.html',
              'label_mapping': MAPPING, 'counts': dict(counts), 'images': rows,
              'excluded': 'No AffectNet placeholder rows or corrupted deanngkl labels used.',
              'split': 'Original RAF test held out; original train hash split 90/10; identity independence NOT established. Exact duplicates excluded across this source and prior dataset manifest.',
              'audit': 'training/overnight-20260927/raf-label-audit.json',
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (building / 'dataset.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    building.replace(output)
    print(json.dumps({'output': str(output), 'counts': dict(counts)}), flush=True)

if __name__ == '__main__':
    main()
