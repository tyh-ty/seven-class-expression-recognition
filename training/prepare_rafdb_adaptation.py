"""Build a local RAF-DB adaptation set with original test images held out."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
import io

LABELS = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")

def link(source: Path, dest: Path) -> None:
    try: os.link(source, dest)
    except OSError: shutil.copy2(source, dest)

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path(r"D:\RAFDB-source"))
    p.add_argument("--base", type=Path, default=Path("data/seven-class/glasses-adaptation-v1"))
    p.add_argument("--output", type=Path, default=Path("data/seven-class/rafdb-adaptation-v2"))
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    temp=args.output.with_name('.'+args.output.name+'-building')
    if temp.exists(): raise FileExistsError(temp)
    counts=Counter(); seen={}
    try:
        for split in ('train','val','test','glasses-val'):
            for label in LABELS:
                dest=temp/split/label;dest.mkdir(parents=True,exist_ok=True)
                for source in (args.base/split/label).iterdir():
                    if source.is_file():link(source,dest/source.name);counts[f'base/{split}/{label}']+=1
        for split in ('rafdb-val','rafdb-test'):
            for label in LABELS:(temp/split/label).mkdir(parents=True,exist_ok=True)
        files=sorted(args.source.glob('train-*.parquet'))
        if len(files)!=5:raise ValueError('Expected all five complete RAF-DB parquet shards')
        for file in files:
            pf=pq.ParquetFile(file)
            for batch in pf.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    image=row['image'];name=Path(image['path']).name;raw=image['bytes'];label=LABELS[row['label']]
                    if not raw:counts['rejected/bad_image']+=1;continue
                    with Image.open(io.BytesIO(raw)) as img:
                        if img.width < 36 or img.height < 36:counts['rejected/small_image']+=1;continue
                        img.verify()
                    digest=hashlib.sha256(raw).hexdigest()
                    if digest in seen:
                        counts['rejected/exact_duplicate']+=1
                        if seen[digest]!=label:counts['rejected/contradictory_duplicate']+=1
                        continue
                    seen[digest]=label
                    if name.startswith('test_'):
                        split='rafdb-test'
                    elif name.startswith('train_'):
                        split='rafdb-val' if int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'big')%10==0 else 'train'
                    elif name == 'affectnet.zip':
                        # The mirror stores AffectNet images under a placeholder path.
                        # Keep them as a separate mirror-derived training/validation source.
                        split='rafdb-val' if int.from_bytes(hashlib.sha256(digest.encode()).digest()[:4],'big')%10==0 else 'train'
                    else:counts['rejected/unknown_split']+=1;continue
                    dest=temp/split/label/('rafdb_'+digest[:24]+'.jpg')
                    dest.write_bytes(raw)
                    counts[f'rafdb/{split}/{label}']+=1
            print(file.name, dict(counts), flush=True)
        report={
            'source':'https://huggingface.co/datasets/deanngkl/raf-db-7emotions',
            'original_dataset':'Mixed mirror: RAF-DB images indicated by train_/test_ filenames, plus AffectNet images under affectnet.zip placeholder. RAF-DB original test never used for tuning.',
            'license_note':'Mirror has no license statement. Keep all source images and extracted data local; verify original RAF-DB terms before external distribution.',
            'split_note':'RAF-DB original train is split 90/10 by filename; AffectNet placeholder rows 90/10 by content digest. The same identities may occur across splits. RAF-DB test has no neutral support.',
            'counts':dict(sorted(counts.items()))}
        (temp/'rafdb-dataset.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        temp.replace(args.output)
    except Exception:
        print('Incomplete output retained for inspection:',temp,flush=True)
        raise

if __name__=='__main__':main()
