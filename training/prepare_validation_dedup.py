"""Create an exact-content-disjoint primary validation view without changing source files."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=ROOT/'training/runs/vit-base-expw-eyewear-v1/dataset-manifest.jsonl'
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    training_hashes={r['sha256'] for r in rows if r['role']=='train'}
    validation=[r for r in rows if r['role']=='validation']
    grouped=defaultdict(list)
    for row in validation:grouped[row['sha256']].append(row)
    output=ROOT/'data/ferplus/validation-dedup-v1'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    classes=('anger','disgust','fear','happiness','neutral','sadness','surprise')
    for label in classes:(output/label).mkdir()
    kept=[];excluded=[];counts=Counter()
    for digest,members in grouped.items():
        labels={Path(r['path']).parent.name for r in members}
        if digest in training_hashes:reason='exact_content_seen_in_training'
        elif len(labels)>1:reason='contradictory_validation_labels'
        else:reason=None
        if reason:
            excluded.extend(dict(row,reason=reason) for row in members)
            counts[reason]+=len(members)
            continue
        row=members[0];source=Path(row['root'])/row['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        os.link(source,output/row['path']);kept.append(row)
        counts['kept/'+Path(row['path']).parent.name]+=1
        for duplicate in members[1:]:excluded.append(dict(duplicate,reason='within_validation_duplicate'))
        counts['within_validation_duplicate']+=len(members)-1
    report={'scope':'Validation-only audit view, selected solely by source labels and byte hashes, not predictions. No source files changed. This is not a new test set and does not remove possible pretraining, identity, or perceptual overlaps.',
            'source_manifest':str(manifest.relative_to(ROOT)), 'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'original_validation_examples':len(validation),'retained_examples':len(kept),'counts':dict(counts),
            'kept':kept,'excluded':excluded}
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'training/overnight-20260927/ferplus-duplicate-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('kept','excluded')}),flush=True)

if __name__=='__main__':main()
