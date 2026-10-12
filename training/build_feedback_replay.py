"""Add three existing labeled camera corrections to training only; exclude smoke duplicates."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

CLASSES=('anger','disgust','fear','happiness','neutral','sadness','surprise')
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/seven-class/expw-enterface-v1'
OUT=ROOT/'data/seven-class/expw-enterface-feedback-v1'


def main():
    temporary=OUT.with_name(OUT.name+'-building')
    for path in (OUT,temporary):
        if not path.resolve().is_relative_to(ROOT/'data/seven-class'):
            raise ValueError('Output escapes intended dataset workspace')
        if path.exists():raise FileExistsError(path)
    source=ROOT/'data/custom/feedback/manifest.jsonl'
    feedback=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    def digest(row):return hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()
    smoke_hashes={digest(r) for r in feedback if r['session'].startswith('_smoke')}
    examples=[r for r in feedback if not r['session'].startswith('_smoke') and digest(r) not in smoke_hashes]
    if len(examples)!=3 or any(r['session']!='natural_20260918' for r in examples):
        raise ValueError('Re-audit unexpected local feedback collection before training')
    metadata=json.loads((BASE/'dataset.json').read_text())
    rows=[dict(r) for r in metadata['images'] if r['split']=='train']
    for split in ('train','val'):
        for label in CLASSES:
            target=temporary/split/label;target.mkdir(parents=True,exist_ok=True)
            for file in (BASE/split/label).iterdir():
                if file.suffix.lower() in {'.jpg','.jpeg','.png'}:os.link(file,target/file.name)
    new=[]
    for row in examples:
        sha=digest(row)
        name='web_feedback_local_'+sha[:24]+'.jpg'
        path=Path('train')/row['label']/name
        os.link(ROOT/row['file'],temporary/path)
        entry={'split':'train','label':row['label'],'path':str(path),'source':row['file'],
               'sha256':sha,'group':'local-feedback-'+row['session'],'real_eyewear':False,'eyewear_possible':False,
               'label_scope':'Existing application user-correction label. Three correlated frames, no unseen personal-session validation.',
               'training_only':True,'previous_diagnostic_retired_for_this_candidate':True}
        rows.append(entry);new.append(entry)
    report={'source_dataset':str(BASE),'feedback_manifest':str(source),'feedback_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'images':rows,'new_unique_training_images':len(new),'new_training_records':new,
            'counts':{split:dict(Counter(p.parent.name for p in (temporary/split).glob('*/*') if p.is_file())) for split in ('train','val')},
            'limit':'Only three nonglasses localcorrections. Deliberate training/calibration use, never independent accuracy. Existing publicvalidation/holdout splits unchanged. Smoke duplicates excluded.'}
    (temporary/'dataset.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    temporary.replace(OUT)
    print(json.dumps({k:v for k,v in report.items() if k!='images'}),flush=True)


if __name__=='__main__':main()
