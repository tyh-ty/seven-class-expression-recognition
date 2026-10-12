"""Create an immutable full-ExpW view with stricter eyewear and no mixed prior split groups."""
from collections import Counter,defaultdict
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CLASSES=('anger','disgust','fear','happiness','neutral','sadness','surprise')

def main():
    source=ROOT/'data/seven-class/expw-original-v1'
    output=ROOT/'data/seven-class/expw-original-eyewear-v2'
    if output.exists():raise FileExistsError(output)
    original=json.loads((source/'dataset.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'data/seven-class/expw-subset-v1/dataset.json').read_text(encoding='utf-8'))
    fixed={r['source']:r for r in old['images']}
    review=json.loads((ROOT/'training/overnight-20260927/expw-eyewear-review-v2.json').read_text(encoding='utf-8'))
    clear={r['source'] for r in review['images'] if r['eyewear']=='clear_eyeglasses'}
    possible={r['source'] for r in review['images']}
    groups=defaultdict(list)
    for row in original['images']:groups[row['group']].append(row)
    conflicts=set()
    for group,members in groups.items():
        prior={fixed[name]['split'] for row in members for name in row['source_aliases'] if name in fixed}
        if len(prior)>1:conflicts.add(group)
    temp=output.with_name(output.name+'-building');temp.mkdir()
    for split in ('train','val','expw-val','expw-holdout','real-eyeglasses-val'):
        for label in CLASSES:(temp/split/label).mkdir(parents=True)
    counts=Counter()
    # Replay files are immutable links to the same prior data, except the new ExpW source.
    for split in ('train','val'):
        for path in (source/split).glob('*/*.jpg'):
            if path.name.startswith('web_expw_full_'):continue
            os.link(path,temp/path.relative_to(source));counts['base/'+split+'/'+path.parent.name]+=1
    rows=[];excluded=[]
    for old_row in original['images']:
        row=dict(old_row)
        if row['group'] in conflicts:
            excluded.append({'source':row['source'],'sha256':row['sha256'],'group':row['group'],'previous_split':row['split'],
                             'reason':'Group merges sources from earlier training, validation and/or holdout; cannot form an independent evaluation cohort.'})
            continue
        row['real_eyewear']=bool(set(row['source_aliases'])&clear)
        row['eyewear_possible']=bool(set(row['source_aliases'])&possible)
        destination=temp/row['path'];os.link(source/row['path'],destination)
        if row['split']=='expw-val' and row['real_eyewear']:
            os.link(destination,temp/'real-eyeglasses-val'/row['label']/destination.name)
        counts['new/'+row['split']+'/'+row['label']]+=1
        if row['real_eyewear']:counts['confirmed_eyewear/'+row['split']+'/'+row['label']]+=1
        rows.append(row)
    report={k:v for k,v in original.items() if k not in ('images','counts','groups','largest_group')}
    report.update(source_dataset=str(source.relative_to(ROOT)),counts=dict(counts),images=rows,
                  quarantined_prior_mixed_groups=len(conflicts),quarantined_images=len(excluded),
                  split_revision='All groups touching multiple previous subsets are excluded from this view. This prevents old training faces becoming evaluation examples through later group merges. Source v1 remains untouched.',
                  eyewear_review='training/overnight-20260927/expw-eyewear-review-v2.json',
                  primary_validation_override='data/ferplus/validation-dedup-v1')
    (temp/'quarantined-mixed-prior-groups.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    (temp/'dataset.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    temp.replace(output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('images','counts')}),flush=True)
    print(json.dumps(dict(counts)),flush=True)

if __name__=='__main__':main()
