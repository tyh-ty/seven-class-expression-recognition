"""Create an expanded eyewear validation view after visual candidate review, no emotion changes."""
from collections import Counter
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIRMED=set([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,36,37,139,140,141,142,143,144,145,147,148,149,151,152,153,155,257,258,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,308,309,310,311,313,314,316,317,318,319,320,321,322,323,324,325,329,334,335,336,337,338,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,462,463,464,467,469,470,471,472,475,476,479,581,582,583,584,585,586,587,588,589,590,591,592,594,595,596,597,598,600,602,604,605,606,607,608,609,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,727,728,729,730,731,733,734,735,741,742,743])

def main():
    work=ROOT/'training/overnight-20260927/eyewear-mining'
    mined=json.loads((work/'review-manifest.json').read_text(encoding='utf-8'))
    reviewed=[dict(r,eyewear_review='confirmed_transparent_eyeglasses' if r['review_index'] in CONFIRMED else 'excluded_not_confirmed_ordinary_transparent_eyeglasses') for r in mined['candidates'] if r['split']=='expw-val']
    assert CONFIRMED=={r['review_index'] for r in reviewed if r['eyewear_review']=='confirmed_transparent_eyeglasses'}
    dataset=ROOT/'data/seven-class/expw-original-eyewear-v2'
    metadata=json.loads((dataset/'dataset.json').read_text(encoding='utf-8'))
    hashes={r['sha256'] for r in reviewed if r['review_index'] in CONFIRMED}
    rows=[r for r in metadata['images'] if r['split']=='expw-val' and (r['real_eyewear'] or r['sha256'] in hashes)]
    output=ROOT/'data/expw/real-eyeglasses-original-expanded-val-v1'
    if output.exists():raise FileExistsError(output)
    for label in ('anger','disgust','fear','happiness','neutral','sadness','surprise'):(output/label).mkdir(parents=True)
    for row in rows:os.link(dataset/row['path'],output/row['label']/Path(row['path']).name)
    report={'scope':'Assistant visually reviewed all217 additional validation candidates at224px per face in validation-00..06.jpg. Kept visible ordinary transparent glasses, including mild tint/reflection and lowered reading glasses. Excluded opaque sunglasses,3D/novelty/protective eyewear, non-glasses, ambiguous dark/blurred cases and ambiguous two-face crops. Labels remain original expressions. No expression predictions used to select images.',
            'limits':'Validation only, not independent expert labels or webcam truth. Includes posed/stylized photos and repeated identity groups. Fear remains very small. Mining model may miss other glasses. Final holdout not viewed.',
            'confirmed_new':len(CONFIRMED),'prior_confirmed':len(rows)-len(CONFIRMED),'total':len(rows),
            'counts':dict(Counter(r['label'] for r in rows)),'unique_groups':len({r['group'] for r in rows}),
            'reviewed_candidates':reviewed,'images':rows}
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (work/'validation-visual-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('reviewed_candidates','images')}),flush=True)

if __name__=='__main__':main()
