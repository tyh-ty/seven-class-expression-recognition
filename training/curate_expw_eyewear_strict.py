"""Stricter visible-eyeglasses re-review at 224 pixels per face; no expression relabeling."""
from collections import Counter
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIRMED_TRAIN=set([13,20,31,45,55,67,131,137,138,223,245,280,287,306,324,387,396,402,408,422,443,503,509,518,537,545,551,595,609,626,686,697,701,727,736,752,762,767,770,819,822,893,953,955,970,1019,1038,1047,1057,1068,1104,1109,1116,1124,1131,1153,1193,1209,1216,1241,1250,1251,1269,1284,1296,1304,1320,1323,1444,1451,1509,1529,1556,1598,1603,1635,1644,1646,1651,1661,1689,1790,1791,1840,1900,1904,1908,1911,1941,1943,2050,2060,2085,2097,2098,2154,2165,2167,2189,2199,2202,2218,2228,2259,2326,2332,2334,2359])
CONFIRMED_VAL=set([4,59,96,188,278,435,488,745,891,928,943,999,1106,1125,1256,1363,1410,1435,1716,1794,1944,2143,2220,2260,2276,2296])

def main():
    previous=json.loads((ROOT/'training/overnight-20260927/expw-eyewear-review.json').read_text(encoding='utf-8'))
    confirmed=CONFIRMED_TRAIN|CONFIRMED_VAL
    source=ROOT/'data/seven-class/expw-subset-v1'
    output=ROOT/'data/expw/real-eyeglasses-val-v2'
    if output.exists():raise FileExistsError(output)
    for label in ('anger','disgust','fear','happiness','neutral','sadness','surprise'):(output/label).mkdir(parents=True,exist_ok=True)
    rows=[];counts=Counter()
    for old in previous['images']:
        row=dict(old)
        row['previous_eyewear_tag']=old['eyewear']
        if row['index'] in confirmed:row['eyewear']='clear_eyeglasses'
        elif old['eyewear']=='sunglasses_or_novelty':row['eyewear']='sunglasses_or_novelty'
        else:row['eyewear']='not_confirmed_clear_eyeglasses'
        counts[row['split']+'/'+row['eyewear']+'/'+row['label']]+=1
        rows.append(row)
        if row['index'] in CONFIRMED_VAL:
            assert row['split']=='expw-val'
            os.link(source/row['path'],output/row['label']/Path(row['path']).name)
        if row['index'] in CONFIRMED_TRAIN:assert row['split']=='train'
    assert confirmed=={r['index'] for r in rows if r['eyewear']=='clear_eyeglasses'}
    report={'method':'Assistant re-reviewed all210 previously clear-tagged train/validation faces at224px per face. Retained only visibly confirmed ordinary transparent eyeglasses, including light tint/reflections and lowered reading glasses. Ambiguous, heavily blurred, no-glasses and dark/novelty cases excluded from strict cohort. Expression labels never changed; no predictions used.',
            'correction':'Earlier small-contact-sheet tagging included definite no-glasses faces. The old39-image glasses metrics are exploratory and must not be described as accuracy on a verified real-eyeglasses cohort. Original files/reports retained for audit.',
            'limits':'Assistant appearance review, not independent expert annotation. Still a small public-photo validation subset, not laboratory webcam ground truth. Posed/stylized images remain; external pretraining overlap unknown.',
            'holdout_inspected':False,'confirmed_training':len(CONFIRMED_TRAIN),'confirmed_validation':len(CONFIRMED_VAL),'counts':dict(counts),'images':rows}
    (ROOT/'training/overnight-20260927/expw-eyewear-review-v2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('images','counts')}),flush=True)

if __name__=='__main__':main()
