"""Record assistant visual eyewear review; never infer emotion labels from predictions."""
from __future__ import annotations
import json
import os
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
# Reviewed contact sheets 00..20. Only training/validation were displayed.
VISIBLE={
4,13,20,31,45,55,59,67,82,96,97,
131,137,138,169,179,188,194,202,223,
243,245,278,280,287,306,308,324,
375,385,387,392,396,402,408,422,431,432,435,443,
477,483,484,488,489,496,503,507,509,518,529,537,545,549,551,552,562,573,575,
595,596,598,609,626,665,667,675,679,686,695,697,701,
724,727,735,736,743,745,752,762,767,770,774,793,800,819,822,
891,893,895,928,
943,950,953,955,956,970,999,1019,1038,1043,1044,1047,
1057,1058,1068,1074,1075,1104,1106,1109,1116,1124,1125,1131,1140,1153,1155,
1193,1209,1216,1241,1250,1251,1256,
1269,1284,1289,1296,1304,1320,1321,1323,1339,1345,1363,1364,1365,
1404,1410,1418,1424,1435,1444,1451,1463,1465,1472,
1509,1519,1529,1535,1542,1548,1556,1567,1586,1598,1603,1611,1623,
1635,1644,1646,1647,1650,1651,1654,1661,1675,1677,1680,1689,1694,1699,1716,
1743,1790,1791,1794,1803,1823,1840,
1887,1890,1894,1895,1900,1904,1908,1911,1941,1943,1944,
1969,1983,1993,2033,2034,2036,2041,2042,2050,2060,2069,2078,
2085,2097,2098,2108,2123,2124,2135,2143,2154,2165,2167,2189,
2198,2199,2202,2203,2209,2218,2220,2222,2228,2259,2260,2264,2276,2287,2296,2299,2309,2313,
2326,2329,2332,2334,2335,2337,2340,2359,2374,2375,
}
SUNGLASSES_OR_NOVELTY={385,392,477,507,596,598,895,1074,1075,1140,1155,1289,1339,1345,1364,1365,1463,1472,1542,1586,1623,1650,1654,1677,2069,2078,2108,2123,2198,2209,2222,2299,2309,2329,2340,2375}

def main():
    base=ROOT/'data/seven-class/expw-subset-v1'
    data=json.loads((base/'dataset.json').read_text(encoding='utf-8'))
    output=ROOT/'data/expw/real-eyeglasses-val-v1'
    output.mkdir(parents=True,exist_ok=True)
    for label in ('anger','disgust','fear','happiness','neutral','sadness','surprise'):
        (output/label).mkdir(exist_ok=True)
    counts=Counter();selected=[]
    for row in data['images']:
        if row['index'] not in VISIBLE:
            continue
        assert row['split']!='expw-holdout'
        eyewear='sunglasses_or_novelty' if row['index'] in SUNGLASSES_OR_NOVELTY else 'clear_eyeglasses'
        selected.append({**row,'eyewear':eyewear})
        counts[f"{row['split']}/{eyewear}/{row['label']}"]+=1
        if row['split']=='expw-val' and eyewear=='clear_eyeglasses':
            dest=output/row['label']/Path(row['path']).name
            if not dest.exists():os.link(base/row['path'],dest)
    assert len(selected)==len(VISIBLE)
    report={'method':'Assistant visual review of all 2048 train/validation crops, before model predictions; apparent eyewear tags only. Not independently human verified. Original expression labels unchanged.',
            'limits':'Some frames blurred or stylized; appearance tags may have mistakes. Public photo subset, not recorded webcam footage. Small per-class validation supports.',
            'holdout_inspected':False,'counts':dict(counts),'images':selected}
    p=ROOT/'training/overnight-20260927/expw-eyewear-review.json'
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected':len(selected),'counts':dict(counts)}),flush=True)

if __name__=='__main__':main()
