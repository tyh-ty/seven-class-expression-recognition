"""Add visually confirmed training eyewear in an immutable dataset view."""
from collections import Counter
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIRMED = set([39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,114,116,117,118,119,120,121,122,123,124,125,126,127,128,131,132,133,134,135,136,137,138,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,236,237,238,239,241,242,243,245,246,248,249,251,253,256,260,261,262,263,264,265,266,267,269,270,271,273,275,277,278,280,281,282,283,284,286,288,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,553,554,555,557,558,559,560,561,562,563,565,566,568,569,570,571,572,574,575,577,579,580,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,668,669,670,671,674,675,676,677,678,679,680,681,682,684,685,686,687,688,689,691,692,693,694,695,696,697,698,699,700,701,702,704,706,707,708,709,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,797,798,799,800,801,802,803,804,806,807,808,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,830,831,832,833,835,836,837,838,839,840,841,842,843,844,845,846,847,848])


def main():
    work = ROOT / 'training/overnight-20260927/eyewear-mining'
    mined = json.loads((work / 'review-manifest.json').read_text(encoding='utf-8'))
    reviewed = [dict(row, eyewear_review=(
        'confirmed_transparent_eyeglasses' if row['review_index'] in CONFIRMED
        else 'excluded_not_confirmed_ordinary_transparent_eyeglasses'))
        for row in mined['candidates'] if row['split'] == 'train']
    assert CONFIRMED == {r['review_index'] for r in reviewed
                         if r['eyewear_review'] == 'confirmed_transparent_eyeglasses'}
    confirmed_hashes = {r['sha256'] for r in reviewed if r['review_index'] in CONFIRMED}
    val_review = json.loads((work / 'validation-visual-review.json').read_text(encoding='utf-8'))
    val_hashes = {r['sha256'] for r in val_review['images']}
    # A candidate probability is not a positive tag. Avoid adding synthetic glasses
    # over any plausible existing frame, including unreviewed candidates.
    possible_hashes = {r['sha256'] for r in map(json.loads,
        (work / 'scores.jsonl').read_text(encoding='utf-8').splitlines())
        if r['eyeglasses_model_probability'] >= .5}
    source = ROOT / 'data/seven-class/expw-original-eyewear-v2'
    output = ROOT / 'data/seven-class/expw-original-eyewear-v3'
    if output.exists():
        raise FileExistsError(output)
    data = json.loads((source / 'dataset.json').read_text(encoding='utf-8'))
    original_assignments = [(r['path'], r['label'], r['split'], r['sha256'], r['group'])
                            for r in data['images']]
    confirmed_counts = Counter()
    for row in data['images']:
        if row['split'] == 'train' and row['sha256'] in confirmed_hashes:
            row['real_eyewear'] = True
        if row['split'] == 'expw-val' and row['sha256'] in val_hashes:
            row['real_eyewear'] = True
        if row['split'] in ('train', 'expw-val'):
            row['eyewear_possible'] = bool(row.get('eyewear_possible') or
                row['real_eyewear'] or row['sha256'] in possible_hashes)
        if row['real_eyewear']:
            confirmed_counts[row['split'] + '/' + row['label']] += 1
    assert original_assignments == [(r['path'], r['label'], r['split'], r['sha256'], r['group'])
                                    for r in data['images']]
    temp = output.with_name(output.name + '-building')
    temp.mkdir()
    linked = 0
    for path in source.rglob('*.jpg'):
        if path.relative_to(source).parts[0] == 'real-eyeglasses-val':
            continue
        dest = temp / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, dest)
        linked += 1
    for row in data['images']:
        if row['split'] == 'expw-val' and row['real_eyewear']:
            dest = temp / 'real-eyeglasses-val' / row['label'] / Path(row['path']).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(source / row['path'], dest)
    data['source_dataset'] = str(source.relative_to(ROOT))
    data['eyewear_review'] = [str((work / 'training-visual-review.json').relative_to(ROOT)),
                             str((work / 'validation-visual-review.json').relative_to(ROOT))]
    data['counts'] = {k: v for k, v in data['counts'].items() if not k.startswith('confirmed_eyewear/')}
    data['counts'].update({'confirmed_eyewear/' + k: v for k, v in confirmed_counts.items()})
    data['eyewear_tag_revision'] = 'Appearance tags only; expressions, identities and splits unchanged. Ordinary transparent frames visually confirmed; uncertain candidates only prevent synthetic frame overlay. Final holdout not viewed.'
    (temp / 'dataset.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    report = {
        'scope': 'Assistant reviewed all632 training candidates at224px per face using training-00..25.jpg. Ordinary transparent glasses include mild tint/reflection and lowered reading glasses. Excluded opaque sunglasses, novelty/protective eyewear, no-glasses, ambiguous or heavily stylized cases, and ambiguous multi-face collages. No expression labels changed or predictions consulted.',
        'limits': 'Appearance review is not independent expert annotation. Some posed, repeated, watermarked and stylized photos remain. These tags are for training sampling, not proof of webcam accuracy.',
        'source_dataset': str(source.relative_to(ROOT)), 'output_dataset': str(output.relative_to(ROOT)),
        'reviewed': len(reviewed), 'confirmed_new_train': len(CONFIRMED),
        'confirmed_counts': dict(confirmed_counts), 'linked_images': linked,
        'assignments_unchanged': True, 'holdout_viewed': False,
        'reviewed_candidates': reviewed}
    (temp / 'eyewear-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    temp.replace(output)
    (work / 'training-visual-review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'reviewed_candidates'}), flush=True)


if __name__ == '__main__':
    main()
