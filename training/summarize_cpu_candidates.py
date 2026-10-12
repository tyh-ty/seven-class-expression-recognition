"""Compact, provenance-preserving summary of actual deployment-graph experiments."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'training/overnight-20260927'
NAMES = ('selective-int8', 'selective-fp32', 'enterface2-int8', 'enterface2-fp32',
         'enterface6-fp32', 'enterface6-channel-int8', 'enterface6-retained75-fp32',
         'enterface2-6-average-fp32', 'rgb-small-best-fp32')


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    output = {'updated_at': datetime.now(timezone.utc).isoformat(), 'candidates': {},
              'scope': 'Actual CPU graphs, repeated validation used for selection. Photo class figures are recall, not precision. Clip aggregate and half-second display metrics remain distinct. Holdout not read by this summarizer.',
              'limits': 'ExpW internet photos; controlled RAV/eNTER speech/acting, not guaranteed webcam/micro-expression accuracy. Eyewear photo fear has only7 examples; video eyewear subjects are9/13 and correlated.'}
    for name in NAMES:
        photos = read(WORK/(name+'-photos.json'))
        rubric = read(WORK/(name+'-photos-rubric.json'))
        directory = WORK/('video-'+name+'-10fps')
        intensity = read(WORK/(name+'-intensity.json'))
        eyewear = read(WORK/(name+'-eyewear.json'))
        if name == 'selective-int8':
            intensity = read(WORK/'video-selective-int8-intensity.json')
            eyewear = read(WORK/'video-selective-int8-eyewear.json')
        row = {'pipeline': read(WORK/(name+'-pipeline.json')), 'variants': {}}
        if photos:
            graph_key = photos['sources'][1]['file']
            key_map = {'baseline': photos['baseline_key'], 'raw': graph_key,
                       'mix-0.5': 'onnx-ensemble-0.5', 'mix-0.75': 'onnx-ensemble-0.75'}
            for tag, key in key_map.items():
                photo = photos['reports'][key]
                item = {'graph_sources': photos['sources'], 'photos': {}}
                for domain, values in photo.items():
                    item['photos'][domain] = {metric: values[metric] for metric in ('examples', 'accuracy', 'macro_f1', 'per_class')}
                if rubric:
                    assessed = next(r for r in rubric['evaluated'] if r['checkpoint'] == key)
                    item['photo_rule_pass'] = assessed['photo_rule_pass']
                    item['failed_photo_checks'] = [k for k, v in assessed['checks'].items() if not v]
                row['variants'][tag] = item
        for tag, filename in [('baseline','baseline-int8'), ('raw',name),
                              ('mix-0.5','ensemble-0.5'), ('mix-0.75','ensemble-0.75')]:
            path = directory/(filename+'.json')
            video = read(path)
            if not video:
                continue
            item = row['variants'].setdefault(tag, {})
            item['videos'] = video['metrics']
            item['video_runtime'] = {k:v for k,v in video.items() if 'latency' in k}
            if intensity:
                r = next((r for r in intensity['reports'].values() if r['model'] == filename), None)
                if r:
                    item['rav_all'] = r['all']
                    item['rav_normal'] = r['normal_intensity']
            if eyewear:
                r = next((r for r in eyewear['reports'].values() if r['model'] == filename), None)
                if r:
                    item['enterface_groups'] = r['groups']
                    item['enterface_eyewear_subjects'] = {key:r['per_subject'][key] for key in ('9','13')}
        output['candidates'][name] = row
    (WORK/'cpu-candidate-matrix.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    for name, row in output['candidates'].items():
        raw = row['variants'].get('raw', {})
        print(json.dumps({'name': name, 'status': (row['pipeline'] or {}).get('status', 'see individual reports'),
            'photo_pass': raw.get('photo_rule_pass'),
            'photos': {d:{k:r[k] for k in ('accuracy','macro_f1')} for d,r in raw.get('photos', {}).items() if d in ('ferplus_clean','expw_original','real_eyeglasses_expanded')},
            'rav_stable': raw.get('rav_all', {}).get('stable_correct_half_second_clip_rate'),
            'enterface_eye_fear': raw.get('enterface_groups', {}).get('eyewear', {}).get('per_class', {}).get('fear')}))


if __name__ == '__main__':
    main()
