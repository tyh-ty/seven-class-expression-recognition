"""Blind contact sheets for fixed facial-expression review, separate from scene labels."""
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = ROOT / 'data/video-validation-20260927'
    source = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    clips = [row for row in source['videos'] if row['source'] == 'CAER-val']
    random.Random(20260928).shuffle(clips)
    work = ROOT / 'training/overnight-20260927/caer-face-review'
    work.mkdir(exist_ok=False)
    rows = []
    for clip in clips:
        for fraction in (.25, .5, .75):
            frame = min(clip['frames'], key=lambda f: abs(f.get('fraction', -1) - fraction))
            rows.append({'review_index': len(rows), 'video_id': clip['id'],
                         'frame_index': frame['index'], 'requested_fraction': fraction,
                         'seconds': frame['seconds'], 'crop': frame.get('crop'),
                         'status': frame['status']})
    for page, start in enumerate(range(0, len(rows), 15)):
        batch = rows[start:start + 15]
        canvas = Image.new('RGB', (672, 1240), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(batch):
            x, y = index % 3 * 224, index // 3 * 248
            if row['crop']:
                with Image.open(cache / row['crop']) as image:
                    canvas.paste(image.convert('RGB').resize((224, 224)), (x, y))
            draw.text((x + 3, y + 226), str(row['review_index']), fill='black')
        canvas.save(work / f'review-{page:02d}.jpg', quality=97)
    report = {'scope': 'Fixed three chronological frames at25/50/75% from all90 CAER validation clips, randomized clip order. Sheet labels contain only indices, not scene labels or model predictions. Intended for exploratory assistant appearance annotation, not expert ground truth.',
              'rules': 'Annotate visible facial expression independently per frame as one of seven classes, uncertain, or no_face. Do not force a source scene label. Retain uncertain cases and report annotation coverage. No training use.',
              'prediction_files_read': False, 'rows': rows}
    (work / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Created', len(rows), 'fixed review frames from', len(clips), 'clips')


if __name__ == '__main__':
    main()
