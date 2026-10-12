"""Render a fixed, failure-inclusive validation selection using actual UI display traces."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_caer_pilot import camera_frame

LABELS = dict(zip(('anger','disgust','fear','happiness','neutral','sadness','surprise'),
                  ('愤怒','厌恶','恐惧','开心','中性','悲伤','惊讶')))
WIDTH, HEIGHT = 1120, 680


def load_reports(paths):
    videos, displays = {}, {}
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        shown = json.loads(path.with_stem(path.stem+'-display').read_text(encoding='utf-8'))
        for video in data['videos']:
            if video['id'] in videos:
                raise ValueError('Duplicate input video')
            videos[video['id']] = video
        displays.update({v['id']:{f['index']:f['displayed'] for f in v['frames']} for v in shown['clips']})
    return videos, displays


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selection',type=Path,default=Path('output/overnight-validation-20260928/comparison-selection.json'))
    p.add_argument('--base-report',type=Path,action='append',required=True)
    p.add_argument('--candidate-report',type=Path,required=True)
    p.add_argument('--candidate-label',default='新版候选')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true',help='Render just the first fixed clip to verify the renderer.')
    args = p.parse_args()
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    manifest = Path(selection['manifest'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != selection['manifest_sha256']:
        raise ValueError('Fixed montage manifest changed')
    original = {v['id']:v for v in json.loads(manifest.read_text())['videos']}
    baseline, base_display = load_reports(args.base_report)
    candidate, new_display = load_reports([args.candidate_report])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    heading = ImageFont.truetype(font_path, 26)
    normal = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 17)
    prediction_font = ImageFont.truetype(font_path, 32)
    cv2.setNumThreads(1)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner','-loglevel','error','-y',
               '-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}','-r','10','-i','pipe:0',
               '-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',
               '-threads','2','-movflags','+faststart',str(args.output)]
    frame_count = 0
    with args.output.with_suffix('.encode.log').open('wb') as log:
        encoder = subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            selected_ids = selection['video_ids'][:1] if args.smoke else selection['video_ids']
            for vi, video_id in enumerate(selected_ids,1):
                row = original[video_id]
                base = {f['index']:f for f in baseline[video_id]['frames']}
                new = {f['index']:f for f in candidate[video_id]['frames']}
                if set(base) != set(new):
                    raise ValueError('Comparison frame indices differ')
                temporary = None
                if 'archive' in row:
                    temporary = args.output.parent/(video_id+'.render-source.mp4')
                    with ZipFile(row['archive']) as archive:
                        temporary.write_bytes(archive.read(row['entry']))
                    path = temporary
                else:
                    path = Path(row['path'])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != row['source_sha256']:
                        raise ValueError('Source video hash changed')
                capture = cv2.VideoCapture(str(path))
                try:
                    index = 0
                    while capture.grab():
                        if index not in base:
                            index += 1
                            continue
                        ok, decoded = capture.retrieve()
                        if not ok:
                            raise ValueError('Cannot decode selected frame')
                        a,b = base[index],new[index]
                        for field in ('seconds','width','height','bbox','status'):
                            if a.get(field) != b.get(field):
                                raise ValueError('Candidate uses different frame/detector selection')
                        frame = camera_frame(decoded,max_side=512)
                        image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                        image.thumbnail((520,430),Image.Resampling.LANCZOS)
                        canvas = Image.new('RGB',(WIDTH,HEIGHT),(17,24,39))
                        draw = ImageDraw.Draw(canvas)
                        draw.text((24,16),'固定验证视频回放 · 10 帧/秒',font=heading,fill='white')
                        source_name = ('RAVDESS 正常强度' if row['source']=='RAVDESS-val'
                                       else f'eNTERFACE 戴眼镜人物 {row["subject"]}')
                        draw.text((24,59),f'{vi}/19  {source_name}   片段原标签：{LABELS[row["label"]]}',font=normal,fill=(190,204,222))
                        for x,label,shown in [(24,'原版本',base_display[video_id][index]),
                                               (576,args.candidate_label,new_display[video_id][index])]:
                            draw.text((x,104),label,font=normal,fill=(127,205,247))
                            px=x+(520-image.width)//2
                            py=150+(430-image.height)//2
                            canvas.paste(image,(px,py))
                            if a.get('bbox'):
                                bb=a['bbox'];sx=image.width/a['width'];sy=image.height/a['height']
                                draw.rectangle((px+bb['x']*sx,py+bb['y']*sy,
                                                px+(bb['x']+bb['width'])*sx,py+(bb['y']+bb['height'])*sy),outline=(80,220,230),width=2)
                            draw.text((x,586),'显示：'+LABELS.get(shown,'判断中'),font=prediction_font,fill='white')
                        draw.text((24,636),'受控表演 / 情绪语音；片段标签不等于每一帧的表情真值。保留错误与不确定结果。',font=small,fill=(165,175,192))
                        if frame_count == 0:
                            canvas.save(args.output.with_suffix('.png'))
                        encoder.stdin.write(np.asarray(canvas).tobytes())
                        frame_count += 1
                        index += 1
                finally:
                    capture.release()
                    if temporary:
                        temporary.unlink(missing_ok=True)
                print(json.dumps({'clips_rendered':vi,'frames':frame_count}),flush=True)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('Video encoder failed')
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise
    report = {'output':str(args.output),'frames':frame_count,'fps':10,'seconds':frame_count/10,
              'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'selection':str(args.selection),
              'input_reports':[str(p) for p in [*args.base_report,args.candidate_report]],
              'input_report_hashes':[hashlib.sha256(p.read_bytes()).hexdigest() for p in [*args.base_report,args.candidate_report]],
              'smoke':args.smoke,'clips':len(selected_ids),
              'scope':'Side-by-side actual frontend display traces on fixedclips; includes failures. Not physicalcamera capture or an independent accuracyestimate.'}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
