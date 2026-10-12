"""Mine candidate eyewear for visual review, without relabeling expressions or reading holdout."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from transformers import AutoModelForImageClassification,ViTImageProcessor

ROOT=Path(__file__).resolve().parents[1]
DEFINITE_NO_GLASSES={82,243,484,496,562,774,1675,2033,97,169,179,194,202,308,431,432,483,573,575,665,667,675,679,950,956,1043,1044,1058,1321,1404,1535,1548,1611,1647,1694,1743,1803,1823,1895,1969,1983,1993,2034,2041,2042,2124,2135,2287,2337}

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=48);args=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path=ROOT/'models/research/eyeglasses-vit'
    provenance=json.loads((model_path/'source.json').read_text(encoding='utf-8'))
    config=json.loads((model_path/'config.json').read_text(encoding='utf-8'))
    assert config['id2label']=={'0':'glasses','1':'no%20glasses'}
    processor=ViTImageProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForImageClassification.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval()
    output=ROOT/'training/overnight-20260927/eyewear-mining';output.mkdir(exist_ok=True)

    def predict(records,root):
        results=[]
        with torch.inference_mode():
            for start in range(0,len(records),args.batch_size):
                batch=records[start:start+args.batch_size];images=[]
                for row in batch:
                    with Image.open(root/row['path']) as image:images.append(image.convert('RGB'))
                inputs=processor(images=images,return_tensors='pt').to(device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    logits=model(**inputs).logits
                scores=torch.softmax(logits.float(),dim=-1)[:,0].cpu().tolist()
                results.extend(dict(row,eyeglasses_model_probability=score) for row,score in zip(batch,scores))
                if start and start%2400==0:print('screened',start,'/',len(records),flush=True)
        return results

    # Screening sanity check: revised visible positives and only definite no-glasses
    # negatives from the same large-face review. Not an independent benchmark.
    previous=json.loads((ROOT/'training/overnight-20260927/expw-eyewear-review-v2.json').read_text(encoding='utf-8'))
    sanity_rows=[dict(r,visible_eyewear=int(r['eyewear']=='clear_eyeglasses')) for r in previous['images']
                 if r['eyewear']=='clear_eyeglasses' or r['index'] in DEFINITE_NO_GLASSES]
    sanity=predict(sanity_rows,ROOT/'data/seven-class/expw-subset-v1')
    matrix=np.zeros((2,2),dtype=int)
    for row in sanity:matrix[row['visible_eyewear'],int(row['eyeglasses_model_probability']>=.5)]+=1
    (output/'sanity.json').write_text(json.dumps({'scope':'Exploratory eyewear screening sanity check against assistant appearance tags, not independent ground truth or expression recognition accuracy.',
                                               'confusion_no_glasses_glasses':matrix.tolist(),'records':sanity},indent=2),encoding='utf-8')
    dataset=ROOT/'data/seven-class/expw-original-eyewear-v2'
    data=json.loads((dataset/'dataset.json').read_text(encoding='utf-8'))
    rows=[{k:r[k] for k in ('index','source','label','split','path','group','sha256','real_eyewear')} for r in data['images'] if r['split'] in ('train','expw-val')]
    results=predict(rows,dataset)
    with (output/'scores.jsonl').open('w',encoding='utf-8') as handle:
        for row in results:handle.write(json.dumps(row)+'\n')
    candidates=[]
    for label in ('anger','disgust','fear','happiness','neutral','sadness','surprise'):
        validation=[r for r in results if r['split']=='expw-val' and r['label']==label and not r['real_eyewear'] and r['eyeglasses_model_probability']>=.5]
        training=[r for r in results if r['split']=='train' and r['label']==label and not r['real_eyewear'] and r['eyeglasses_model_probability']>=.85]
        candidates+=sorted(validation,key=lambda r:-r['eyeglasses_model_probability'])
        candidates+=sorted(training,key=lambda r:-r['eyeglasses_model_probability'])[:100]
    for index,row in enumerate(candidates):row['review_index']=index
    for page,start in enumerate(range(0,len(candidates),36)):
        batch=candidates[start:start+36];canvas=Image.new('RGB',(1344,((len(batch)+5)//6)*248),'white');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(batch):
            with Image.open(dataset/row['path']) as image:canvas.paste(image.convert('RGB').resize((224,224)),((i%6)*224,(i//6)*248))
            label=f"{row['review_index']} {row['split']} {row['label']}"
            draw.text(((i%6)*224+3,(i//6)*248+226),label,fill='black')
        canvas.save(output/f'candidates-{page:03d}.jpg',quality=96)
    report={'source':provenance,'purpose':'Candidate mining only. Probabilities are not verified eyewear annotations and are not expression predictions. No automatic emotion relabeling or dataset modification.',
            'holdout_read':False,'scanned':len(results),'candidate_counts':dict(Counter(r['split']+'/'+r['label'] for r in candidates)),
            'sanity_confusion_no_glasses_glasses':matrix.tolist(),'candidates':candidates}
    (output/'review-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source','candidates')}),flush=True)

if __name__=='__main__':main()
