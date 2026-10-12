"""Download published Transformer candidates pinned to model commits; verify LFS SHA256."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODELS={
 'jihed-vit':('jihedjabnoun/vit-face-emotion-recognition','ec667156e4f9a2e329911a1da5c13572ae3d75d4'),
 'dima-vit':('dima806/facial_emotions_image_detection','747cf16692eea925b54b7b543cf436848128b68d'),
 'beit-large':('Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large','68cbe4de6a93da0ec70267ab792b539311ad3d3f'),
 'mauregato-affectnet':('Mauregato/vit-base-patch16-224-best-finetuned-on-affectnet_short','f45e825be0639eb23e5aea16af6c100019dd0d3b'),
 'michael-affectnet':('michaelgathara/vit-face-affectnet','fe45987c7ce97626304117fab9d059cca9cd7d2b'),
 'eyeglasses-vit':('youngp5/eyeglasses_detection','92b649d59ec05cf225dfd8440b7a1cf8f0ab946f'),
 'trpakov-vit':('trpakov/vit-face-expression','ef0bc6fc34241b6587e7e009e7711357be28c024'),
 'mo-vit':('mo-thecreator/vit-Facial-Expression-Recognition','5aae4f09e1da247eb0f1c1e6f53271601fcafe09'),
 'micro-vit':('LaurenGurgiolo/vit-micro-facial-expressions','616ac2a55c59cba5c04e46e34bfda870ad157cf9'),
 'hardly-vit':('HardlyHumans/Facial-expression-detection','32132bdc0b6c9ad71e47a905db5cdf0ae51de85d'),
 'dinov2-small':('timm/vit_small_patch14_dinov2.lvd142m','4610ca143709d58a633b6397a74412c2c3842454'),
}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=list(MODELS),default=['jihed-vit','dima-vit','beit-large'])
    args=parser.parse_args()
    for key in args.models:
        repo,revision=MODELS[key]
        output=ROOT/'models/research'/key;output.mkdir(parents=True,exist_ok=True)
        req=urllib.request.Request(f'https://hf-mirror.com/api/models/{repo}?blobs=true',headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=30) as response:metadata=json.load(response)
        if metadata['sha']!=revision:raise ValueError('Repository changed; re-audit revision')
        files={f['rfilename']:f for f in metadata['siblings']}
        weights_name='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
        for name in ('README.md','config.json','preprocessor_config.json',weights_name):
            if key == 'dinov2-small' and name == 'preprocessor_config.json':
                continue  # timm publishes its input contract in config.json.
            info=files[name];size=info['size'];target=output/name
            expected=info.get('lfs',{}).get('sha256')
            if target.exists() and target.stat().st_size==size:
                with target.open('rb') as f:existing=hashlib.file_digest(f,'sha256').hexdigest()
                if not expected or existing==expected:continue
            url=f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}'
            if size<1024*1024:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as response:raw=response.read()
                if len(raw)!=size:raise ValueError('Metadata file size mismatch')
                target.write_bytes(raw)
                continue
            partial=target.with_suffix('.partial')
            with partial.open('wb') as f:f.truncate(size)
            chunks=list(range(0,size,8*1024*1024))
            def fetch(start):
                end=min(size-1,start+8*1024*1024-1)
                for attempt in range(4):
                    try:
                        request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Range':f'bytes={start}-{end}'})
                        with urllib.request.urlopen(request,timeout=40) as response:
                            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Range mismatch')
                            raw=response.read(end-start+2)
                        if len(raw)!=end-start+1:raise ValueError('Truncated range')
                        return start,raw
                    except Exception as exc:
                        if attempt==3:raise RuntimeError(f'{key} range {start}: {type(exc).__name__}') from None
                        time.sleep(2**attempt)
                raise AssertionError
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(fetch,start) for start in chunks}
                for i,future in enumerate(as_completed(futures),1):
                    start,raw=future.result()
                    futures.remove(future)
                    with partial.open('r+b',buffering=0) as f:f.seek(start);f.write(raw)
                    if i%10==0 or i==len(chunks):print(key,f'{i}/{len(chunks)}',flush=True)
            with partial.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            if not expected or digest!=expected:raise ValueError('Model SHA256 mismatch')
            partial.replace(target)
        if weights_name=='pytorch_model.bin':
            import torch
            from safetensors.torch import save_file
            tensors=torch.load(output/weights_name,map_location='cpu',weights_only=True)
            if not isinstance(tensors,dict) or not all(isinstance(v,torch.Tensor) for v in tensors.values()):
                raise ValueError('Expected tensor-only state dictionary; custom objects are not accepted')
            save_file({k:v.contiguous() for k,v in tensors.items()},output/'model.safetensors',metadata={'format':'pt'})
            del tensors
        with (output/'model.safetensors').open('rb') as f:local_weights_hash=hashlib.file_digest(f,'sha256').hexdigest()
        report={'source':f'https://huggingface.co/{repo}','revision':revision,'model_sha256':local_weights_hash,
                'published_weights_format':weights_name,'published_weights_sha256':files[weights_name]['lfs']['sha256'],
                'conversion':'Tensor-only torch.load(weights_only=True), then safetensors; no custom object allowlist or remote model code.' if weights_name.endswith('.bin') else None,
                'license':(metadata.get('cardData') or {}).get('license','No license declared in model-card metadata; deployment rights not established.'),
                'scope':'Public candidate models; claimed metrics not trusted as measured local accuracy. External training-set overlap with local benchmarks must be disclosed.',
                'remote_code_executed':False}
        (output/'source.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('verified_model',key,flush=True)

if __name__=='__main__':main()
