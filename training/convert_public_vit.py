"""Convert the pinned public ViT to timm, proving identical logits and folding input normalization."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import timm
import torch
from transformers import AutoModelForImageClassification

ROOT=Path(__file__).resolve().parents[1]
CLASSES=('anger','disgust','fear','happiness','neutral','sadness','surprise')

def main():
    torch.set_num_threads(4)
    source=ROOT/'models/research/jihed-vit'
    provenance=json.loads((source/'source.json').read_text())
    original=AutoModelForImageClassification.from_pretrained(source,local_files_only=True,trust_remote_code=False,use_safetensors=True).eval()
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=7).eval()
    old=original.state_dict();state={}
    mapping={'cls_token':'vit.embeddings.cls_token','pos_embed':'vit.embeddings.position_embeddings',
             'patch_embed.proj.weight':'vit.embeddings.patch_embeddings.projection.weight',
             'patch_embed.proj.bias':'vit.embeddings.patch_embeddings.projection.bias',
             'norm.weight':'vit.layernorm.weight','norm.bias':'vit.layernorm.bias',
             'head.weight':'classifier.weight','head.bias':'classifier.bias'}
    for dest,src in mapping.items():state[dest]=old[src].clone()
    for i in range(12):
        a=f'blocks.{i}.';b=f'vit.encoder.layer.{i}.'
        for suffix in ('weight','bias'):
            state[a+'attn.qkv.'+suffix]=torch.cat([old[b+'attention.attention.'+key+'.'+suffix] for key in ('query','key','value')],dim=0)
            for dest,src in [('attn.proj','attention.output.dense'),('norm1','layernorm_before'),('norm2','layernorm_after'),('mlp.fc1','intermediate.dense'),('mlp.fc2','output.dense')]:
                state[a+dest+'.'+suffix]=old[b+src+'.'+suffix].clone()
    model.load_state_dict(state,strict=True)
    torch.manual_seed(123)
    rgb=torch.rand(3,3,224,224)
    with torch.inference_mode():
        expected=original(pixel_values=(rgb-.5)/.5).logits
        raw=model((rgb-.5)/.5)
    before=float((expected-raw).abs().max())
    assert before<2e-4,(before,'HF to timm mismatch')
    mean=torch.tensor([.485,.456,.406]);std=torch.tensor([.229,.224,.225])
    weight=state['patch_embed.proj.weight'].clone()
    state['patch_embed.proj.weight']=weight*(std/.5).view(1,3,1,1)
    state['patch_embed.proj.bias']+= (weight*((mean-.5)/.5).view(1,3,1,1)).sum(dim=(1,2,3))
    model.load_state_dict(state,strict=True)
    with torch.inference_mode():actual=model((rgb-mean.view(1,3,1,1))/std.view(1,3,1,1))
    after=float((expected-actual).abs().max())
    assert after<2e-4,(after,'Input normalization folding mismatch')
    output=ROOT/'training/runs/vit-base-public-initial';output.mkdir(exist_ok=True)
    torch.save({'model_name':'vit_base_patch16_224','model_state':state,'classes':CLASSES,'image_size':224,
                'grayscale':False,'mean':mean.tolist(),'std':std.tolist(),'epoch':0,'source':provenance,
                'conversion_max_error':after},output/'initial.pt')
    report={'source':provenance,'grayscale':False,'hf_to_timm_max_error':before,'folded_normalization_max_error':after,
            'note':'Input normalization changed algebraically to ImageNet mean/std; output logits preserved on three random RGB inputs. No training represented by this conversion.'}
    (output/'conversion.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
