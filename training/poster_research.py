"""Strict official-code loading for a frozen-feature, newly labeled head trial.

The original checkpoint output order is UNKNOWN. Never assign its anonymous
seven logits emotion labels. Only features before its classifier may be reused;
a new classifier must learn our seven labels from existing training images.
"""
import hashlib
import importlib
from pathlib import Path
import sys
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
EXTERNAL=ROOT/'training/external/poster-v2'
WEIGHTS=ROOT/'models/research/poster-v2-affectnet7/affectnet-7-model_best.pth'
SHA='7ca3533b076594ccd573d30c9b666167c8241e811a70f9cfe2b6a4586dcaf5b5'


def build_features():
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()!=SHA:
        raise ValueError('Unverified mirror checkpoint bytes')
    sys.path.insert(0,str(EXTERNAL))
    module=importlib.import_module('models.PosterV2_7cls')
    model=module.pyramid_trans_expr2(load_auxiliary=False)
    checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    raw=checkpoint['model']
    state={k.removeprefix('module.'):v for k,v in raw.items()}
    if len(state)!=len(raw):raise ValueError('Colliding renamed keys')
    model.load_state_dict(state,strict=True)
    # The entire original classifier is discarded, no output permutation guessed.
    model.VIT.head=nn.Identity()
    model.requires_grad_(False)
    return model.eval()
