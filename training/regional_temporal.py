"""Regional visual features plus causal temporal residual; a research prototype.

Inspired by MAE-DFER's regional/motion modeling, NOT its pretrained model or a
reproduction. The deployed ViT is frozen; its CLS and upper/lower patch means
are preserved instead of reducing each frame to seven probabilities first.
"""
import math
import numpy as np
import torch
from torch import nn
from temporal_residual import iou

WINDOW = 8


class RegionalVisual(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        tokens = self.model.forward_features(images)
        patches = tokens[:, 1:].reshape(-1, 14, 14, tokens.shape[-1])
        features = torch.stack((tokens[:, 0], patches[:, :7].mean((1, 2)),
                                patches[:, 7:].mean((1, 2))), dim=1)
        return self.model.forward_head(tokens), features


def causal_indices(video):
    history=[];last=None;rows=[]
    for fi,frame in enumerate(video['frames']):
        if frame.get('status')!='detected' or 'feature_index' not in frame:
            history=[];last=None;continue
        if last and (frame['seconds']-last['seconds']>.3 or iou(frame['bbox'],last['bbox'])<.3):
            history=[]
        history.append((frame['seconds'],frame['feature_index']))
        history=[(s,i) for s,i in history[-WINDOW:] if frame['seconds']-s<=.85]
        indices=[i for _,i in history]
        rows.append((fi,[indices[0]]*(WINDOW-len(indices))+indices))
        last=frame
    return rows


class RegionalTemporal(nn.Module):
    def __init__(self, feature_dim=384, hidden=64):
        super().__init__()
        self.input_norm=nn.LayerNorm(feature_dim)
        self.projection=nn.Linear(feature_dim,hidden)
        self.region=nn.Parameter(torch.zeros(1,1,3,hidden))
        nn.init.normal_(self.region,std=.01)
        position=torch.arange(WINDOW).float()[:,None]
        freq=torch.exp(torch.arange(0,hidden,2).float()*(-math.log(10000.)/hidden))
        enc=torch.zeros(WINDOW,hidden)
        enc[:,::2]=torch.sin(position*freq);enc[:,1::2]=torch.cos(position*freq)
        self.register_buffer('position',enc[None,:,None,:])
        block=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,1,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(hidden)
        self.head=nn.Linear(hidden,7)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def residual(self, features):
        values=self.projection(self.input_norm(features.contiguous()))+self.region+self.position
        encoded=self.encoder(values.reshape(len(features),WINDOW*3,-1))
        return self.head(self.norm(encoded[:,-3:].mean(1)))

    def forward(self, features, current_logits):
        # The window contains only current/past frames. A constant video exactly
        # preserves the static model even after temporal parameters are trained.
        static=features[:,-1:].expand_as(features).contiguous()
        delta=self.residual(features)-self.residual(static)
        return current_logits+2*torch.tanh(delta)


def load_features(video):
    with np.load(video['feature_file'],allow_pickle=False) as data:
        return data['features'],data['logits']
