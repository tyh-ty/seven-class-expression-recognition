"""Bounded training-only upsampling of existing local corrections; standard replay retained."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_camera_robust as camera

ORIGINAL_LOADER=camera.source_balanced_loader
MULTIPLIER=100.


def feedback_loader(dataset,batch_size,workers,shuffle,device):
    loader=ORIGINAL_LOADER(dataset,batch_size,workers,shuffle,device)
    if not shuffle:return loader
    weights=loader.sampler.weights.clone()
    selected=torch.tensor([Path(p).name.startswith('web_feedback_local_') for p,_ in dataset.samples])
    if int(selected.sum())!=3:raise ValueError('Expected exactly three audited localtraining images')
    weights[selected]*=MULTIPLIER
    report={'unique_training_corrections':3,'multiplier':MULTIPLIER,
            'expected_sampling_fraction':float(weights[selected].sum()/weights.sum()),
            'scope':'Trainingfit on these3frames is not independent webcamaccuracy. Publicvalidation andsealedholdout remainseparate.'}
    if '--output' in sys.argv:
        out=Path(sys.argv[sys.argv.index('--output')+1]);(out/'feedback-sampling.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('feedback_sampling='+json.dumps(report),flush=True)
    return DataLoader(dataset,batch_size=batch_size,
                      sampler=WeightedRandomSampler(weights,len(dataset),replacement=True),num_workers=workers,
                      pin_memory=device.type=='cuda',persistent_workers=workers>0,drop_last=True)


def main():
    global MULTIPLIER
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--feedback-multiplier',type=float,default=100.)
    args,remaining=parser.parse_known_args()
    if not 1<=args.feedback_multiplier<=200:raise ValueError('Feedback multiplier outside bounded range')
    MULTIPLIER=args.feedback_multiplier
    sys.argv=[sys.argv[0],*remaining]
    camera.source_balanced_loader=feedback_loader
    camera.main()


if __name__=='__main__':main()
