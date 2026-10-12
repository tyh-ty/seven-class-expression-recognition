"""ViT training with reproducible camera degradation and landmark-placed training-only eyewear."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_transformer as trainer
from benchmark_glasses_product import add_glasses

SAMPLING_OPTIONS = {'new_source_fraction': .35, 'real_eyewear_multiplier': 3.,
                    'real_eyewear_class_weighting': 'none', 'new_source_prefixes': ['web_expw_']}

class CameraDegradation:
    def __call__(self, image):
        if random.random() >= .45:
            return image
        array = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        if random.random() < .75:
            size = random.randint(72, 180)
            array = cv2.resize(array, (size,size), interpolation=cv2.INTER_AREA)
        if random.random() < .2:
            array = cv2.GaussianBlur(array,(3,3),random.uniform(.1,.6))
        if random.random() < .75:
            ok, encoded = cv2.imencode('.jpg',array,[cv2.IMWRITE_JPEG_QUALITY,random.randint(55,95)])
            if ok:
                array = cv2.imdecode(encoded,cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB))

def camera_transforms(image_size, grayscale=True):
    _, evaluation = ORIGINAL_TRANSFORMS(image_size,grayscale=grayscale)
    return transforms.Compose([
        CameraDegradation(),
        *([transforms.Grayscale(num_output_channels=3)] if grayscale else []),
        transforms.RandomResizedCrop(image_size,scale=(.86,1.0),ratio=(.95,1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=.25,contrast=.25),
        transforms.ToTensor(),
        transforms.Normalize(trainer.IMAGENET_MEAN,trainer.IMAGENET_STD),
    ]), evaluation

class EyewearImageFolder(trainer.SourceAwareImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        dataset_path = Path(self.root).parent / 'dataset.json'
        self.landmarks = {}
        self.real_eyewear = set()
        if dataset_path.exists():
            metadata = json.loads(dataset_path.read_text(encoding='utf-8'))
            for row in metadata.get('images',[]):
                if row.get('split') == 'train' and row.get('real_eyewear',False):
                    self.real_eyewear.add(Path(row['path']).name)
                if row.get('split') == 'train' and row.get('landmarks') and not row.get('eyewear_possible',row.get('real_eyewear',False)):
                    self.landmarks[Path(row['path']).name] = np.asarray(row['landmarks'],dtype=np.float32)

    def __getitem__(self,index):
        path,target = self.samples[index]
        image = self.loader(path)
        points = self.landmarks.get(Path(path).name)
        if points is not None and random.random() < .25:
            image_bgr = cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2BGR)
            altered = add_glasses(image_bgr,points,random.randrange(10000))
            image = Image.fromarray(cv2.cvtColor(altered,cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image,target,Path(path).name.startswith(trainer.EXTRA_SOURCE_PREFIXES)

ORIGINAL_TRANSFORMS = trainer.build_transforms
ORIGINAL_LOADER = trainer.make_loader

def source_balanced_loader(dataset, batch_size, workers, shuffle, device):
    if not shuffle:
        return ORIGINAL_LOADER(dataset,batch_size,workers,False,device)
    real_eyewear=getattr(dataset,'real_eyewear',set())
    new_prefixes=tuple(SAMPLING_OPTIONS['new_source_prefixes'])
    # Keep replay substantial as the original ExpW corpus grows; avoid a fixed 5x
    # weight causing a large new source to silently dominate the entire epoch.
    new_count=sum(Path(path).name.startswith(new_prefixes) for path,_ in dataset.samples)
    base_count=len(dataset)-new_count
    fraction = SAMPLING_OPTIONS['new_source_fraction']
    multiplier = SAMPLING_OPTIONS['real_eyewear_multiplier']
    new_weight=1.0 if base_count == 0 else min(5.0,base_count*fraction/((1-fraction)*max(1,new_count)))
    eyewear_counts = Counter(target for path, target in dataset.samples if Path(path).name in real_eyewear)
    eyewear_class_weights = {target: 1. for target in eyewear_counts}
    if SAMPLING_OPTIONS['real_eyewear_class_weighting'] == 'inverse-sqrt':
        raw = {target: count ** -.5 for target, count in eyewear_counts.items()}
        average = sum(raw.values()) / max(1, len(raw))
        eyewear_class_weights = {target: value / average for target, value in raw.items()}
    weights = [(new_weight if Path(path).name.startswith(new_prefixes) else 1.0)
               * (multiplier*eyewear_class_weights.get(target,1.) if Path(path).name in real_eyewear else 1.0)
               for path,target in dataset.samples]
    sampling={'new_images':new_count,'base_images':base_count,'new_weight':new_weight,'real_eyewear_weight':new_weight*multiplier,
              'configuration': dict(SAMPLING_OPTIONS),
              'real_eyewear_counts': {trainer.CLASSES[k]: v for k, v in eyewear_counts.items()},
              'real_eyewear_class_weights': {trainer.CLASSES[k]: v for k, v in eyewear_class_weights.items()},
              'expected_new_fraction':sum(w for w,(path,_) in zip(weights,dataset.samples) if Path(path).name.startswith(new_prefixes))/sum(weights)}
    print('sampling_policy='+json.dumps(sampling),flush=True)
    if '--output' in sys.argv:
        (Path(sys.argv[sys.argv.index('--output')+1])/'sampling-policy.json').write_text(json.dumps(sampling,indent=2),encoding='utf-8')
    return DataLoader(dataset,batch_size=batch_size,
                      sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(dataset),replacement=True),
                      num_workers=workers,pin_memory=device.type=='cuda',persistent_workers=workers>0,drop_last=True)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--new-source-fraction', type=float, default=.35)
    parser.add_argument('--real-eyewear-multiplier', type=float, default=3.)
    parser.add_argument('--real-eyewear-class-weighting', choices=('none', 'inverse-sqrt'), default='none')
    parser.add_argument('--new-source-prefix', dest='new_source_prefixes', action='append',
                        help='Repeat for each new source. Defaults to web_expw_. Must also be a recognized extra-source prefix.')
    options, remaining = parser.parse_known_args()
    options.new_source_prefixes = options.new_source_prefixes or ['web_expw_']
    if any(not p or not p.startswith(trainer.EXTRA_SOURCE_PREFIXES) for p in options.new_source_prefixes):
        raise ValueError('New source prefixes must identify extra-source training files')
    if not 0 < options.new_source_fraction < 1 or options.real_eyewear_multiplier <= 0:
        raise ValueError('Source fraction must lie within(0,1), and eyewear multiplier must be positive')
    SAMPLING_OPTIONS.update(vars(options))
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *remaining]
    cv2.setNumThreads(1)
    trainer.build_transforms = camera_transforms
    trainer.SourceAwareImageFolder = EyewearImageFolder
    trainer.make_loader = source_balanced_loader
    # Store augmentation configuration before the main routine serializes its run.
    if '--output' in sys.argv:
        out = Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
        (out/'camera-augmentation.json').write_text(json.dumps({
            'camera_degradation_probability':.45,'resolution_range':[72,180],
            'jpeg_quality_range':[55,95],'glasses_probability_for_landmarked_new_faces':.25,
            'expw_sampling_weight_max':5.0,'base_sampling_weight':1.0,
            'real_eyewear_sampling_multiplier':options.real_eyewear_multiplier,
            'source_share_target_before_eyewear_boost':options.new_source_fraction,
            'real_eyewear_class_weighting':options.real_eyewear_class_weighting,
            'original_argv':original_argv,
            'limits':'Simulated eyewear/degradation is training augmentation, not proof of real-eyewear accuracy.',
        },indent=2),encoding='utf-8')
    trainer.main()

if __name__ == '__main__':
    main()
