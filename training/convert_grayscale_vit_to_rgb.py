"""Initialize an RGB ViT with the same continuous-luminance function as a gray ViT.

This is an algebraic input conversion, not training or evidence of higher accuracy.
The deployment pipeline rounds grayscale to bytes, so also measure that difference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

from emotion_preprocessing import emotion_tensor_bgr
from train_transformer import CLASSES, IMAGENET_MEAN, IMAGENET_STD


def convert_projection(weight, bias, mean, std):
    mean = torch.as_tensor(mean, dtype=weight.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(std, dtype=weight.dtype).view(1, 3, 1, 1)
    luminance = weight.new_tensor((.299, .587, .114)).view(1, 3, 1, 1)
    gray_weight = (weight / std).sum(dim=1, keepdim=True)
    rgb_weight = gray_weight * luminance * std
    # Sum the scalar gray offset separately so it is counted once per patch.
    rgb_bias = bias + gray_weight.sum(dim=(1, 2, 3)) * (luminance * mean).sum() - (weight * mean / std).sum(dim=(1, 2, 3))
    return rgb_weight, rgb_bias


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    assert source.get('grayscale', True)
    assert tuple(source['classes']) == CLASSES
    assert np.allclose(source['mean'], IMAGENET_MEAN)
    assert np.allclose(source['std'], IMAGENET_STD)
    original = timm.create_model(source['model_name'], pretrained=False, num_classes=7).eval()
    original.load_state_dict(source['model_state'], strict=True)
    projection = original.patch_embed.proj
    assert tuple(projection.padding) == (0, 0)
    state = {key: value.clone() for key, value in source['model_state'].items()}
    state['patch_embed.proj.weight'], state['patch_embed.proj.bias'] = convert_projection(
        state['patch_embed.proj.weight'], state['patch_embed.proj.bias'], source['mean'], source['std'])
    converted = timm.create_model(source['model_name'], pretrained=False, num_classes=7).eval()
    converted.load_state_dict(state, strict=True)
    mean = torch.tensor(source['mean']).view(1, 3, 1, 1)
    std = torch.tensor(source['std']).view(1, 3, 1, 1)
    torch.manual_seed(624)
    rgb = torch.rand(4, 3, source['image_size'], source['image_size'])
    gray = (rgb * torch.tensor((.299, .587, .114)).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    with torch.inference_mode():
        expected = original((gray.expand(-1, 3, -1, -1) - mean) / std)
        actual = converted((rgb - mean) / std)
    error = float((expected - actual).abs().max())
    if error > 3e-4:
        raise AssertionError(f'Continuous grayscale conversion mismatch: {error}')
    # No validation/holdout selection: fourteen deterministic training images only.
    root = Path('data/seven-class/expw-original-eyewear-v3/train')
    images = [path for label in CLASSES for path in sorted((root / label).glob('*.jpg'))[:2]]
    before = []; after = []
    for path in images:
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        before.append(emotion_tensor_bgr(bgr, grayscale=True))
        after.append(emotion_tensor_bgr(bgr, grayscale=False))
    with torch.inference_mode():
        before_logits = original(torch.from_numpy(np.stack(before)))
        after_logits = converted(torch.from_numpy(np.stack(after)))
    report = {'source': str(args.source), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'continuous_luminance_max_logit_error': error,
              'training_image_rounding_check': {'count': len(images),
                 'max_logit_difference': float((before_logits - after_logits).abs().max()),
                 'top1_agreement': float((before_logits.argmax(1) == after_logits.argmax(1)).float().mean())},
              'status': 'Converted initialization only, not trained; final deployment still requires validation.',
              'note': 'RGB projection initially depends only on luminance. It must be trainable to learn useful color differences; only unfreezing late blocks would leave the model color-blind.'}
    converted_checkpoint = {key: source[key] for key in ('model_name', 'classes', 'image_size', 'mean', 'std')}
    converted_checkpoint.update(model_state=state, grayscale=False, epoch=0, input_conversion=report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing initialization')
    torch.save(converted_checkpoint, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
