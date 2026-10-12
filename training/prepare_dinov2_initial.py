"""Initialize a seven-class RGB ViT from verified public DINOv2 features.

The classification head is random. This conversion is not emotion training.
"""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file
import timm
from timm.layers import resample_abs_pos_embed
import torch

from train_transformer import CLASSES, IMAGENET_MEAN, IMAGENET_STD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('models/research/dinov2-small'))
    parser.add_argument('--output', type=Path, default=Path('training/runs/dinov2-small-initial/initial.pt'))
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(42)
    source = json.loads((args.source / 'source.json').read_text())
    weights = args.source / 'model.safetensors'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != source['model_sha256'] or digest != '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081':
        raise ValueError('Unexpected DINOv2 source weights')
    name = 'vit_small_patch14_dinov2.lvd142m'
    model_kwargs = {'img_size': 224}
    model = timm.create_model(name, pretrained=False, num_classes=7, **model_kwargs).eval()
    state = load_file(str(weights), device='cpu')
    original_shape = tuple(state['pos_embed'].shape)
    state['pos_embed'] = resample_abs_pos_embed(state['pos_embed'], (16, 16), old_size=(37, 37),
                                             num_prefix_tokens=1, interpolation='bicubic', antialias=True)
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != {'head.weight', 'head.bias'} or incompatible.unexpected_keys:
        raise ValueError(str(incompatible))
    with torch.inference_mode():
        result = model(torch.randn(2, 3, 224, 224))
    assert result.shape == (2, 7) and torch.isfinite(result).all()
    payload = {'model_name': name, 'model_kwargs': model_kwargs, 'model_state': model.state_dict(),
               'classes': CLASSES, 'image_size': 224, 'grayscale': False,
               'mean': IMAGENET_MEAN, 'std': IMAGENET_STD, 'epoch': 0,
               'source': source, 'classification_head': 'Random seven-class initialization, seed42; not trained.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(payload, args.output)
    metadata = {k: v for k, v in payload.items() if k != 'model_state'}
    metadata.update(original_position_shape=original_shape, new_position_shape=tuple(state['pos_embed'].shape),
                    positional_resize='37x37 to16x16, bicubic antialias, CLS token unchanged',
                    missing_keys=incompatible.missing_keys, unexpected_keys=incompatible.unexpected_keys,
                    output_shape=list(result.shape), initialization_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                    scope='No expression training or evaluation. Public self-supervised features may contain unknown benchmark overlap.')
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
