"""Three-view consistency inspired by AugMix (ICLR2020), not full AugMix.

Our views use existing landmark eyewear and mild camera changes, rather than
AugMix's mixture chains. All three views keep the original human label. Their
mean supervised CE and clean baseline distillation are identical in the paired
control. Only the JSD coefficient differs. The teacher gates an auxiliary loss;
its prediction never replaces a dataset label. Inference remains a single ViT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.nn import functional as F

import train_camera_robust as camera
import train_transformer as trainer

ORIGINAL_EPOCH = trainer.run_epoch
OPTIONS = {'jsd_weight': 12.0}


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def shared_transform(transform, images):
    """Same geometry/color/degradation draws, with RNG advanced by one transform."""
    before = rng_state()
    result = [transform(images[0])]
    after = rng_state()
    try:
        for image in images[1:]:
            restore_rng(before)
            result.append(transform(image))
    finally:
        restore_rng(after)
    return result


def mild_camera_view(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(.85, 1.15))
    array = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    if random.random() < .5:
        h, w = array.shape[:2]
        size = random.randint(96, 180)
        array = cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)
        array = cv2.resize(array, (w, h), interpolation=cv2.INTER_LINEAR)
    ok, encoded = cv2.imencode('.jpg', array, [cv2.IMWRITE_JPEG_QUALITY, random.randint(65, 95)])
    if not ok:
        raise ValueError('Camera augmentation JPEG failed')
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))


class ThreeViewFolder(camera.EyewearImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        clean = self.loader(path)
        points = self.landmarks.get(Path(path).name)
        synthetic = points is not None
        if synthetic:
            bgr = cv2.cvtColor(np.asarray(clean.convert('RGB')), cv2.COLOR_RGB2BGR)
            altered = camera.add_glasses(bgr, points, random.randrange(10000))
            first = Image.fromarray(cv2.cvtColor(altered, cv2.COLOR_BGR2RGB))
        else:
            first = mild_camera_view(clean)
        second = mild_camera_view(clean)
        views = shared_transform(self.transform, [clean, first, second])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (*views, target, Path(path).name.startswith(trainer.EXTRA_SOURCE_PREFIXES), synthetic)


def consistency_jsd(logits, eligible):
    """Full-gradient mean KL(p_view || mean(p_views)); gate only uses frozen teacher."""
    # Softmax/KL stay FP32 under BF16 training; neither side is stop-gradient.
    probabilities = [F.softmax(x.float(), dim=1) for x in logits]
    log_mixture = torch.stack(probabilities).mean(0).clamp_min(1e-7).log()
    per_sample = torch.stack([F.kl_div(log_mixture, p, reduction='none').sum(1)
                              for p in probabilities]).mean(0)
    return (per_sample * eligible.to(per_sample.dtype)).mean()


def train_loss(logits, targets, extra, criterion, teacher_logits,
               distillation_weight, temperature, scope, jsd_weight):
    supervised = torch.stack([criterion(x.float(), targets, extra) for x in logits]).mean()
    correct = teacher_logits.argmax(1) == targets
    selected = ~extra
    if scope == 'base-and-correct-extra':
        selected = selected | correct
    elif scope != 'base':
        raise ValueError('Unsupported distillation scope')
    distill = logits[0].float().sum() * 0
    if bool(selected.any()) and distillation_weight:
        distill = F.kl_div(F.log_softmax(logits[0][selected].float() / temperature, dim=1),
                          F.softmax(teacher_logits[selected].float() / temperature, dim=1),
                          reduction='batchmean') * temperature ** 2
    consistency = consistency_jsd(logits, correct)
    total = supervised + distillation_weight * distill + jsd_weight * consistency
    return total, consistency, correct


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              accumulation_steps=1, max_batches=None, amp_dtype=torch.float16,
              teacher_model=None, distillation_weight=0., distillation_temperature=2.,
              distillation_scope='base'):
    if optimizer is None:
        return ORIGINAL_EPOCH(model, loader, criterion, device, max_batches=max_batches,
                              amp_dtype=amp_dtype)
    if accumulation_steps != 1 or (scaler is not None and scaler.is_enabled()):
        raise ValueError('Consistency experiment supports unscaled FP32/BF16, accumulation=1')
    if device.type == 'cuda' and amp_dtype != torch.bfloat16:
        raise ValueError('CUDA BF16 required')
    if teacher_model is None:
        raise ValueError('Frozen original baseline teacher is required')
    teacher_model.eval(); model.train()
    confusion = torch.zeros((7, 7), dtype=torch.int64)
    count = updates = eligible_count = glasses_count = 0
    total_loss = jsd_sum = 0.
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        if len(batch) != 6:
            raise ValueError('Expected three aligned views and original metadata')
        clean, first, second, targets, extra, synthetic = [x.to(device, non_blocking=True) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                             enabled=device.type == 'cuda'):
            teacher_logits = teacher_model(clean).detach()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            logits = model(torch.cat([clean, first, second])).split(len(targets))
        loss, jsd, correct = train_loss(logits, targets, extra, criterion, teacher_logits,
                                        distillation_weight, distillation_temperature,
                                        distillation_scope, OPTIONS['jsd_weight'])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite consistency loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        confusion += torch.bincount(targets.cpu() * 7 + logits[0].detach().argmax(1).cpu(), minlength=49).reshape(7, 7)
        count += len(targets); updates += 1
        eligible_count += int(correct.sum()); glasses_count += int(synthetic.sum())
        total_loss += float(loss.detach()) * len(targets); jsd_sum += float(jsd.detach()) * len(targets)
    if not count:
        raise RuntimeError('No training batches processed')
    print(json.dumps({'jsd_weight': OPTIONS['jsd_weight'], 'updates': updates, 'examples': count,
                      'loss': total_loss/count, 'jsd': jsd_sum/count,
                      'teacher_correct_fraction': eligible_count/count,
                      'synthetic_eyewear_views': glasses_count,
                      'note': 'Synthetic augmentation count is not real-eyewear test support.'}), flush=True)
    return trainer.metrics_from_confusion(confusion, total_loss/count), confusion, updates


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--jsd-weight', type=float, default=12.)
    options, remaining = p.parse_known_args()
    if options.jsd_weight < 0:
        raise ValueError('Negative consistency weight')
    OPTIONS['jsd_weight'] = options.jsd_weight
    sys.argv = [sys.argv[0], *remaining]
    if '--output' in remaining:
        out = Path(remaining[remaining.index('--output')+1]); out.mkdir(parents=True, exist_ok=True)
        trainer.save_json(out/'consistency-method.json', {
            'paper': 'https://arxiv.org/abs/1912.02781', 'official_code_commit': '9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4',
            'method': 'AugMix-inspired three-view JSD; not full AugMix mixture augmentation',
            'jsd_weight': options.jsd_weight, 'supervision': 'mean CE of 3 views, original human label unchanged',
            'gating': 'JSD only when frozen original model predicts original training label correctly; mean over full batch',
            'views': 'shared crop/geometry: clean, landmark synthetic eyewear (otherwise camera), independent mild camera',
            'real_eyewear': 'Never overlay synthetic glasses on annotated real/possible eyewear; retain original sampler',
            'inference': 'unchanged single ViT; no extra head or extra model'})
    camera.EyewearImageFolder = ThreeViewFolder
    trainer.run_epoch = run_epoch
    camera.main()


if __name__ == '__main__':
    main()
