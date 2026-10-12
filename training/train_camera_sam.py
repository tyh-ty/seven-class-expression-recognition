"""Bounded SAM/AdamW experiment (ICLR 2021, arXiv:2010.01412 Algorithm 1).

The image model and inference path are unchanged. Only CPU FP32 or CUDA BF16
are supported: FP16 scaling and gradient accumulation fail explicitly rather
than silently changing the SAM neighborhood. Camera augmentation is sampled
once and the exact same minibatch is used for both gradient computations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

import train_camera_robust as camera
import train_transformer as trainer

ORIGINAL_EPOCH = trainer.run_epoch
OPTIONS = {'rho': .05}


def sam_update(model, optimizer, closure, rho=.05, clip_norm=1.):
    """Restore exact original parameters even when the second closure fails."""
    if rho < 0:
        raise ValueError('rho must be nonnegative')
    optimizer.zero_grad(set_to_none=True)
    loss, logits = closure()
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        raise RuntimeError('SAM received no gradients')
    norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for p in parameters]))
    if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite first SAM loss/gradient; no update applied')
    originals = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                p.add_(p.grad, alpha=float(rho / (norm + 1e-12)))
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss, _ = closure()
        perturbed_loss.backward()
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)
    if not bool(torch.isfinite(perturbed_loss)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError('Nonfinite perturbed SAM loss; original weights restored')
    torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach(), logits.detach(), perturbed_loss.detach()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              accumulation_steps=1, max_batches=None, amp_dtype=torch.float16,
              teacher_model=None, distillation_weight=0., distillation_temperature=2.,
              distillation_scope='base'):
    if optimizer is None:
        return ORIGINAL_EPOCH(model, loader, criterion, device, max_batches=max_batches,
                              amp_dtype=amp_dtype)
    if accumulation_steps != 1 or (scaler is not None and scaler.is_enabled()):
        raise ValueError('SAM experiment requires accumulation=1 and unscaled FP32/BF16')
    if device.type == 'cuda' and amp_dtype != torch.bfloat16:
        raise ValueError('CUDA SAM requires BF16; FP16 not implemented')
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('This ViT SAM experiment does not support BatchNorm')
    if distillation_scope not in ('base', 'base-and-correct-extra'):
        raise ValueError('Unknown distillation scope')
    model.train()
    confusion = torch.zeros((len(trainer.CLASSES), len(trainer.CLASSES)), dtype=torch.int64)
    total_loss = perturbed_sum = 0.
    count = updates = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images, targets = (b.to(device, non_blocking=True) for b in batch[:2])
        is_extra = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None
        selected = teacher_logits = None
        if teacher_model is not None and is_extra is not None and distillation_weight > 0:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                                 enabled=device.type == 'cuda'):
                selected = ~is_extra
                if distillation_scope == 'base-and-correct-extra':
                    all_teacher_logits = teacher_model(images)
                    selected = selected | (all_teacher_logits.argmax(1) == targets)
                    teacher_logits = all_teacher_logits[selected].detach()
                elif bool(selected.any()):
                    teacher_logits = teacher_model(images[selected]).detach()

        # Replay stochastic network masks between both passes; data augmentation
        # has already happened. Advance RNG by one model forward, not two.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None

        def closure():
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                logits = model(images)
                loss = criterion(logits, targets, is_extra)
                if teacher_logits is not None and bool(selected.any()):
                    t = distillation_temperature
                    loss = loss + distillation_weight * F.kl_div(
                        F.log_softmax(logits[selected] / t, dim=1),
                        F.softmax(teacher_logits / t, dim=1), reduction='batchmean') * t**2
            return loss, logits

        loss, logits, adversarial = sam_update(model, optimizer, closure, OPTIONS['rho'])
        confusion += torch.bincount(targets.cpu() * len(trainer.CLASSES) + logits.argmax(1).cpu(),
                                    minlength=len(trainer.CLASSES)**2).reshape(confusion.shape)
        total_loss += float(loss) * len(targets)
        perturbed_sum += float(adversarial) * len(targets)
        count += len(targets)
        updates += 1
    if count == 0:
        raise RuntimeError('No SAM batches processed')
    print(json.dumps({'sam_rho': OPTIONS['rho'], 'updates': updates, 'examples': count,
                      'unperturbed_loss': total_loss/count, 'perturbed_loss': perturbed_sum/count}), flush=True)
    return trainer.metrics_from_confusion(confusion, total_loss/count), confusion, updates


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sam-rho', type=float, default=.05)
    options, remaining = parser.parse_known_args()
    if options.sam_rho < 0:
        raise ValueError('SAM rho must be nonnegative')
    OPTIONS['rho'] = options.sam_rho
    sys.argv = [sys.argv[0], *remaining]
    if '--output' in remaining:
        out = Path(remaining[remaining.index('--output') + 1])
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_json(out/'sam-method.json', {
            'paper': 'https://arxiv.org/abs/2010.01412', 'algorithm': 'SAM Algorithm 1, p=2',
            'rho': options.sam_rho, 'base_optimizer': 'AdamW',
            'precision': 'CUDA BF16 forward, FP32 weights/gradient accumulation; CPU FP32',
            'same_augmented_batch': True, 'same_dropout_masks': True,
            'restore': 'exact FP32 parameter copy before optimizer step',
            'limitations': 'Adaptation of optimization method, not reproduction of paper benchmarks.'})
    trainer.run_epoch = run_epoch
    camera.main()


if __name__ == '__main__':
    main()
