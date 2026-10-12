"""Numerical and data-alignment checks for paired consistency training."""
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

import train_camera_consistency as method
import train_camera_robust as camera
import train_transformer as trainer


def main():
    torch.set_num_threads(2); torch.manual_seed(818); random.seed(818); np.random.seed(818)
    logits = [torch.randn(5, 7, requires_grad=True) for _ in range(3)]
    mask = torch.tensor([True, False, True, True, False])
    actual = method.consistency_jsd(logits, mask)
    p = [x.softmax(1) for x in logits]; mixture = sum(p)/3
    reference = sum((v * (v.log()-mixture.log())).sum(1) for v in p)/3
    reference = (reference * mask).mean()
    assert torch.allclose(actual, reference, atol=2e-7)
    ga = torch.autograd.grad(actual, logits, retain_graph=True)
    gr = torch.autograd.grad(reference, logits, retain_graph=True)
    assert all(torch.allclose(a, r, atol=2e-7) for a, r in zip(ga, gr))
    assert all(float(g[~mask].abs().max()) == 0 for g in ga)
    assert abs(float(method.consistency_jsd([logits[0]]*3, mask))) < 2e-7
    assert abs(float(method.consistency_jsd(logits, torch.zeros(5, dtype=torch.bool)))) == 0
    targets = torch.tensor([0,1,2,3,4]); extra = torch.tensor([False,True,False,True,False])
    teacher = torch.randn(5,7)
    criterion = trainer.SampleWeightedLabelSmoothingLoss(torch.ones(7), .05)
    zero, _, _ = method.train_loss(logits, targets, extra, criterion, teacher, 1., 2., 'base-and-correct-extra', 0.)
    selected = (~extra) | (teacher.argmax(1)==targets)
    baseline = sum(criterion(v,targets,extra) for v in logits)/3
    baseline += F.kl_div(F.log_softmax(logits[0][selected]/2,dim=1), F.softmax(teacher[selected]/2,dim=1),reduction='batchmean')*4
    assert torch.allclose(zero,baseline,atol=2e-7)
    transform, _ = camera.camera_transforms(224)
    arr=np.random.default_rng(818).integers(0,256,(250,250,3),dtype=np.uint8)
    sample=Image.fromarray(arr);before=method.rng_state()
    single=transform(sample);after_single=method.rng_state()
    method.restore_rng(before);views=method.shared_transform(transform,[sample]*3)
    assert all(torch.equal(single,v) for v in views)
    after_views=method.rng_state()
    assert after_single[0]==after_views[0] and torch.equal(after_single[2],after_views[2])
    assert np.array_equal(after_single[1][1],after_views[1][1])
    data=method.ThreeViewFolder('data/seven-class/expw-enterface-v1/train',transform=transform)
    assert set(data.landmarks).isdisjoint(data.real_eyewear)
    selected_indices=[next(i for i,(p,_) in enumerate(data.samples) if Path(p).name in data.real_eyewear),
                      next(i for i,(p,_) in enumerate(data.samples) if Path(p).name in data.landmarks)]
    summaries=[]
    for i in selected_indices:
        result=data[i];path,target=data.samples[i]
        assert result[3]==target and result[4]==Path(path).name.startswith(trainer.EXTRA_SOURCE_PREFIXES)
        assert result[5]==(Path(path).name in data.landmarks)
        assert all(tuple(v.shape)==(3,224,224) and bool(torch.isfinite(v).all()) for v in result[:3])
        summaries.append({'file':path,'real_eyewear':Path(path).name in data.real_eyewear,'synthetic_view':result[5]})
    report={'jsd_reference_absolute_error':float(abs(actual-reference)),
            'gradient_max_error':max(float((a-r).abs().max()) for a,r in zip(ga,gr)),
            'masked_rows_zero_gradient':True,'identical_views_zero_jsd':True,'zero_weight_equals_matched_control':True,
            'same_geometry_and_single_rng_advance':True,'original_label_preserved':True,
            'no_synthetic_overlay_on_real_eyewear':True,'samples':summaries,'train_count':len(data)}
    Path('training/noon-20260928/consistency-numerical-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
