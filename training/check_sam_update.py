"""Numerical guards for the two-pass optimizer, not an accuracy benchmark."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from train_camera_sam import sam_update


def main():
    torch.set_num_threads(2)
    torch.manual_seed(817)
    m = nn.Linear(3, 2).double()
    x, y = torch.randn(5, 3, dtype=torch.double), torch.randn(5, 2, dtype=torch.double)
    a, b = copy.deepcopy(m), copy.deepcopy(m)
    opt = torch.optim.SGD(a.parameters(), lr=.1)
    reference = torch.optim.SGD(b.parameters(), lr=.1)
    def closure_a():
        out = a(x)
        return (out-y).square().mean(), out
    sam_update(a, opt, closure_a, rho=.05, clip_norm=100.)
    loss = (b(x)-y).square().mean()
    grads = torch.autograd.grad(loss, tuple(b.parameters()))
    norm = torch.cat([g.reshape(-1) for g in grads]).norm()
    originals = [p.detach().clone() for p in b.parameters()]
    with torch.no_grad():
        for p, g in zip(b.parameters(), grads):
            p.add_(.05*g/norm)
    perturbed = (b(x)-y).square().mean()
    perturbed.backward()
    with torch.no_grad():
        for p, original in zip(b.parameters(), originals): p.copy_(original)
    reference.step()
    error = max((p-q).abs().max().item() for p,q in zip(a.parameters(),b.parameters()))
    assert error < 1e-8, error
    a, b = copy.deepcopy(m), copy.deepcopy(m)
    opt = torch.optim.AdamW(a.parameters(), lr=.001)
    reference = torch.optim.AdamW(b.parameters(), lr=.001)
    sam_update(a, opt, closure_a, rho=0.)
    (b(x)-y).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
    reference.step()
    zero_error=max((p-q).abs().max().item() for p,q in zip(a.parameters(),b.parameters()))
    assert zero_error == 0., zero_error
    before = [p.detach().clone() for p in a.parameters()]
    calls = 0
    def failing():
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError('injected second-pass failure')
        return closure_a()
    try: sam_update(a,opt,failing)
    except RuntimeError as e: assert str(e) == 'injected second-pass failure'
    else: raise AssertionError('Failure was not propagated')
    assert all(torch.equal(p,q) for p,q in zip(a.parameters(),before))
    result = {'algorithm_reference_max_error':error, 'rho_zero_adamw_max_error':zero_error,
              'second_pass_failure_restores_exact_parameters':True}
    path=Path('training/noon-20260928/sam-numerical-check.json')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': main()
