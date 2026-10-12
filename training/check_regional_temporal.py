"""Critical invariants for a causal visual residual, independent of accuracy."""
import json
from pathlib import Path
import torch
from regional_temporal import RegionalTemporal,causal_indices

torch.set_num_threads(2);torch.manual_seed(918)
model=RegionalTemporal().eval()
x=torch.randn(3,8,3,384);base=torch.randn(3,7)
assert torch.equal(model(x,base),base)
with torch.no_grad():model.head.weight.normal_();model.head.bias.normal_()
static=x[:,-1:].expand_as(x).contiguous()
assert torch.equal(model(static,base),base)
box={'x':0,'y':0,'width':100,'height':100}
video={'frames':[{'status':'detected','feature_index':i,'seconds':i/10,'bbox':box} for i in range(10)]}
windows=causal_indices(video)
assert all(max(indices)<=fi for fi,indices in windows)
video['frames'][3]['status']='no_face'
windows=dict(causal_indices(video));assert windows[4]==[4]*8
video['frames'][7]['bbox']={'x':500,'y':500,'width':100,'height':100}
windows=dict(causal_indices(video));assert windows[7]==[7]*8 and windows[8]==[8]*8
model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
before=model.head.weight.detach().clone()
loss=torch.nn.functional.cross_entropy(model(x,base),torch.tensor([0,1,2]));loss.backward();optimizer.step()
assert not torch.equal(before,model.head.weight)
result={'zero_head_preserves_baseline':True,'trained_static_preserves_baseline_exactly':True,
        'no_future_frames':True,'no_face_and_iou_reset':True,'parameters_really_update':True}
Path('training/noon-20260928/regional-numerical-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
