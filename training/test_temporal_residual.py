import copy
import unittest
import numpy as np
import torch
from temporal_residual import TemporalResidual, video_windows


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_model_is_baseline_and_trained_static_is_unchanged(self):
        torch.manual_seed(19)
        model=TemporalResidual().eval()
        x=torch.rand(4,8,7);x/=x.sum(-1,keepdim=True)
        with torch.no_grad():
            self.assertLess((model(x).softmax(-1)-x[:,-1]).abs().max().item(),1e-7)
            torch.nn.init.normal_(model.head.weight,std=.4)
            static=x[:,-1:].expand_as(x)
            self.assertLess((model(static).softmax(-1)-x[:,-1]).abs().max().item(),1e-7)
            self.assertGreater((model(x).softmax(-1)-x[:,-1]).abs().max().item(),1e-4)

    def test_optimizer_really_changes_weights(self):
        torch.manual_seed(7)
        model=TemporalResidual()
        old=model.head.weight.detach().clone()
        x=torch.rand(8,8,7);x/=x.sum(-1,keepdim=True)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
        loss=torch.nn.functional.cross_entropy(model(x),torch.arange(8)%7)
        loss.backward();opt.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater((model.head.weight-old).abs().max().item(),0)

    def test_causal_windows_and_missing_face_reset(self):
        frames=[]
        for i in range(12):
            p=np.arange(1,8,dtype=np.float32)+i*.1;p/=p.sum()
            frames.append({'status':'detected','probabilities':p.tolist(),'seconds':i*.1,
                           'bbox':{'x':10,'y':10,'width':100,'height':100}})
        original=video_windows({'frames':frames})
        prefix=video_windows({'frames':frames[:6]})
        np.testing.assert_array_equal(prefix[-1][1],original[5][1])
        altered=copy.deepcopy(frames)
        altered[5]={'status':'no_face','seconds':.5}
        windows=dict(video_windows({'frames':altered}))
        self.assertNotIn(5,windows)
        np.testing.assert_array_equal(windows[6],np.repeat(windows[6][-1:],8,axis=0))
        altered=copy.deepcopy(frames)
        altered[6]['bbox']['x']=1000
        windows=dict(video_windows({'frames':altered}))
        np.testing.assert_array_equal(windows[6],np.repeat(windows[6][-1:],8,axis=0))


if __name__=='__main__':
    unittest.main()
