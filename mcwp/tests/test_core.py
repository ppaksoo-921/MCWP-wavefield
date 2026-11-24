import sys
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ModelConfig, MediumConditionedWavePropagator
from physics import AcousticConsistencyLoss
from data import WaveDataset, EpochBatchSampler
from evaluate import load_case, metrics


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_model_forward_backward_and_medium_dependence(self):
        torch.manual_seed(0)
        m=MediumConditionedWavePropagator(ModelConfig(width=4,modes=2))
        h=torch.randn(2,5,128,128)*.1
        vp=torch.ones(2,1,128,128)*3000
        out=m(h,vp,torch.tensor([1,50]),horizon=3)
        self.assertEqual(tuple(out.shape),(2,3,128,128))
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        with torch.no_grad():
            shifted=m(h,vp+500,torch.tensor([1,50]),horizon=3)
        self.assertGreater(float((shifted-out).abs().max().detach()),0)

    def test_physics_zero_at_truth_and_positive_perturbation(self):
        loss=AcousticConsistencyLoss()
        ref=torch.randn(2,3,128,128)*.1
        prev=torch.randn(2,1,128,128)*.1
        vp=torch.ones(2,1,128,128)*3000
        valid=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.float32)
        exact, diag=loss(ref,ref,prev,vp,valid)
        self.assertTrue(torch.equal(exact,torch.zeros_like(exact)))
        self.assertGreater(float(diag[0]),0)  # true coarse residual need NOT be zero
        pred=(ref+.01*torch.randn_like(ref)).requires_grad_()
        value,_=loss(pred,ref,prev,vp,valid)
        self.assertGreater(float(value[0].detach()),0)
        self.assertEqual(float(value[1].detach()),0)  # late padded data excluded
        value.mean().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()),0)

    def test_residual_difference_identity(self):
        phy=AcousticConsistencyLoss()
        a=torch.randn(1,3,128,128)*.01
        b=torch.randn_like(a)*.01
        prev=torch.randn(1,1,128,128)*.01
        vp=torch.ones_like(prev)*3000
        left=phy.residual(torch.cat((prev,a),1),vp)-phy.residual(torch.cat((prev,b),1),vp)
        right=phy.residual(torch.cat((torch.zeros_like(prev),a-b),1),vp)
        self.assertTrue(torch.allclose(left,right,atol=3e-6,rtol=1e-5))

    def test_sampler_resume_and_rank_partition(self):
        rank0=list(EpochBatchSampler(100,12,0,2,123,3))
        rank1=list(EpochBatchSampler(100,12,1,2,123,3))
        resumed=list(EpochBatchSampler(100,12,0,2,123,3,offset=4))
        self.assertEqual(rank0[4:],resumed)
        used=[i for batch in rank0+rank1 for i,n in batch]
        self.assertEqual(len(used),96)
        self.assertEqual(len(set(used)),96)
        self.assertTrue(all(1<=n<=100 for batch in rank0 for i,n in batch))

    def test_hdf5_history_order_and_terminal_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            u=np.broadcast_to(np.arange(101,dtype=np.float32)[:,None,None],(101,128,128))
            with h5py.File(Path(temp)/'test.h5','w') as f:
                f['snaps']=u; f['vp']=np.ones((128,128),np.float32)*3000; f['loc_x']=12
            ds=WaveDataset(temp,3)
            b=ds[(0,50)]
            np.testing.assert_array_equal(b['history'][:,0,0],[49,48,47,46,45])
            np.testing.assert_array_equal(b['target'][:,0,0],[50,51,52])
            b=ds[(0,100)]
            np.testing.assert_array_equal(b['valid'],[1,0,0])
            b=ds[(0,1)]
            np.testing.assert_array_equal(b['history'][:,0,0],[0,0,0,0,0])
            ds.close()

    def test_metric_denominators(self):
        ref=np.ones((101,4,4),np.float32)
        pred=ref.copy(); pred[1:]+=1
        m=metrics(pred,ref)
        self.assertAlmostEqual(m['mae_all101'],100/101)
        self.assertEqual(m['mae_pred100'],1)
        self.assertAlmostEqual(m['snr_db_pred100'],0)

if __name__=='__main__':
    unittest.main(verbosity=2)
