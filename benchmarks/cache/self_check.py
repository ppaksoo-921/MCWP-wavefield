"""CPU numerical checks only; these timings are never research benchmark results."""
from collections import Counter
import json
from pathlib import Path
import unittest

import numpy as np
import torch

import cache_ops
import fd_adapter
from reference_model import ModelConfig, MediumConditionedWavePropagator
import run_cache

torch.set_num_threads(1)


class CacheChecks(unittest.TestCase):
    def test_full_architecture_prediction_parity(self):
        torch.manual_seed(24)
        model = MediumConditionedWavePropagator().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6307702)
        rng = np.random.default_rng(24)
        vp = rng.uniform(1450, 5000, (128,128)).astype('f4')
        ref = np.zeros((4,128,128), 'f4')
        ref[0] = rng.normal(0, .05, (128,128))
        a = cache_ops.predict(model, vp, ref, 'cpu', False).numpy()
        b = cache_ops.predict(model, vp, ref, 'cpu', True).numpy()
        check = run_cache.compare(a[1:], b[1:], 'Full model')
        self.assertTrue(check['passed'], check)

    def test_100_step_recursive_parity(self):
        torch.manual_seed(25)
        model = MediumConditionedWavePropagator(ModelConfig(width=4, modes=2, spectral_pad=1)).eval()
        # A nonzero correction exercises all cached blocks during the complete rollout.
        with torch.no_grad():
            model.head.weight.mul_(.01)
        rng = np.random.default_rng(25)
        vp = rng.uniform(1450, 5000, (32,32)).astype('f4')
        ref = np.zeros((101,32,32), 'f4')
        ref[0] = rng.normal(0, .05, (32,32))
        a = cache_ops.predict(model, vp, ref, 'cpu', False).numpy()
        b = cache_ops.predict(model, vp, ref, 'cpu', True).numpy()
        check = run_cache.compare(a[1:], b[1:], '100-step rollout')
        self.assertTrue(check['passed'], check)
        np.testing.assert_array_equal(a[0], ref[0])

    def test_initial_frame_excluded_from_accuracy(self):
        ref = np.ones((101,2,2), 'f4')
        pred = ref.copy(); pred[0] = 10000
        self.assertEqual(run_cache.metrics(pred,ref)['mae'], 0)

    def test_equal_group_runtime_aggregation(self):
        rows = []
        for group, n, value in [('A',3,1.),('B',1,9.)]:
            for case in range(n):
                for method in run_cache.METHODS:
                    for repeat in range(3):
                        rows.append(dict(group=group,case=group+str(case),method=method,
                            generation_wall_s=value, cuda_stream_interval_s=value,
                            generation_plus_cpu_output_s=value))
        _, _, overall = run_cache.summarize(rows)
        self.assertEqual(next(r['generation_wall_s'] for r in overall
                              if r['method']=='MCWP_cached' and r['weighting']=='equal_group'),5.)
        self.assertEqual(next(r['generation_wall_s'] for r in overall
                              if r['method']=='MCWP_cached' and r['weighting']=='equal_case'),3.)

    def test_catalog_has_final_raw_marmousi_and_all_groups(self):
        data = json.loads((Path(__file__).parent/'paper_cases_final66.json').read_text())['cases']
        self.assertEqual(dict(Counter(r['group'] for r in data)), run_cache.GROUPS)
        self.assertEqual(len({r['case'] for r in data}),66)
        for r in data:
            self.assertEqual(len(r['expected_sha256']),64)
            self.assertGreater(r['expected_mcwp_mae'],0)
        marm = [r for r in data if r['group']=='Marmousi_source']
        self.assertEqual(len(marm),15)
        self.assertTrue(all(r['case'].startswith('MarmousiOriginalV1500_5000_') for r in marm))

    def test_fd_adapter_initial_step_count_and_output_order(self):
        class Solver:
            def __init__(self, **kwargs):
                self.ext_cells = 0; self.nt = kwargs['nt']
                self.pressure = torch.zeros((2,2))
                for name in ('Vx','Vz','dVx','dVz','dpressure_Vx','dpressure_Vz',
                             'sigmaxx0','sigmaxx','sigmazz0','sigmazz','Vx0','Vz0'):
                    setattr(self,name,torch.zeros((2,2)))
            def set_method(self,x): pass
            def extend_wave_array(self): pass
            def build_boundary(self): pass
            def set_source_loc_torch(self,x): pass
            def reset_wavefields(self): self.pressure.zero_()
            def acoustic_iso_stress_torch(self,it,src): self.pressure.fill_(it)
            def acoustic_iso_stress_plus_torch(self): pass
            def acoustic_velocity_torch(self): pass
            def extract_matrix_torch(self): return self.pressure
        class Operator:
            def __init__(self, **kwargs): pass
            def to(self,device): return self
            def derx_backward_(self,*args): pass
            def derz_backward_(self,*args): pass
            def derx_forward_(self,*args): pass
            def derz_forward_(self,*args): pass
        from types import SimpleNamespace
        module = SimpleNamespace(acoustic_iso_torch=Solver, FD10StaggeredConv2d=Operator,
                                 set_wavelet_torch=lambda **kwargs: None)
        config = dict(run_cache.FD, nt=11, snapshot_stride=2)
        out = fd_adapter.generate(module,np.ones((2,2),'f4'),0,'cpu',config).numpy()
        np.testing.assert_array_equal(out[:,0,0], np.arange(0,11,2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
