"""CPU software checks; these are not physical-accuracy benchmark results."""
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import (ModelConfig, PRESETS, REQUIRED, build_model, preset,
                   parameter_table, StaticFusionBlock, StaticWaveBlock)
from reference_model import MediumConditionedWavePropagator, MediumConditionedBlock
from data import WaveDataset, EpochBatchSampler
from evaluate import metrics


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self, size=32):
        torch.manual_seed(11)
        h = torch.randn(2, 5, size, size) * .1
        vp = torch.ones(2, 1, size, size) * 3000
        return h, vp, torch.tensor([1, 50])

    def test_all_presets_forward_backward_no_unused_parameters(self):
        h, vp, index = self.inputs()
        for name in PRESETS:
            with self.subTest(variant=name):
                torch.manual_seed(20260909)
                model = build_model(ModelConfig(width=4, modes=2), name)
                horizon = preset(name)['horizon']
                y = model(h, vp, index, horizon)
                self.assertEqual(tuple(y.shape), (2, horizon, 32, 32))
                y.square().mean().backward()
                missing = [n for n, p in model.named_parameters() if p.grad is None]
                self.assertEqual(missing, [])
                self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters()))
                with torch.no_grad():
                    y2 = model(h, vp + 500, index, horizon)
                self.assertGreater((y-y2).abs().max().item(), 0)

    def test_base_and_horizon1_weights_and_predictions_identical_to_original(self):
        h, vp, index = self.inputs()
        cfg = ModelConfig(width=4, modes=2)
        torch.manual_seed(17)
        ref = MediumConditionedWavePropagator(cfg)
        for name in ('base', 'horizon1', 'first_order'):
            torch.manual_seed(17)
            new = build_model(cfg, name)
            self.assertEqual(list(ref.state_dict()), list(new.state_dict()))
            self.assertTrue(all(torch.equal(v, new.state_dict()[k]) for k, v in ref.state_dict().items()))
            if name != 'first_order':
                self.assertTrue(torch.equal(ref(h, vp, index, 1), new(h, vp, index, 1)))

    def test_first_order_changes_only_explicit_output_prior(self):
        h, vp, index = self.inputs()
        cfg = ModelConfig(width=4, modes=2)
        torch.manual_seed(7); a = build_model(cfg, 'base')
        torch.manual_seed(7); b = build_model(cfg, 'first_order')
        ya, yb = a(h, vp, index, 1), b(h, vp, index, 1)
        expected = (index >= 2).float()[:, None, None, None] * (h[:, :1]-h[:, 1:2])
        self.assertTrue(torch.allclose(ya-yb, expected, atol=2e-7, rtol=2e-6))
        self.assertTrue(torch.equal(ya[0], yb[0]))

    def test_static_fusion_keeps_branches_but_removes_dynamic_modulation(self):
        m = build_model(ModelConfig(width=4, modes=2), 'static_fusion')
        self.assertTrue(hasattr(m, 'medium_stem'))
        self.assertTrue(hasattr(m, 'temporal_stem'))
        self.assertFalse(any(isinstance(x, MediumConditionedBlock) for x in m.modules()))
        self.assertEqual(sum(isinstance(x, StaticFusionBlock) for x in m.modules()), 7)
        for block in list(m.encoder_interactions) + list(m.decoder_interactions):
            self.assertEqual(tuple(block.logits.shape), (3,))
            self.assertFalse(hasattr(block, 'mod'))
            self.assertFalse(hasattr(block, 'gates'))

    def test_monolithic_really_has_one_joint_encoder(self):
        m = build_model(ModelConfig(width=4, modes=2), 'monolithic')
        self.assertEqual(m.joint_stem.conv1.in_channels, 10)
        self.assertFalse(hasattr(m, 'medium_stem'))
        self.assertFalse(hasattr(m, 'medium_down'))
        self.assertFalse(hasattr(m, 'temporal_stem'))
        self.assertEqual(sum(isinstance(x, StaticWaveBlock) for x in m.modules()), 7)

    def test_shared_weights_preserved_for_architectural_controls(self):
        cfg = ModelConfig(width=4, modes=2)
        torch.manual_seed(101); ref = build_model(cfg, 'base')
        for name in ('static_fusion', 'monolithic'):
            torch.manual_seed(101); m = build_model(cfg, name)
            for child in ('time_embedding', 'wave_down', 'spectral', 'up', 'head'):
                a, b = getattr(ref, child).state_dict(), getattr(m, child).state_dict()
                self.assertTrue(all(torch.equal(v, b[k]) for k, v in a.items()), (name, child))

    def test_published_width32_parameter_match(self):
        rows = parameter_table()
        self.assertEqual(rows[0]['parameters'], 6307702)
        self.assertTrue(all(abs(row['difference_pct']) < .04 for row in rows))

    def test_sampler_same_h1_h3_and_exact_resume_position(self):
        a = list(EpochBatchSampler(100, 12, 0, 2, 123, 3))
        b = list(EpochBatchSampler(100, 12, 1, 2, 123, 3))
        self.assertEqual(a[4:], list(EpochBatchSampler(100, 12, 0, 2, 123, 3, 4)))
        used = [i for batch in a+b for i, n in batch]
        self.assertEqual(len(used), len(set(used)))
        # The sample index generator contains no horizon/model dependency.
        self.assertEqual(a, list(EpochBatchSampler(100, 12, 0, 2, 123, 3)))

    def test_history_and_terminal_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            u = np.broadcast_to(np.arange(101, dtype=np.float32)[:, None, None], (101, 128, 128))
            with h5py.File(Path(temp)/'test.h5', 'w') as f:
                f['snaps'] = u; f['vp'] = np.ones((128, 128), np.float32)*3000; f['loc_x'] = 12
            for horizon in (1, 3):
                ds = WaveDataset(temp, horizon)
                b = ds[(0, 50)]
                np.testing.assert_array_equal(b['history'][:, 0, 0], [49, 48, 47, 46, 45])
                self.assertEqual(b['target'].shape[0], horizon)
                b = ds[(0, 100)]
                self.assertEqual(b['valid'].sum(), 1)
                ds.close()

    def test_mae_denominators(self):
        ref = np.ones((101, 4, 4), np.float32)
        pred = ref.copy(); pred[1:] += 1
        result = metrics(pred, ref)
        self.assertAlmostEqual(result['mae_all101'], 100/101)
        self.assertEqual(result['mae_pred100'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
