"""CPU forward/backward at production width/modes/grid, batch 1, not GPU memory testing."""
import gc
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import REQUIRED,build_model,ModelConfig,preset

torch.set_num_threads(1)
for name in REQUIRED:
    torch.manual_seed(99)
    model=build_model(ModelConfig(),name)
    history=torch.randn(1,5,128,128)*.1
    velocity=torch.full((1,1,128,128),3000.)
    horizon=preset(name)['horizon']
    prediction=model(history,velocity,torch.tensor([50]),horizon)
    assert prediction.shape==(1,horizon,128,128)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('DEFAULT WIDTH32 GRID128 FORWARD/BACKWARD PASS:',name,tuple(prediction.shape),flush=True)
    del model,prediction,history,velocity
    gc.collect()
