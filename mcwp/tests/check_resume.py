"""Optional standalone CPU integration test. Creates only temporary test files."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import trusted_load

ROOT=Path(__file__).resolve().parents[1]
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
with tempfile.TemporaryDirectory(prefix='mcwp_resume_test_') as temp:
    root=Path(temp)
    subprocess.run([sys.executable,str(ROOT/'tests/make_fixture.py'),str(root/'h5')],check=True)
    def train(name,steps,resume=False):
        cmd=[sys.executable,str(ROOT/'train.py'),'--variant','physics','--data-dir',str(root/'h5'),
             '--run-dir',str(root/name),'--cpu','--width','4','--modes','2','--global-batch','4',
             '--workers','0','--max-steps',str(steps),'--log-every','1','--save-every','2',
             '--eval-every','2','--preserve-full','','--eval-at','']
        if resume:
            cmd += ['--resume','auto']
        subprocess.run(cmd,check=True)
    train('continuous',4)
    train('resumed',2)
    train('resumed',4,True)
    a=trusted_load(root/'continuous/checkpoints/resume_000004.pt')
    b=trusted_load(root/'resumed/checkpoints/resume_000004.pt')
    for part in ('model','ema'):
        assert all(torch.equal(v,b[part][k]) for k,v in a[part].items()),part
    assert torch.equal(a['causal']['table'],b['causal']['table'])
    for p,state in a['optimizer']['state'].items():
        for name,value in state.items():
            other=b['optimizer']['state'][p][name]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other
    print('RESUME EQUIVALENCE PASS: model, EMA, causal and optimizer states match on CPU.')
