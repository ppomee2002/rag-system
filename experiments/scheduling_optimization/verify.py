import importlib.util
import sys
from pathlib import Path
import numpy as np
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'experiments/scheduling_optimization'))
import run as experiment
import simulator as new
spec=importlib.util.spec_from_file_location('old_sim',ROOT/'experiments/correction_scheduling/scheduling.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
jobs=experiment.synthetic_jobs();new.attach_oof_predictions(jobs);model=new.fit_cost_model(jobs)
rng=np.random.default_rng(7);arrivals=new.poisson_arrivals(len(jobs),.8/np.mean([j.work for j in jobs]),rng)
for policy in ['fifo_call','sjf_pred','srpt_est','srpt_oracle']:
    a=old.simulate(jobs,arrivals,policy,model)
    b=new.simulate(jobs,arrivals,policy,model,aging_alpha=0)
    assert np.allclose([r.finish for r in a],[r.finish for r in b])
aged=new.simulate(jobs,arrivals,'srpt_est',model,aging_alpha=.5)
assert not np.allclose([r.finish for r in aged],[r.finish for r in b])
assert all(r.gpu_wait>=0 and np.isfinite(r.finish) for r in aged)
print('PASS: all four unchanged policies match original simulator with alpha=0; aged jobs all complete with nonnegative wait.')
