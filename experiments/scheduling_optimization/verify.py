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
plain_srpt=new.simulate(jobs,arrivals,'srpt_est',model)
assert not np.allclose([r.finish for r in aged],[r.finish for r in plain_srpt])
assert all(r.gpu_wait>=0 and np.isfinite(r.finish) for r in aged)
print('PASS: all four unchanged policies match original simulator with alpha=0; aged jobs all complete with nonnegative wait.')

# Deterministic threshold checks: real event loop, known dispatch order.
class FixedModel:
    def call_est(self, kind, chars):
        return float(chars)
    def est_remaining(self, kind, idx, chars, n_docs, feats):
        return float(chars)

def toy_job(qid, phases, n_docs=0):
    work=sum(c.service for k,cs in phases if k=='gpu' for c in cs)
    delay=sum(v for k,v in phases if k=='delay')
    return new.Job(qid,'correct',np.ones(6),phases,work,work+delay,n_docs=n_docs)

def toy_case(grade_service=1.):
    return [toy_job('blocker',[('gpu',[new.Call('generate',0,1,10.)])]),
            toy_job('A',[('gpu',[new.Call('grade',0,20,grade_service)]),('gpu',[new.Call('generate',0,20,1.)])],1),
            toy_job('B',[('gpu',[new.Call('generate',0,1,1.)])])]

def dispatch(jobs,threshold):
    log=[]
    runs=new.simulate(jobs,[0.,1.,3.],'srpt_threshold',FixedModel(),wait_threshold=threshold,dispatch_log=log)
    for r in runs:
        assert np.isclose(r.finish-r.arrival,r.job.solo+r.gpu_wait)
    return runs,log

runs,log=dispatch(toy_case(),5.)
assert [(r[1],r[2]) for r in log]==[('blocker','generate'),('A','grade'),('A','generate'),('B','generate')],log
assert log[2][4]==9.  # prior grading wait carries forward to generation
assert log[1][0]==10.  # never preempt the running blocker
_,boundary=dispatch(toy_case(),9.)
assert boundary[1][1]=='B',boundary  # A wait == T is NOT overdue
_,long_call=dispatch(toy_case(20.),5.)
assert [r[1] for r in long_call]==['blocker','A','B','A'],long_call
assert long_call[2][0]==30.  # execution time is not added to A's age
huge,_=dispatch(toy_case(),1e12)
plain=new.simulate(toy_case(),[0.,1.,3.],'srpt_est',FixedModel())
assert np.allclose([r.finish for r in huge],[r.finish for r in plain])
metrics=new.summarize(runs,threshold_seconds=8.)
assert metrics['threshold_exceed_fraction']==2/3
assert metrics['max_gpu_wait']==9.
for threshold in [-1,float('nan'),None]:
    try:new.simulate(toy_case(),[0.,1.,3.],'srpt_threshold',FixedModel(),wait_threshold=threshold)
    except ValueError:pass
    else:raise AssertionError('invalid threshold accepted')
try:new.simulate(toy_case(),[0.,1.,3.],'srpt_threshold',FixedModel(),wait_threshold=5,aging_alpha=.5)
except ValueError:pass
else:raise AssertionError('Aging + threshold accepted')
print('PASS: strict threshold, cumulative wait across calls, longest wait, no execution aging, nonpreemption, SRPT fallback, metrics and invalid configuration.')
print('Example T=5 dispatch:',log)

# One-call rescue must return to SRPT after consuming eligibility.
rescue_log=[];rescue_stats={}
rescued=new.simulate(toy_case(),[0.,1.,3.],'srpt_rescue',FixedModel(),wait_threshold=5.,dispatch_log=rescue_log,dispatch_stats=rescue_stats)
assert [r[1] for r in rescue_log]==['blocker','A','B','A'],rescue_log
assert rescued[1].rescue_count==1 and rescued[1].gpu_wait==10.
assert rescued[1].last_rescue_wait==9.  # total wait survives eligibility reset
assert rescue_stats['srpt_override_dispatches']==1
assert rescue_stats['noncorrection_shorter_bypass_events']==1
assert rescue_stats['total_dispatches']==4
# Re-eligibility after additional wait; service time never ages eligibility.
repeat_jobs=toy_case()
repeat_jobs[2]=toy_job('B',[('gpu',[new.Call('generate',0,1,10.)])])
repeat_jobs.append(toy_job('C',[('gpu',[new.Call('generate',0,1,1.)])]))
repeat_log=[]
repeat_runs=new.simulate(repeat_jobs,[0.,1.,3.,20.],'srpt_rescue',FixedModel(),wait_threshold=5.,dispatch_log=repeat_log)
assert [r[1] for r in repeat_log]==['blocker','A','B','A','C'],repeat_log
assert repeat_runs[1].rescue_count==2
for r in repeat_runs:
    assert np.isclose(r.finish-r.arrival,r.job.solo+r.gpu_wait)
long_log=[]
new.simulate(toy_case(20.),[0.,1.,3.],'srpt_rescue',FixedModel(),wait_threshold=5.,dispatch_log=long_log)
assert [r[1] for r in long_log]==['blocker','A','B','A']
assert long_log[2][0]==30.
boundary_log=[]
new.simulate(toy_case(),[0.,1.,3.],'srpt_rescue',FixedModel(),wait_threshold=9.,dispatch_log=boundary_log)
assert boundary_log[1][1]=='B'
huge=new.simulate(toy_case(),[0.,1.,3.],'srpt_rescue',FixedModel(),wait_threshold=1e12)
plain=new.simulate(toy_case(),[0.,1.,3.],'srpt_est',FixedModel())
assert np.allclose([r.finish for r in huge],[r.finish for r in plain])
for threshold in [-1,float('nan'),None]:
    try:new.simulate(toy_case(),[0.,1.,3.],'srpt_rescue',FixedModel(),wait_threshold=threshold)
    except ValueError:pass
    else:raise AssertionError('invalid rescue threshold accepted')
try:new.simulate(toy_case(),[0.,1.,3.],'srpt_rescue',FixedModel(),wait_threshold=5,aging_alpha=.5)
except ValueError:pass
else:raise AssertionError('rescue combined with Aging')
print('PASS: one-call rescue, SRPT return, repeat eligibility, nonpreemption, preserved total wait, strict boundary, diagnostics, invalid parameters.')
print('One-call rescue dispatch:',rescue_log)
