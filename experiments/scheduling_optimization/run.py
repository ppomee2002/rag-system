import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from simulator import Job, Call, job_from_row, attach_oof_predictions, fit_cost_model, simulate, summarize, poisson_arrivals
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def synthetic_jobs():
    rng=np.random.default_rng(42)
    jobs=[]
    for i in range(80):
        corrected=i%5==0
        phases=[('delay',.02)]+[('gpu',[Call('grade',j,100+j*30,float(rng.uniform(.5,1.5)))]) for j in range(5)]
        if corrected:phases += [('gpu',[Call('rewrite',0,100,2.)]),('delay',.5)]
        phases += [('gpu',[Call('generate',0,1000 if corrected else 500,float(rng.uniform(6,10) if corrected else rng.uniform(1,3)))])]
        work=sum(c.service for k,cs in phases if k=='gpu' for c in cs)
        solo=work+sum(v for k,v in phases if k=='delay')
        jobs.append(Job(str(i),'incorrect' if corrected else 'correct',np.array([1.,rng.random(),rng.random(),rng.random(),.1,1.]),phases,work,solo,n_docs=5))
    return jobs

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--demo',action='store_true')
    p.add_argument('--profile',type=Path)
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if not args.demo and args.profile is None:p.error('Specify --profile for real trace replay; --demo is SYNTHETIC validation only')
    jobs=synthetic_jobs() if args.demo else [job_from_row(json.loads(l)) for l in args.profile.read_text(encoding='utf-8-sig').splitlines() if l.strip() and json.loads(l)['pipeline']=='crag']
    if len(jobs)<10:p.error('At least 10 query traces required')
    attach_oof_predictions(jobs)
    model=fit_cost_model(jobs)
    mean_work=float(np.mean([j.work for j in jobs]))
    results=[]
    policies=[('fifo_call',0),('sjf_pred',0),('srpt_est',0)]+[('srpt_est',a) for a in cfg['aging_alphas']]
    for load in cfg['loads']:
        for seed in cfg['seeds']:
            rng=np.random.default_rng(seed)
            sample=[jobs[i] for i in rng.integers(0,len(jobs),cfg['n_queries'])]
            arrivals=poisson_arrivals(len(sample),load/mean_work,rng)
            for policy,alpha in policies:
                runs=simulate(sample,arrivals,policy,model,aging_alpha=alpha)
                assert all(np.isfinite(r.finish) and r.finish>=r.arrival for r in runs)
                results.append({'mode':'SYNTHETIC_VALIDATION' if args.demo else 'REAL_TRACE_SIMULATION','load':load,'seed':seed,'policy':policy,'aging_alpha':alpha,**summarize(runs,cfg['warmup_frac'])})
    out=ROOT/'results/scheduling_optimization'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
    with (out/'sim_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    lines=['# Scheduling comparison', '', 'SYNTHETIC VALIDATION ONLY; not PopQA or measured GPU results.' if args.demo else 'Trace replay simulation; not live vLLM validation.', '', '| load | policy | aging alpha | mean E2E | correction mean | correction p95 |','|---|---|---|---|---|---|']
    for load in cfg['loads']:
        for policy,alpha in policies:
            subset=[r for r in results if r['load']==load and r['policy']==policy and r['aging_alpha']==alpha]
            means=[np.nanmean([r[m] for r in subset]) for m in ['mean','mean_corrected','p95_corrected']]
            lines.append(f'| {load} | {policy} | {alpha} | '+ ' | '.join(f'{v:.3f}' for v in means)+' |')
    lines+=['','Age = cumulative GPU queue waiting seconds, including prior stages. Priority = baseline estimated remaining work - alpha * age. Queue priorities refreshed at every dispatch. Existing correction-aware SRPT is baseline. Calls are nonpreemptive.','Calibrate alpha on separate data, not final evaluation queries. No live vLLM scheduling integration implemented in this simulator.','Call cost model follows the inherited baseline and is fitted to the trace pool; query-level predictions are out-of-fold. This is trace replay, not validation on unseen hardware.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print('Saved:',out)
if __name__=='__main__':main()
