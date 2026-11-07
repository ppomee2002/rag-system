"""Plot replay metrics. Seed standard deviations are not confidence intervals."""
import argparse
import csv
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'experiments/grading_optimization/_extras'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args()
with (a.run_dir/'sim_results.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
out=a.run_dir/'figures';out.mkdir(exist_ok=True)
if rows[0].get('study') == 'threshold_tradeoff':
    labels = list(dict.fromkeys(r['label'] for r in rows))
    loads = sorted({float(r['load']) for r in rows})
    def values(label, load, metric):
        return np.array([float(r[metric]) for r in rows if r['label']==label and float(r['load'])==load])
    for metric, title in [('mean','Overall mean E2E (s)'), ('p95','Overall p95 E2E (s)'),
                          ('mean_correct','Non-correction mean E2E (s)'), ('p95_correct','Non-correction p95 E2E (s)'),
                          ('mean_corrected','Correction mean E2E (s)'), ('p95_corrected','Correction p95 E2E (s)'),
                          ('mean_gpu_wait','Mean cumulative queue wait (s)'), ('max_gpu_wait','Mean of seed maximum queue wait (s)')]:
        fig,ax=plt.subplots(figsize=(10,5.5))
        for label in labels:
            data=[values(label,load,metric) for load in loads]
            style='--' if label.startswith('T=') else '-'
            ax.errorbar(loads,[v.mean() for v in data],yerr=[v.std(ddof=1) for v in data],
                        marker='o',linestyle=style,capsize=2,label=label,alpha=.85)
        ax.set_xlabel('Offered load rho');ax.set_ylabel(title)
        ax.set_title('Full trace / original persistent threshold policy\nSeed SD error bars; no composite-score selection')
        ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
        fig.tight_layout();target=out/(metric+'.png')
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    for ax,load in zip(axes.flat,loads):
        for family,color in [('Aging=','tab:orange'),('T=','tab:blue')]:
            family_labels=[s for s in labels if s.startswith(family)]
            xs=[values(s,load,'mean').mean() for s in family_labels]
            ys=[values(s,load,'p95_corrected').mean() for s in family_labels]
            ax.plot(xs,ys,'o--',color=color,label='Aging' if family=='Aging=' else 'Threshold')
            for i,(s,x,y) in enumerate(zip(family_labels,xs,ys)):
                ax.annotate(s,(x,y),xytext=(5,5 if i%2==0 else -13),textcoords='offset points',fontsize=7,color=color)
        x=values('SRPT',load,'mean').mean();y=values('SRPT',load,'p95_corrected').mean()
        ax.scatter([x],[y],marker='*',s=130,color='black',label='SRPT')
        ax.annotate('SRPT',(x,y),xytext=(5,5),textcoords='offset points',fontsize=8)
        ax.set_title(f'rho={load}');ax.set_xlabel('Overall mean E2E (s)');ax.set_ylabel('Correction p95 E2E (s)')
        ax.grid(alpha=.2);ax.margins(.18);ax.legend(fontsize=7)
    fig.suptitle('Mean E2E vs Correction tail latency / lower-left is better\nAll prespecified thresholds shown; seed-averaged metrics',fontsize=13)
    fig.tight_layout();target=out/'tradeoff.png'
    if target.exists():raise FileExistsError(target)
    fig.savefig(target,dpi=160);plt.close(fig)
    print('Saved threshold tradeoff figures:',out)
    raise SystemExit(0)


is_study='stage' in rows[0]
metrics=['mean','p95','mean_corrected','p95_corrected','mean_gpu_wait','throughput']
if is_study:
    rows=[r for r in rows if r['stage']=='evaluation']
    metrics+=['max_gpu_wait','threshold_exceed_fraction','mean_correct','p95_correct']

def label(r):
    suffix=f" + aging {r['aging_alpha']}" if float(r['aging_alpha']) else ''
    if r.get('threshold_seconds'):suffix+=f" T={r['threshold_seconds']}s"
    return r['policy']+suffix

def curves(ax, data, metric):
    for name in sorted({label(r) for r in data}):
        group=[r for r in data if label(r)==name]
        loads=sorted({float(r['load']) for r in group})
        values=[np.array([float(r[metric]) for r in group if float(r['load'])==load]) for load in loads]
        means=[np.nanmean(v) for v in values]
        sd=[np.nanstd(v,ddof=1) if len(v)>1 else 0 for v in values]
        ax.errorbar(loads,means,yerr=sd,marker='o',capsize=2,label=name)
    ax.set_xlabel('Offered load rho (Full work basis)' if is_study else 'Load rho')
    ax.set_ylabel(metric);ax.legend(fontsize=7);ax.grid(alpha=.2)

for metric in metrics:
    if is_study:
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,arm in zip(axes,['Scheduling Only','Combined']):
            curves(ax,[r for r in rows if r['arm']==arm],metric)
            ax.set_title(arm+(' / Full: primary comparison' if arm=='Scheduling Only' else ' / Lazy-hit: below accuracy criterion'),fontsize=10)
    else:
        fig,ax=plt.subplots(figsize=(8,5));curves(ax,rows,metric)
    fig.suptitle('Fixed-trace replay / error bars: seed SD / no live GPU measurement',fontsize=11)
    fig.tight_layout()
    target=out/(metric+'.png')
    if target.exists():raise FileExistsError(target)
    fig.savefig(target,dpi=160);plt.close(fig)
if is_study:
    chosen_policy = 'srpt_rescue' if any(r['policy']=='srpt_rescue' for r in rows) else 'srpt_threshold'
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric in zip(axes,['mean','p95_corrected']):
        for arm in ['Baseline','Grading Only','Scheduling Only','Combined']:
            group=[r for r in rows if r['arm']==arm and (r['policy']=='fifo_call' or r['policy']==chosen_policy)]
            loads=sorted({float(r['load']) for r in group})
            ax.plot(loads,[np.mean([float(r[metric]) for r in group if float(r['load'])==v]) for v in loads],marker='o',label=arm)
        ax.set_xlabel('Offered load rho (Full work basis)');ax.set_ylabel(metric);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Four arms: selected threshold / Lazy-hit arms fail historical accuracy criterion',fontsize=11)
    fig.tight_layout();target=out/'four_arms.png'
    if target.exists():raise FileExistsError(target)
    fig.savefig(target,dpi=160);plt.close(fig)
print('Saved replay figures:',out)
