"""Plot real-trace scheduling replay; not live policy validation."""
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
for metric in ['mean','p95','mean_corrected','p95_corrected','mean_gpu_wait','throughput']:
    fig,ax=plt.subplots(figsize=(8,5))
    for policy,alpha in sorted({(r['policy'],float(r['aging_alpha'])) for r in rows}):
        group=[r for r in rows if r['policy']==policy and float(r['aging_alpha'])==alpha]
        loads=sorted({float(r['load']) for r in group})
        values=[np.array([float(r[metric]) for r in group if float(r['load'])==load]) for load in loads]
        means=[np.nanmean(v) for v in values]
        sd=[np.nanstd(v,ddof=1) if len(v)>1 else 0 for v in values]
        ax.errorbar(loads,means,yerr=sd,marker='o',capsize=2,label=policy+(f' + aging {alpha}' if alpha else ''))
    ax.set_xlabel('Load rho');ax.set_ylabel('Model-server queue wait (s)' if metric=='mean_gpu_wait' else metric)
    ax.set_title('Local model-server TRACE REPLAY (CPU offload may apply)\nError bars: standard deviation across arrival seeds; not live vLLM')
    ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/(metric+'.png'),dpi=160);plt.close(fig)
print('Saved 6 scheduling replay figures:',out)
