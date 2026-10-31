"""Summarize only measured fields; unavailable GPU work stays blank."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'_extras'))
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args()
rows=[json.loads(l) for l in (a.run_dir/'results.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
summary=[]
for variant in ['full','lazy_hit','lazy_rest']:
    group=[r for r in rows if r['variant']==variant]
    if not group:continue
    latency=[r['local_wall_seconds'] for r in group]
    corrected=[r['local_wall_seconds'] for r in group if r['correction']]
    def mean(values):return float(np.mean(values)) if values else None
    def percentile(values):return float(np.percentile(values,95)) if values else None
    summary.append({'variant':variant,'mode':group[0]['mode'],'n_observations':len(group),'unique_queries':len({r['query_id'] for r in group}),'mean_latency_s':mean(latency),'p95_latency_s':percentile(latency),'mean_correction_latency_s':mean(corrected),'p95_correction_latency_s':percentile(corrected),'correction_observations':len(corrected),'mean_grade_calls':mean([r['n_grade'] for r in group]),'mean_input_tokens':mean([r.get('input_tokens',0) for r in group]),'mean_output_tokens':mean([r.get('output_tokens',0) for r in group]),'accuracy':mean([r['acc'] for r in group]),'f1':mean([r['f1'] for r in group]),'mean_model_server_seconds':mean([r['model_server_seconds'] for r in group if r.get('model_server_seconds') is not None]),'gpu_service_seconds':None,'gpu_status':'LOCAL_DEVICE_TELEMETRY_SEPARATE_SERVER_TIME_MAY_INCLUDE_CPU' if group[0]['mode']=='local_ollama_local_faiss' else 'NOT_MEASURED'})
with (a.run_dir/'measurement_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('Saved measurement_summary.csv. See run mode/config for timing scope. GPU service is NOT measured.')

# Figures are generated from measured rows only; synthetic data is labelled.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
figures=a.run_dir/'figures';figures.mkdir(exist_ok=True)
labels=[r['variant'] for r in summary]
mode=summary[0]['mode']
for filename,title,keys in [
    ('latency','Response time (s)',['mean_latency_s','p95_latency_s']),
    ('quality','Answer quality',['accuracy','f1']),
    ('grading_calls','Mean grading calls',['mean_grade_calls']),
    ('correction_latency','Correction response time (s)',['mean_correction_latency_s','p95_correction_latency_s']),
    ('tokens','Mean token usage',['mean_input_tokens','mean_output_tokens'])]:
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(len(labels));width=.75/len(keys)
    for i,key in enumerate(keys):
        values=[r[key] if r[key] is not None else np.nan for r in summary]
        ax.bar(x+i*width,values,width,label=key)
    ax.set_xticks(x+width*(len(keys)-1)/2,labels)
    ax.set_title(title+'\n'+mode,fontsize=10)
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/(filename+'.png'),dpi=160);plt.close(fig)
lines=['# Grading results','', 'Mode: '+mode, '', 'Pure GPU kernel time is NOT MEASURED. For local Ollama, model-server duration and local device telemetry are separate metrics; CPU offloading may contribute.', '', '| variant | queries | mean s | p95 s | grade calls | accuracy | F1 | correction n |','|---|---|---|---|---|---|---|---|']
for r in summary:
    lines.append(f"| {r['variant']} | {r['unique_queries']} | {r['mean_latency_s']:.3f} | {r['p95_latency_s']:.3f} | {r['mean_grade_calls']:.2f} | {r['accuracy']:.3f} | {r['f1']:.3f} | {r['correction_observations']} |")
lines+=['','No correction observations means its latency is missing, not zero.','Failure count: '+str(len((a.run_dir/'errors.jsonl').read_text(encoding='utf-8').splitlines()) if (a.run_dir/'errors.jsonl').exists() else 0), 'Pilot/small-query results do not establish stable p95 or statistical significance.','Inspect config.json for corpus, cache and correction settings. Do not compare absolute API latency to teammate local GPU results.','Group differences require paired complete query IDs; failures must be resolved before drawing conclusions.']
(a.run_dir/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print('Saved report.md and five figures.')
# Optional local-device telemetry collected during new runs only.
gpu_path=a.run_dir/'local_gpu_summary.csv'
if gpu_path.exists():
    with gpu_path.open(encoding='utf-8-sig') as f:gpu_rows=list(csv.DictReader(f))
    measured=[r for r in gpu_rows if r['phase']=='measured']
    if measured:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,key,title in [(axes[0],'mean_utilization_percent','Local GPU mean utilization (%)'),(axes[1],'max_memory_used_mib','Local GPU peak memory (MiB)')]:
            labels=[r['variant']+' / GPU '+r['gpu_index'] for r in measured]
            vals=[float(r[key]) if r[key] else np.nan for r in measured]
            ax.bar(labels,vals);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',labelrotation=20)
        fig.suptitle('Whole local device; NOT OpenAI model GPU work',fontsize=10)
        fig.tight_layout();fig.savefig(figures/'local_gpu.png',dpi=160);plt.close(fig)
    with (a.run_dir/'report.md').open('a',encoding='utf-8') as f:
        f.write('\n\nLocal GPU telemetry: local_gpu_samples.csv and local_gpu_summary.csv. Whole-device samples include other applications. Remote model GPU service remains unavailable. Setup/model load excluded; warmup separately labelled. See local_gpu_status.json for coverage and errors.\n')

if any(r.get('mean_model_server_seconds') is not None for r in summary):
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar([r['variant'] for r in summary],[r.get('mean_model_server_seconds') or 0 for r in summary])
    ax.set_title('Mean local model-server workload (s)\nIncludes CPU offload if present; not pure GPU kernel time')
    fig.tight_layout();fig.savefig(figures/'model_server_workload.png',dpi=160);plt.close(fig)
