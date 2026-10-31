"""Launch full local experiment on the user's Windows host, no OpenAI key needed."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNTIME=Path(r'C:\Users\lolam\Documents\Codex\2026-10-31\llm-rag-2-6\work\ollama')

def request(path,payload=None,timeout=30):
    req=urllib.request.Request('http://127.0.0.1:11434'+path,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=50);p.add_argument('--with-scheduling',action='store_true');p.add_argument('--run-measurement',action='store_true');p.add_argument('--repeats',type=int,default=1);p.add_argument('--setup-only',action='store_true');a=p.parse_args()
    os.chdir(ROOT)
    if not 1<=a.limit<=200:p.error('limit must be 1..200')
    if not a.run_measurement and not a.setup_only:
        corpus=ROOT/'data/grading_optimization'/f'corpus_{a.limit}'
        if not (corpus/'manifest.json').exists():
            subprocess.run([sys.executable,str(HERE/'prepare_retrieval.py'),'--limit',str(a.limit),'--full-text'],check=True)
        print('PREPARATION_COMPLETE. Waiting for explicit measurement authorization. No Grading or Scheduling run started.',flush=True)
        return
    if not (RUNTIME/'ollama.exe').exists():raise RuntimeError('Portable Ollama runtime missing')
    try:version=request('/api/version',timeout=2)
    except Exception:
        env=os.environ.copy();env.update({'OLLAMA_HOST':'127.0.0.1:11434','OLLAMA_MODELS':str(RUNTIME/'models'),'OLLAMA_NUM_PARALLEL':'1','OLLAMA_MAX_LOADED_MODELS':'1','OLLAMA_KEEP_ALIVE':'-1','OLLAMA_VULKAN':'1','OLLAMA_LLM_LIBRARY':'vulkan','OLLAMA_NO_CLOUD':'1'})
        logs=(RUNTIME/'host_server.log').open('a',encoding='utf-8')
        subprocess.Popen([str(RUNTIME/'ollama.exe'),'serve'],cwd=RUNTIME,env=env,stdout=logs,stderr=logs,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(60):
            try:version=request('/api/version',timeout=2);break
            except Exception:time.sleep(1)
        else:raise RuntimeError('Local server did not start; inspect '+str(RUNTIME/'host_server.log'))
    print('Local Ollama:',version,flush=True)
    tags=request('/api/tags')
    if not any(m.get('name')=='llama3.1:8b' for m in tags.get('models',[])):
        print('Downloading llama3.1:8b on local server...',flush=True)
        request('/api/pull',{'model':'llama3.1:8b','stream':False},timeout=3600)
    print('Verifying actual GPU inference...',flush=True)
    result=request('/api/generate',{'model':'llama3.1:8b','prompt':'Return only the word ready.','stream':False,'keep_alive':-1,'options':{'num_ctx':6144,'num_predict':8,'temperature':0}},timeout=900)
    placement=request('/api/ps')
    print('Model placement:',json.dumps(placement),flush=True)
    gpu=next((m for m in placement.get('models',[]) if m.get('name')=='llama3.1:8b'),None)
    if not gpu or gpu.get('size_vram',0)<=0:raise RuntimeError('GPU inference not confirmed. No full experiment started.')
    print('GPU memory confirmed:',gpu['size_vram'],'bytes. CPU offload may also be present.',flush=True)
    print('Initial server response seconds:',result.get('total_duration',0)/1e9,flush=True)
    validation=ROOT/'results/grading_optimization/local_setup_validation.json'
    validation.write_text(json.dumps({'ollama':version,'model_placement':placement,'initial_response_seconds':result.get('total_duration',0)/1e9,'backend':'Vulkan requested','driver_change':False,'gpu_memory_confirmed':True,'note':'This validates local inference only, not 200-query experiment completion.'},indent=2),encoding='utf-8')
    if a.setup_only:return
    corpus=ROOT/'data/grading_optimization'/f'corpus_{a.limit}'
    if not (corpus/'manifest.json').exists():
        subprocess.run([sys.executable,str(HERE/'prepare_retrieval.py'),'--limit',str(a.limit),'--full-text'],check=True)
    command=[sys.executable,str(HERE/'run_live.py'),'--config',str(HERE/'config.local.json'),'--limit',str(a.limit),'--repeats',str(a.repeats)]
    print(f'Starting {a.limit}-query Full/Lazy-hit/Lazy-rest local evaluation. This may take hours; keep the laptop plugged in and awake.',flush=True)
    subprocess.run(command,check=True)
    if not a.with_scheduling:
        print('Grading finished. Scheduling was NOT started; request it separately.',flush=True)
        return
    candidates=[]
    for folder in (ROOT/'results/grading_optimization').iterdir():
        if (folder/'profile.jsonl').exists():
            run_cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
            if run_cfg.get('provider')=='ollama_native' and run_cfg.get('limit')==a.limit:
                candidates.append(folder)
    if not candidates:raise RuntimeError('No local full trace found for scheduling replay')
    run_dir=max(candidates,key=lambda path:path.stat().st_mtime)
    count=len((run_dir/'profile.jsonl').read_text(encoding='utf-8').splitlines())
    if count!=a.limit:raise RuntimeError('Incomplete Full traces; got '+str(count))
    scheduling=ROOT/'experiments/scheduling_optimization'
    print('Starting scheduling TRACE REPLAY using actual local model-server records (not live vLLM).',flush=True)
    subprocess.run([sys.executable,str(scheduling/'run.py'),'--profile',str(run_dir/'profile.jsonl'),'--config',str(scheduling/'config.local.json')],check=True)
    sim_dir=max((ROOT/'results/scheduling_optimization').iterdir(),key=lambda path:path.stat().st_mtime)
    subprocess.run([sys.executable,str(scheduling/'plot.py'),str(sim_dir)],check=True)
if __name__=='__main__':main()
