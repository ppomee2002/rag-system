"""Actual LangGraph C-RAG evaluation with local BGE/FAISS and OpenAI/vLLM.
Correction is explicitly LOCAL FAISS re-search, not actual external web search.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TypedDict
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'_extras'))
from run import Backend
from gpu_monitor import GpuMonitor
from ollama_backend import OllamaBackend
from rag.pipelines import prompts as team_prompts
import re
from langgraph.graph import StateGraph, START, END
from rag.retrieval.faiss_retriever import FaissRetriever
from rag.utils.timing import StageTimer, stopwatch
from rag.eval.metrics import score_all

class State(TypedDict,total=False):
    row:dict
    variant:str
    docs:list
    scores:list
    index:int
    contexts:list
    action:str
    rewritten:str
    answer:str
    timer:object
    calls:list

def make_graph(backend,retriever,upper,lower,legacy_prompts=False):
    def retrieve(s):
        with s['timer'].stage('retrieval'):
            docs=retriever.retrieve(s['row']['question'],5)
        return {'docs':docs}
    def grade(s):
        i=s['index']
        with s['timer'].stage('grading'):
            prompt='Return ONLY JSON {"score": number from 0 to 1} rating how well the document supports answering the question.\nQuestion: '+s['row']['question']+'\nDocument: '+s['docs'][i].text
            if legacy_prompts:
                prompt=team_prompts.GRADE.format(question=s['row']['question'],document=s['docs'][i].text)
            raw=backend.request('grade',prompt,'',s['calls'])
            if legacy_prompts:
                match=re.search(r'\d*\.?\d+',raw)
                score=min(max(float(match.group()),0.0),1.0) if match else 0.0
            else:
                score=float(json.loads(raw)['score'])
            if not 0<=score<=1:raise ValueError('Invalid relevance score')
        return {'scores':s['scores']+[score],'index':i+1}
    def after_grade(s):
        hit=s['scores'][-1]>=upper
        return 'select' if s['index']==len(s['docs']) or (s['variant']!='full' and hit) else 'grade'
    def select(s):
        hit=next((i for i,v in enumerate(s['scores']) if v>=upper),None)
        if hit is not None:
            contexts=([s['docs'][hit]] if s['variant']=='lazy_hit' else s['docs'][hit:]) if s['variant']!='full' else [d for d,v in zip(s['docs'],s['scores']) if v>=upper]
            return {'contexts':contexts,'action':'correct'}
        action='incorrect' if all(v<lower for v in s['scores']) else 'ambiguous'
        contexts=[] if action=='incorrect' else [d for d,v in zip(s['docs'],s['scores']) if v>=lower]
        return {'contexts':contexts,'action':action}
    def rewrite(s):
        with s['timer'].stage('rewrite'):
            prompt=team_prompts.REWRITE.format(question=s['row']['question']) if legacy_prompts else 'Rewrite this question as a short English search query. Return only the query, without answering.\n'+s['row']['question']
            query=backend.request('rewrite',prompt,'',s['calls'])
        return {'rewritten':query}
    def search(s):
        with s['timer'].stage('web_search'):
            docs=retriever.retrieve(s['rewritten'],5)
        # Same local correction model for all variants; labelled in output/config.
        contexts=docs if s['action']=='incorrect' else s['contexts']+docs
        return {'contexts':contexts}
    def generate(s):
        with s['timer'].stage('generation'):
            prompt='Answer using the evidence. Return only a short answer; if evidence is insufficient return unknown.\nEvidence:\n'+'\n\n'.join(d.text for d in s['contexts'])+'\nQuestion: '+s['row']['question']
            if legacy_prompts:prompt=team_prompts.GENERATE.format(context=team_prompts.format_context(s['contexts']),question=s['row']['question'])
            answer=backend.request('generate',prompt,'',s['calls'])
        return {'answer':answer}
    g=StateGraph(State)
    for name,fn in [('retrieve',retrieve),('grade',grade),('select',select),('rewrite',rewrite),('search',search),('generate',generate)]:g.add_node(name,fn)
    g.add_edge(START,'retrieve');g.add_edge('retrieve','grade')
    g.add_conditional_edges('grade',after_grade,{'grade':'grade','select':'select'})
    g.add_conditional_edges('select',lambda s:'generate' if s['action']=='correct' else 'rewrite',{'generate':'generate','rewrite':'rewrite'})
    g.add_edge('rewrite','search');g.add_edge('search','generate');g.add_edge('generate',END)
    return g.compile()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--no-gpu-monitor',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--limit',type=int,default=5);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--corpus',type=Path);p.add_argument('--config',type=Path,default=HERE/'config.openai.json')
    a=p.parse_args()
    if not 1<=a.limit<=200 or a.repeats<1:p.error('limit must be 1..200 and repeats positive')
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    if cfg.get('provider')=='openai':cfg['model']=os.getenv('EXPERIMENT_MODEL') or cfg['model']
    if cfg.get('provider')=='openai' and not os.getenv('OPENAI_API_KEY'):p.error('Set OPENAI_API_KEY in your terminal first. Do not paste it into chat.')
    data=ROOT/'data/grading_optimization';corpus=a.corpus or data/f'corpus_{a.limit}'
    manifest=json.loads((corpus/'manifest.json').read_text(encoding='utf-8'))
    rows=[json.loads(l) for l in (data/'selected_203.jsonl').read_text(encoding='utf-8').splitlines()]
    warmup=rows[:3];measured=rows[3:3+a.limit]
    if not {r['id'] for r in measured} <= set(manifest['evaluation_ids']):p.error('Corpus manifest does not contain selected query IDs')
    os.environ['HF_HOME']=str(data/'hf_cache');os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
    out=a.resume or ROOT/'results/grading_optimization'/time.strftime('%Y%m%d_%H%M%S')
    if a.resume:
        previous=json.loads((out/'config.json').read_text(encoding='utf-8'))
        for key in ['provider','model','max_tokens','num_ctx','temperature','limit','repeats']:
            expected={'limit':a.limit,'repeats':a.repeats}.get(key,cfg.get(key))
            if previous.get(key)!=expected:raise ValueError('Resume config mismatch: '+key)
        if previous['corpus']['index_sha256']!=manifest['index_sha256']:raise ValueError('Resume corpus mismatch')
    out.mkdir(parents=True,exist_ok=bool(a.resume))
    done=set()
    if (out/'results.jsonl').exists():
        for line in (out/'results.jsonl').read_text(encoding='utf-8').splitlines():
            r=json.loads(line);done.add((r['query_id'],r['variant'],r['repeat']))
    (out/'config.json').write_text(json.dumps({**cfg,'repeats':a.repeats,'limit':a.limit,'corpus':manifest,'gpu_work':'LOCAL_SERVER_DURATION_AND_DEVICE_TELEMETRY_RECORDED_KERNEL_TIME_NOT_MEASURED' if cfg.get('provider')=='ollama_native' else 'REMOTE_LLM_SERVICE_NOT_MEASURED','local_gpu_monitor':not a.no_gpu_monitor,'retrieval_device':'cpu','correction_backend':'local_faiss_rewrite','cache_control':'no server cache control; variant order rotates across repetitions'},indent=2),encoding='utf-8')
    (out/'STATUS.txt').write_text('RUNNING. See provider config for local/remote model; model server time is not pure GPU kernel time. Corpus differs from teammate baseline. Correction uses local FAISS, not external web search.\n',encoding='utf-8')
    backend=OllamaBackend(cfg) if cfg.get('provider')=='ollama_native' else Backend(cfg,False)
    retriever=FaissRetriever(corpus/'index.faiss',corpus/'chunks.jsonl',manifest['embedding'],query_prefix='Represent this sentence for searching relevant passages: ',device='cpu')
    graph=make_graph(backend,retriever,cfg['upper'],cfg['lower'],cfg.get('team_prompts',False))
    gpu=None if a.no_gpu_monitor else GpuMonitor(out).start()
    def execute(row,variant):
        timer=StageTimer()
        with stopwatch() as sw:
            result=graph.invoke({'row':row,'variant':variant,'scores':[],'index':0,'timer':timer,'calls':[]},{'recursion_limit':100})
        calls=result['calls']
        return {'query_id':row['id'],'s_pop':row.get('s_pop'),'retrieval_chars':[len(d.text) for d in result['docs']],'extra':{'retrieval_scores':[d.score for d in result['docs']]},'question':row['question'],'gold_answers':row['answers'],'variant':variant,'mode':'local_ollama_local_faiss' if cfg.get('provider')=='ollama_native' else 'actual_remote_api_local_faiss','correction':result['action']!='correct','corrective_action':result['action'],'n_grade':result['index'],'scores':result['scores'],'answer':result['answer'],'context_ids':[d.id for d in result['contexts']],'stage_latencies':{k:timer.totals.get(k,0) for k in ['retrieval','grading','rewrite','web_search','generation']},'local_wall_seconds':sw['elapsed'],'input_tokens':sum(c['usage'].get('prompt_tokens',0) for c in calls),'output_tokens':sum(c['usage'].get('completion_tokens',0) for c in calls),'model_server_seconds':sum(c.get('server_total',0) for c in calls) if cfg.get('provider')=='ollama_native' else None,'gpu_service_seconds':None,'gpu_service_status':'SERVER_TIME_RECORDED_GPU_CPU_SPLIT_NOT_PROFILED' if cfg.get('provider')=='ollama_native' else 'NOT_MEASURED','calls':calls,**score_all(result['answer'],row['answers'])}
    variants=['full','lazy_hit','lazy_rest'];failures=0
    for repeat in range(a.repeats):
        order=variants[repeat%3:]+variants[:repeat%3]
        for variant in order:
            print(f'Warmup: repeat={repeat} variant={variant}',flush=True)
            for row in warmup:
                if gpu:gpu.mark('warmup',variant,repeat,row['id'])
                execute(row,variant)
            if cfg.get('provider')=='ollama_native':
                import urllib.request
                with urllib.request.urlopen(cfg['base_url']+'/api/ps',timeout=20) as response:placement=json.load(response)
                (out/'model_placement.json').write_text(json.dumps(placement,indent=2),encoding='utf-8')
                loaded=[m for m in placement.get('models',[]) if m.get('name')==cfg['model'] or m.get('model')==cfg['model']]
                if not loaded or loaded[0].get('size_vram',0)<=0:
                    raise RuntimeError('Local model is not confirmed in GPU memory; refusing to label a CPU run as a GPU experiment.')
                print('Model placement: size_vram='+str(loaded[0]['size_vram'])+' bytes; total size='+str(loaded[0].get('size')),flush=True)
            for i,row in enumerate(measured):
                if (row['id'],variant,repeat) in done:continue
                try:
                    if gpu:gpu.mark('measured',variant,repeat,row['id'])
                    result={**execute(row,variant),'repeat':repeat}
                    with (out/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                    print(f'{variant} {i+1}/{a.limit}: {result["local_wall_seconds"]:.2f}s; grades={result["n_grade"]}; acc={result["acc"]}',flush=True)
                except Exception as e:
                    failures+=1
                    with (out/'errors.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'query_id':row['id'],'variant':variant,'repeat':repeat,'error_type':type(e).__name__,'http_status':getattr(e,'code',None)})+'\n')
                    print('Failed:',row['id'],type(e).__name__,flush=True)
    if gpu:gpu.stop()
    (out/'STATUS.txt').write_text(f'ACTUAL API RUN FINISHED; failures={failures}. Local model server time is recorded only with ollama_native; pure GPU kernel time is not measured. Local FAISS correction and selected-query corpus; not comparable to previous GPU absolute times.\n',encoding='utf-8')
    print('RUN_DIR:',out,flush=True)
    import subprocess
    if cfg.get('provider')=='ollama_native' and (out/'results.jsonl').exists():
        records=[json.loads(l) for l in (out/'results.jsonl').read_text(encoding='utf-8').splitlines()]
        full=[r for r in records if r['variant']=='full' and r['repeat']==0]
        for r in full:r['pipeline']='crag'
        (out/'profile.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in full),encoding='utf-8')
    if (out/'results.jsonl').exists():subprocess.run([sys.executable,str(HERE/'analyze.py'),str(out)],check=True)
if __name__=='__main__':main()
