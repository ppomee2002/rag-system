import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
import re
from pathlib import Path
from typing import TypedDict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / '_deps'))
sys.path.insert(0, str(ROOT / 'src'))
from langgraph.graph import StateGraph, START, END
from rag.eval.metrics import score_all
from rag.utils.timing import StageTimer

class State(TypedDict, total=False):
    row: dict
    variant: str
    index: int
    scores: list
    selected: list
    correction: bool
    answer: str
    timer: object
    calls: list

class Backend:
    def __init__(self, config, demo):
        self.config, self.demo = config, demo
    def request(self, kind, prompt, fallback, calls):
        if self.demo:
            return fallback
        payload = {'model': self.config['model'], 'messages': [{'role':'user','content':prompt}], 'max_completion_tokens':min(self.config['max_tokens'],64) if kind in ('grade','rewrite') else self.config['max_tokens']}
        if self.config.get('provider') == 'vllm':
            payload['priority'] = 0
        if self.config.get('temperature') is not None:
            payload['temperature'] = self.config['temperature']
        if kind == 'grade':
            payload['response_format'] = {'type':'json_object'}
        headers = {'Content-Type':'application/json'}
        key_name = 'OPENAI_API_KEY' if self.config.get('provider') == 'openai' else 'VLLM_API_KEY'
        key = os.getenv(key_name)
        if self.config.get('provider') == 'openai' and not key:
            raise RuntimeError('OPENAI_API_KEY is not set. Never paste the key into chat or source code.')
        if key:
            headers['Authorization'] = 'Bearer ' + key
        req = urllib.request.Request(self.config['base_url'].rstrip('/')+'/chat/completions', json.dumps(payload).encode(), headers)
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.load(response)
                request_id = response.headers.get('x-request-id')
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read().decode('utf-8')).get('error', {})
            except Exception:
                detail = {}
            message = str(detail.get('message', 'No error detail supplied'))
            if key:
                message = message.replace(key, '[REDACTED]')
            message = re.sub(r'sk-[A-Za-z0-9_\-]+', '[REDACTED]', message)
            raise RuntimeError(f"API HTTP {error.code}; code={detail.get('code')}; param={detail.get('param')}; {message[:1500]}") from None
        calls.append({'kind':kind, 'wall_seconds':time.perf_counter()-start, 'usage':result.get('usage',{}), 'model':result.get('model'), 'request_id':request_id})
        return result['choices'][0]['message']['content'].strip()

def build_graph(backend, upper, lower):
    def grade(s):
        i, row = s['index'], s['row']
        with s['timer'].stage('grading'):
            prompt = 'Rate relevance to the question from 0 to 1. Return ONLY a JSON object with numeric key score.\nQuestion: '+row['question']+'\nDocument: '+row['docs'][i]['text']
            fallback = json.dumps({'score': row['docs'][i].get('demo_score',0)})
            score = float(json.loads(backend.request('grade', prompt, fallback, s['calls']))['score'])
            if not 0 <= score <= 1:
                raise ValueError('Invalid grading score')
        return {'index':i+1, 'scores':s['scores']+[score]}
    def next_grade(s):
        if s['variant'] != 'full' and s['scores'][-1] >= upper:
            return 'select'
        return 'select' if s['index'] == len(s['row']['docs']) else 'grade'
    def select(s):
        scores, docs = s['scores'], s['row']['docs']
        hit = next((i for i,v in enumerate(scores) if v >= upper),None)
        if hit is not None:
            chosen = ([docs[hit]] if s['variant']=='lazy_hit' else docs[hit:]) if s['variant']!='full' else [d for d,v in zip(docs,scores) if v>=upper]
            return {'selected':chosen, 'correction':False}
        # Same no-hit fallback for all variants; correction documents must be supplied.
        correction_docs = s['row'].get('correction_docs')
        if not correction_docs:
            raise ValueError('No passing document: correction_docs required; refusing to silently skip correction')
        retained = [d for d,v in zip(docs,scores) if v>=lower]
        return {'selected':retained + correction_docs, 'correction':True}
    def generate(s):
        with s['timer'].stage('generation'):
            context = '\n\n'.join(d['text'] for d in s['selected'])
            prompt = 'Answer the question using the context. Return only a short answer.\nContext:\n'+context+'\nQuestion: '+s['row']['question']
            fallback = next((d.get('demo_answer') for d in s['selected'] if d.get('demo_answer')), 'unknown')
            answer = backend.request('generate',prompt,fallback,s['calls'])
        return {'answer':answer}
    g = StateGraph(State)
    g.add_node('grade',grade)
    g.add_node('select',select)
    g.add_node('generate',generate)
    g.add_edge(START,'grade')
    g.add_conditional_edges('grade',next_grade,{'grade':'grade','select':'select'})
    g.add_edge('select','generate')
    g.add_edge('generate',END)
    return g.compile()

def demo_rows():
    rows=[]
    for i,hit in enumerate([0,1,2,3,4,None]):
        docs=[{'id':f'd{j}', 'text':f'Example document {j}', 'demo_score':0.8 if j==hit else 0.1, 'demo_answer':f'answer{i}' if j==hit else None} for j in range(5)]
        rows.append({'id':f'demo{i}','question':f'Synthetic question {i}', 'answers':[f'answer{i}'],'docs':docs,'correction_docs':[{'id':'correction','text':'Synthetic correction evidence','demo_answer':f'answer{i}'}]})
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--demo',action='store_true')
    p.add_argument('--input',type=Path)
    p.add_argument('--warmup-input',type=Path)
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if cfg.get('provider')=='openai' and os.getenv('EXPERIMENT_MODEL'):
        cfg['model']=os.environ['EXPERIMENT_MODEL']
    if not args.demo and args.input is None:
        p.error('Real measurement requires --input and a running vLLM server. Use --demo only for functional validation.')
    rows=demo_rows() if args.demo else [json.loads(l) for l in args.input.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    if not rows or any(not r.get('docs') or not r.get('answers') for r in rows):
        p.error('Input must contain nonempty docs and answers')
    if args.limit is not None:
        rows=rows[:args.limit]
    if args.repeats < 1 or not rows:
        p.error('repeats and query count must be positive')
    warmup_rows=[]
    if not args.demo:
        if args.warmup_input is None:
            p.error('Provide --warmup-input with prepared evidence for 3 separate warmup queries')
        warmup_rows=[json.loads(l) for l in args.warmup_input.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        if len(warmup_rows)!=3 or {r['id'] for r in warmup_rows} & {r['id'] for r in rows}:
            p.error('Use exactly 3 warmup queries disjoint from evaluation queries')
    backend=Backend(cfg,args.demo)
    graph=build_graph(backend,cfg['upper'],cfg['lower'])
    results=[]
    for variant in ['full','lazy_hit','lazy_rest']:
        if not args.demo:
            for warm in warmup_rows:
                graph.invoke({'row':warm,'variant':variant,'index':0,'scores':[],'timer':StageTimer(),'calls':[]}, {'recursion_limit':100})
        for repeat,row in [(rep,row) for rep in range(args.repeats) for row in rows]:
            timer=StageTimer()
            start=time.perf_counter()
            result=graph.invoke({'row':row,'variant':variant,'index':0,'scores':[],'timer':timer,'calls':[]}, {'recursion_limit':100})
            results.append({'query_id':row['id'],'repeat':repeat,'variant':variant,'mode':'synthetic_functional_validation' if args.demo else cfg.get('provider','vllm')+'_pre_retrieved_evaluation','n_grade':result['index'],'scores':result['scores'],'correction':result['correction'],'answer':result['answer'],'context_ids':[d['id'] for d in result['selected']],'stage_latencies':dict(timer.totals),'local_wall_seconds':time.perf_counter()-start,'gpu_service_seconds':None,'gpu_service_status':'unavailable_remote_api_or_uninstrumented_server','input_tokens':sum(c['usage'].get('prompt_tokens',0) for c in result['calls']),'output_tokens':sum(c['usage'].get('completion_tokens',0) for c in result['calls']),'calls':result['calls'],**score_all(result['answer'],row['answers'])})
    out=ROOT/'results/grading_optimization'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True)
    (out/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    summary=[]
    for v in ['full','lazy_hit','lazy_rest']:
        rs=[r for r in results if r['variant']==v]
        summary.append({'variant':v,'n_queries':len(rs),'mean_grade_calls':sum(r['n_grade'] for r in rs)/len(rs),'accuracy':sum(r['acc'] for r in rs)/len(rs),'f1':sum(r['f1'] for r in rs)/len(rs)})
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (out/'STATUS.txt').write_text('SYNTHETIC FUNCTIONAL VALIDATION ONLY. No PopQA, real LLM, vLLM GPU latency or quality conclusion.\n' if args.demo else 'Pre-retrieved input evaluation; retrieval and correction search latency excluded. Cache effects and repeat runs must be checked before reporting latency.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2));print('Saved:',out)

if __name__=='__main__':main()
