"""Prepare a clearly labelled selected-query Wikipedia corpus and BGE/FAISS index."""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'_extras'))
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/crag_latency'))
from build_index import chunk_by_tokens

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=5);p.add_argument('--full-text',action='store_true');a=p.parse_args()
    data=ROOT/'data/grading_optimization'
    selected=[json.loads(l) for l in (data/'selected_203.jsonl').read_text(encoding='utf-8').splitlines()]
    queries=selected[:3]+selected[3:3+a.limit]
    out=data/f'corpus_{a.limit}';out.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(data/'hf_cache')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
    titles=sorted({r[k] for r in queries for k in ['s_wiki_title','o_wiki_title'] if r[k]})
    pages={};errors=[]
    batch_size=1 if a.full_text else 10
    for start in range(0,len(titles),batch_size):
        batch=titles[start:start+batch_size]
        cache=out/(hashlib.sha256((('full-v1|' if a.full_text else 'intro-v1|')+'|'.join(batch)).encode()).hexdigest()+'.json')
        if cache.exists():result=json.loads(cache.read_text(encoding='utf-8'))
        else:
            params={'action':'query','prop':'extracts','explaintext':1,'exlimit':'max','format':'json','formatversion':2,'redirects':1,'titles':'|'.join(batch)}
            if not a.full_text:params['exintro']=1
            url='https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode(params)
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RAGResearch/0.1 (academic comparison)'}),timeout=60) as r:result=json.load(r)
                    if 'error' in result:raise RuntimeError(result['error'])
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(15*(attempt+1))
            cache.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');time.sleep(1.2)
        for page in result.get('query',{}).get('pages',[]):
            if page.get('extract'):
                pages[page['title']]={'title':page['title'],'text':page['extract']}
            else:errors.append(page.get('title'))
        print(f'Wikipedia {min(start+batch_size,len(titles))}/{len(titles)}; pages={len(pages)}',flush=True)
    if len(pages)<5:raise RuntimeError('Too few Wikipedia pages; cannot construct five-document retrieval')
    (out/'pages.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pages.values()),encoding='utf-8')
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    print('Loading BAAI/bge-base-en-v1.5 on CPU (first run downloads model)',flush=True)
    encoder=SentenceTransformer('BAAI/bge-base-en-v1.5',device='cpu')
    size=min(512,encoder.max_seq_length-32)
    docs=[]
    for page in pages.values():
        for i,text in enumerate(chunk_by_tokens(page['text'],encoder.tokenizer,size,64)):
            docs.append({'id':page['title']+'#'+str(i),'text':page['title']+'\n'+text,'title':page['title']})
    print(f'Encoding {len(docs)} chunks',flush=True)
    checkpoint=out/'embedding_checkpoints';checkpoint.mkdir(exist_ok=True)
    corpus_key=hashlib.sha256(''.join(d['id']+d['text'] for d in docs).encode()).hexdigest()[:16]
    shards=[];started=time.perf_counter();fresh_chunks=0
    for offset in range(0,len(docs),64):
        shard_path=checkpoint/(corpus_key+'_'+str(offset)+'.npy')
        part=docs[offset:offset+64]
        if shard_path.exists():vectors_part=np.load(shard_path)
        else:
            vectors_part=encoder.encode([d['text'] for d in part],batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype('float32')
            temp_path=shard_path.with_suffix('.tmp.npy');np.save(temp_path,vectors_part);temp_path.replace(shard_path)
            fresh_chunks+=len(part)
        shards.append(vectors_part)
        done=min(offset+64,len(docs));elapsed=time.perf_counter()-started
        remaining=(len(docs)-done)*elapsed/fresh_chunks if fresh_chunks else 0
        print(f'Embedding {done}/{len(docs)} chunks; estimated remaining {remaining/60:.1f} min',flush=True)
    vectors=np.concatenate(shards)
    index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors)
    (out/'index.faiss').write_bytes(faiss.serialize_index(index).tobytes())
    (out/'chunks.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs),encoding='utf-8')
    manifest={'mode':f'selected_query_fulltext_{a.limit}' if a.full_text else f'selected_query_introductions_{a.limit}','full_text':a.full_text,'embedding':'BAAI/bge-base-en-v1.5','device':'cpu','chunk_size':size,'overlap':64,'top_k':5,'n_pages':len(pages),'n_chunks':len(docs),'warmup_ids':[r['id'] for r in queries[:3]],'evaluation_ids':[r['id'] for r in queries[3:]],'missing_titles':errors,'corpus_source':('Wikipedia full articles' if a.full_text else 'Wikipedia introductions')+'; subject/object pages of selected queries. Smaller corpus than original all-longtail corpus; not directly comparable with teammate results.','correction':'query rewrite followed by local FAISS search; not real web search and no replayed Tavily delay','index_sha256':hashlib.sha256((out/'index.faiss').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('READY:',out,flush=True)
if __name__=='__main__':main()
