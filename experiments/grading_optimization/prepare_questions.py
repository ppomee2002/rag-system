"""Download official PopQA TSV, filter s_pop<100, shuffle with seed 42.
203 selected queries comprise 3 warmup + 200 measured, matching existing runner.
"""
import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE='https://huggingface.co'
meta=json.load(urllib.request.urlopen(BASE+'/api/datasets/akariasai/PopQA',timeout=30))
revision=meta['sha']
url=BASE+'/datasets/akariasai/PopQA/resolve/'+revision+'/test.tsv'
raw=urllib.request.urlopen(url,timeout=60).read()
folder=ROOT/'data/grading_optimization';folder.mkdir(parents=True,exist_ok=True)
(folder/'popqa_test.tsv').write_bytes(raw)
rows=[]
for r in csv.DictReader(io.StringIO(raw.decode('utf-8')),delimiter='\t'):
    if int(r['s_pop']) < 100:
        rows.append({'id':str(r['id']),'question':r['question'],'answers':json.loads(r['possible_answers']),'s_pop':int(r['s_pop']),'prop':r['prop'],'s_wiki_title':r['s_wiki_title'],'o_wiki_title':r['o_wiki_title']})
def save(name,values):
    (folder/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values),encoding='utf-8')
save('popqa_longtail.jsonl',rows)
random.Random(42).shuffle(rows)
selected=rows[:203]
save('selected_203.jsonl',selected)
save('warmup_3.jsonl',selected[:3])
save('evaluation_200.jsonl',selected[3:])
manifest={'source':url,'revision':revision,'sha256':hashlib.sha256(raw).hexdigest(),'filter':'s_pop < 100','seed':42,'longtail_count':len(rows),'selected_count':len(selected),'warmup_count':3,'evaluation_count':len(selected[3:]),'query_ids':[r['id'] for r in selected],'note':'Same filter/shuffle/limit code as existing experiment; identity with historical files must be checked by query IDs. Corpus and FAISS index not yet prepared.'}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='query_ids'},indent=2))
print('Saved:',folder)
