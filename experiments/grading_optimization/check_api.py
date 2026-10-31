"""Small API check; never prints the API key. One short generation if key exists."""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

def redact(value,key):
    text=str(value)
    if key:text=text.replace(key,'[REDACTED]')
    text=re.sub(r'sk-[A-Za-z0-9_\-]+','[REDACTED]',text)
    return text[:1500]

def main():
    key=os.getenv('OPENAI_API_KEY','')
    if not key:
        print('API_KEY_MISSING: set the key in this terminal first.');return
    if len(key)<20 or any(c.isspace() for c in key) or '...' in key or '\u2026' in key:
        print('API_KEY_INPUT_INCOMPLETE: input appears truncated or contains whitespace. Re-copy the full generated secret. Key was not printed and no request was sent.');return
    cfg=json.loads((Path(__file__).parent/'config.openai.json').read_text(encoding='utf-8-sig'))
    model=os.getenv('EXPERIMENT_MODEL') or cfg['model']
    payload={'model':model,'messages':[{'role':'user','content':'Return only JSON: {"score": 0.8}'}],'max_completion_tokens':64,'temperature':cfg.get('temperature',0),'response_format':{'type':'json_object'}}
    req=urllib.request.Request(cfg['base_url'].rstrip('/')+'/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
    print('Checking model:',model,flush=True)
    try:
        with urllib.request.urlopen(req,timeout=60) as r:
            result=json.load(r)
        print('API_CHECK_OK')
        print('Output:',result['choices'][0]['message']['content'])
        print('Usage:',json.dumps(result.get('usage',{})))
    except urllib.error.HTTPError as e:
        try:detail=json.loads(e.read().decode('utf-8')).get('error',{})
        except Exception:detail={}
        print('HTTP_STATUS:',e.code)
        for name in ['type','code','param','message']:
            print(name.upper()+':',redact(detail.get(name,'not supplied'),key))
    except Exception as e:
        print('CONNECTION_ERROR:',type(e).__name__)
if __name__=='__main__':main()
