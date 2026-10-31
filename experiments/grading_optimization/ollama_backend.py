"""Native local Ollama backend with server-side duration and token tracing."""
import json
import time
import urllib.request

class OllamaBackend:
    def __init__(self,config):self.config=config
    def request(self,kind,prompt,fallback,calls):
        payload={'model':self.config['model'],'prompt':prompt,'stream':False,'keep_alive':-1,'options':{'temperature':self.config.get('temperature',0),'num_predict':self.config.get('max_tokens',512),'num_ctx':self.config.get('num_ctx',6144),'seed':42}}
        if not self.config.get('team_prompts') and kind=='grade':payload['format']='json'
        req=urllib.request.Request(self.config['base_url'].rstrip('/')+'/api/generate',json.dumps(payload).encode(),{'Content-Type':'application/json'})
        start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=900) as response:result=json.load(response)
        calls.append({'t_start':start,'idx':sum(c['kind']=='grade' for c in calls) if kind=='grade' else 0,'kind':kind,'wall_seconds':time.perf_counter()-start,'server_total':result['total_duration']*1e-9,'load_seconds':result.get('load_duration',0)*1e-9,'prompt_eval_seconds':result.get('prompt_eval_duration',0)*1e-9,'decode_seconds':result.get('eval_duration',0)*1e-9,'prompt_chars':len(prompt),'usage':{'prompt_tokens':result.get('prompt_eval_count',0),'completion_tokens':result.get('eval_count',0)},'model':result['model']})
        return result['response'].strip()
