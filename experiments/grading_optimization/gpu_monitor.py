"""System-wide local NVIDIA telemetry, NOT remote LLM GPU service time."""
import atexit
import csv
import json
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FIELDS=['utc_time','query_start_monotonic','query_end_monotonic','phase','variant','repeat','query_id','gpu_index','gpu_name','utilization_percent','memory_used_mib','memory_total_mib','power_watts']

def number(value):
    try:return float(value.strip())
    except ValueError:return None

class GpuMonitor:
    def __init__(self,out,interval=1.0):
        self.out=Path(out);self.interval=interval
        self.context=('between','',-1,'');self.lock=threading.Lock()
        self.event=threading.Event();self.rows=[];self.errors=[];self.thread=None;self.stopped=False
    def mark(self,phase,variant='',repeat=-1,query_id=''):
        with self.lock:self.context=(phase,variant,repeat,str(query_id))
    def start(self):
        existing=self.out/'local_gpu_samples.csv'
        if existing.exists():
            with existing.open(encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    for key in ['utilization_percent','memory_used_mib','memory_total_mib','power_watts']:
                        row[key]=number(row[key]) if row[key] else None
                    self.rows.append(row)
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
        atexit.register(self.stop)
        return self
    def _loop(self):
        path=self.out/'local_gpu_samples.csv'
        existing=path.exists() and path.stat().st_size>0
        with path.open('a',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            if not existing:writer.writeheader()
            f.flush()
            while not self.event.is_set():
                start=time.monotonic()
                with self.lock:before=self.context
                try:
                    proc=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    if proc.returncode:raise RuntimeError('nvidia-smi exit '+str(proc.returncode))
                    end=time.monotonic()
                    with self.lock:after=self.context
                    context=before if before==after else ('transition','',-1,'')
                    for values in csv.reader(proc.stdout.splitlines()):
                        if len(values)!=6:continue
                        idx,name,util,used,total,power=values
                        row=dict(zip(FIELDS,[datetime.now(timezone.utc).isoformat(),start,end,*context,idx.strip(),name.strip(),number(util),number(used),number(total),number(power)]))
                        self.rows.append(row);writer.writerow(row);f.flush()
                except Exception as exc:
                    self.errors.append({'error_type':type(exc).__name__,'time':time.monotonic()})
                self.event.wait(max(0,self.interval-(time.monotonic()-start)))
    def stop(self):
        if self.stopped:return
        self.event.set()
        if self.thread:self.thread.join(timeout=6)
        self.stopped=True
        groups=defaultdict(list)
        for row in self.rows:groups[(row['gpu_index'],row['phase'],row['variant'])].append(row)
        summary=[]
        for (idx,phase,variant),rows in sorted(groups.items()):
            item={'gpu_index':idx,'phase':phase,'variant':variant,'samples':len(rows)}
            for key in ['utilization_percent','memory_used_mib','power_watts']:
                values=[r[key] for r in rows if r[key] is not None]
                item['mean_'+key]=statistics.mean(values) if values else None
                item['max_'+key]=max(values) if values else None
            summary.append(item)
        if summary:
            with (self.out/'local_gpu_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
        (self.out/'local_gpu_status.json').write_text(json.dumps({'scope':'LOCAL_DEVICE_WIDE_NOT_PROCESS_ATTRIBUTED','remote_llm_gpu_service':'NOT_MEASURED','interval_target_seconds':self.interval,'samples':len(self.rows),'errors':self.errors,'limitations':'Includes other applications and desktop activity. Device sampling windows overlap phase boundaries. Approximate phase labels; transitions excluded from measured phase summaries. Power is a device reading, not experiment-attributed energy. Missing values remain missing. Monitoring overhead may affect CPU timing.'},indent=2),encoding='utf-8')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=4);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    monitor=GpuMonitor(a.out).start();monitor.mark('diagnostic_idle')
    time.sleep(a.seconds);monitor.stop();print('Local GPU telemetry samples:',len(monitor.rows));print('Saved:',a.out)
