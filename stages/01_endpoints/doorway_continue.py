"""Bounded query-recovery stages, with remote process accounting."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

base=Path(__file__).resolve().parent;root=base.parent;new=base/'query_round';started=time.time()
while not (base/'results/targeted_requests.json').exists():
    if time.time()-started>900:raise TimeoutError('Query preparation did not finish')
    time.sleep(3)
new.mkdir(exist_ok=True)
for name in ('logs','results','sequences','features','crops'):(new/name).mkdir(exist_ok=True)
for name in ('features','states','retrieval','assign'):
    text=(base/f'doorway_{name}.py').read_text()
    if name=='assign':text=text.replace("'historical_gate_baseline':8","'historical_gate_baseline':9").replace('recovered>=8','recovered>=9').replace("'baseline_recovered':8","'baseline_recovered':9")
    (new/f'doorway_{name}.py').write_text(text)

def run(stage,script,cwd,gpu,seconds):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONPATH':str(root/'rebuild_20261107/vendor')}
    with (new/'logs'/f'{stage}.log').open('w') as log:
        p=subprocess.Popen(['timeout',str(seconds),str(root/'.venv-kpr/bin/python'),'-u',str(script)],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        record={'stage':stage,'pid':p.pid,'started_at':datetime.now().isoformat(),'cwd':str(cwd),'gpu':gpu,'cpu_threads':2,'log':log.name,'output':str(new),'stop_condition':f'completion or {seconds} seconds'}
        with (root/'logs/process_ledger.jsonl').open('a') as ledger:ledger.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True);rc=p.wait()
    with (root/'logs/process_ledger.jsonl').open('a') as ledger:ledger.write(json.dumps({'stage':stage,'pid':p.pid,'finished_at':datetime.now().isoformat(),'exit_code':rc})+'\n')
    print(json.dumps({'stage':stage,'exit_code':rc,'log_tail':(new/'logs'/f'{stage}.log').read_text()[-2000:]}),flush=True)
    if rc:raise SystemExit(rc)

run('query_redetection',base/'doorway_redetect.py',base,1,2400)
run('query_features',new/'doorway_features.py',new,1,1500)
(new/'sequences').rename(new/'query_sequences');(new/'features').rename(new/'query_features');(new/'sequences').mkdir();(new/'features').mkdir()
for clip in range(106):
    name=f'clip_{clip:03d}';original=json.loads((base/'sequences'/f'{name}.json').read_text());extra=json.loads((new/'query_sequences'/f'{name}.json').read_text())
    original['measurements'].update(extra['measurements']);original['crops'].extend(extra['crops']);(new/'sequences'/f'{name}.json').write_text(json.dumps(original))
    a=np.load(base/'features'/f'{name}.npz');b=np.load(new/'query_features'/f'{name}.npz');np.savez_compressed(new/'features'/f'{name}.npz',**{k:np.concatenate([a[k],b[k]]) for k in a.files})
for path in (base/'crops').glob('*.jpg'):os.link(path,new/'crops'/path.name)
run('query_states',new/'doorway_states.py',new,None,300)
run('query_retrieval',new/'doorway_retrieval.py',new,1,1800)
run('query_assignment',new/'doorway_assign.py',new,None,300)
events=json.loads((new/'events_states.json').read_text());queries=[e for e in events if e['source_category']=='query_recovered_candidate'];assignment=json.loads((new/'results/assignment_summary.json').read_text())
summary={'query_windows':len(queries),'supported_new_entries':sum(e['state_supported'] and e['direction']=='entry' for e in queries),'supported_new_exits':sum(e['state_supported'] and e['direction']=='exit' for e in queries),
    'query_events_without_supported_crossing':sum(not e['state_supported'] for e in queries),'assignment':assignment,'elapsed_seconds':round(time.time()-started,1)}
(new/'results/query_recovery_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
