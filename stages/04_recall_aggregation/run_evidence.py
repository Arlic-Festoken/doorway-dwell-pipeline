from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;children=[]
for script,extra,gpu,limit in [('extract_frames.py',['--shard','0'],0,600),('extract_frames.py',['--shard','1'],1,600),('render_videos.py',[],None,1800)]:
    label='anchor_'+str(gpu) if gpu is not None else 'videos';log=(HERE/'logs'/f'{label}.log').open('w');env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONPATH':str(ROOT/'rebuild_20261107/vendor')}
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    child=subprocess.Popen(['timeout','--kill-after=10s',str(limit),sys.executable,str(HERE/script),*extra],env=env,stdout=log,stderr=subprocess.STDOUT);children.append((label,child,log))
    with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps({'stage':'recall_'+label,'pid':child.pid,'cwd':str(HERE),'started_at':datetime.now().isoformat(),'gpu':gpu,'log':str(log.name),'output':str(HERE),'stop_condition':f'completion or {limit} seconds'})+'\n')
codes=[]
for label,child,log in children:
    code=child.wait();codes.append(code);log.close()
    with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps({'stage':'recall_'+label+'_finished','pid':child.pid,'cwd':str(HERE),'exit_code':code,'finished_at':datetime.now().isoformat()})+'\n')
if any(codes):raise RuntimeError(f'Evidence jobs failed: {codes}')
(HERE/'evidence_complete.json').write_text(json.dumps({'status':'complete','completed_at':datetime.now().isoformat()}))
