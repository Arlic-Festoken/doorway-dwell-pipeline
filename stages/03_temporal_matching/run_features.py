from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
ledger=ROOT/'logs/process_ledger.jsonl'
record={'stage':'doorway_temporal_dense_sequences','pid':os.getpid(),'cwd':str(HERE),'output':str(HERE/'features'),'started_at':datetime.now().isoformat(),'gpu_ids':[0,1],'stop_condition':'prepare and encode complete or timeout 7200 seconds','historical_records_used':False}
with ledger.open('a') as f:f.write(json.dumps(record)+'\n')
subprocess.run([sys.executable,str(HERE/'prepare_sequences.py')],check=True,timeout=1200)
children=[]
for shard in (0,1):
    log=(HERE/'logs'/f'extract_{shard}.log').open('w');p=subprocess.Popen(['timeout','--kill-after=10s','6500s',sys.executable,str(HERE/'extract_frames.py'),'--shard',str(shard)],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(shard),'PYTHONPATH':str(ROOT/'rebuild_20261107/vendor')},stdout=log,stderr=subprocess.STDOUT)
    children.append(p)
    with ledger.open('a') as f:f.write(json.dumps({**record,'stage':'ordered_frame_encoding','pid':p.pid,'gpu_ids':[shard],'log':str(log.name)})+'\n')
for p in children:
    if p.wait(timeout=6500)!=0:raise RuntimeError(f'Frame encoder failed: {p.pid}')
record.update({'stage':'doorway_temporal_features_complete','completed_at':datetime.now().isoformat()})
with ledger.open('a') as f:f.write(json.dumps(record)+'\n')
(HERE/'features_complete.json').write_text(json.dumps(record,indent=2))
