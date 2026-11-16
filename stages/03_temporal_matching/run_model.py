from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;started=time.time()
while not (HERE/'features_complete.json').exists():
    if time.time()-started>600:raise TimeoutError('Feature preparation did not complete')
    time.sleep(5)
record={'stage':'temporal_model_training','pid':os.getpid(),'cwd':str(HERE),'gpu_ids':[0],'output':str(HERE/'temporal_encoder.pt'),'log':str(HERE/'logs/train.log'),'started_at':datetime.now().isoformat(),'stop_condition':'training complete or 1800 seconds'}
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
subprocess.run(['timeout','--kill-after=10s','1800s',sys.executable,str(HERE/'train_temporal.py')],env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},check=True)
record.update(stage='temporal_model_complete',completed_at=datetime.now().isoformat())
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
(HERE/'model_complete.json').write_text(json.dumps(record,indent=2))
