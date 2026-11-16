from datetime import datetime
import json
import os
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;cv2.setNumThreads(2)
ledger={'stage':'recall_anchor_contexts','pid':os.getpid(),'cwd':str(HERE),'output':str(HERE/'contexts'),'started_at':datetime.now().isoformat(),'stop_condition':'101 original contexts complete or 120 seconds','gpu':None}
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(ledger)+'\n')
for r in json.loads((HERE/'results/anchor_sources.json').read_text()):
    cap=cv2.VideoCapture(str(ROOT/'raw/DCIMA'/f'VID{r["clip"]:05d}.AVI'));cap.set(cv2.CAP_PROP_POS_FRAMES,r['native_frame_index']);ok,frame=cap.read();cap.release()
    if not ok:raise ValueError(f'Could not read anchor for event {r["event_id"]}')
    x1,y1,x2,y2=np.rint(r['box']).astype(int);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,190,255),3);frame=cv2.resize(frame,(960,540));t=r['time'];stamp=f'{int(t)//3600:02d}:{int(t)%3600//60:02d}:{t%60:05.2f}'
    cv2.putText(frame,f'Event {r["event_id"]}  {stamp}  ANCHOR ONLY - identity continuity unverified',(16,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,210,255),2)
    if not cv2.imwrite(str(HERE/'contexts'/f'anchor_{r["event_id"]:04d}.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,92]):raise OSError('Cannot save anchor context')
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps({**ledger,'stage':'recall_anchor_contexts_complete','completed_at':datetime.now().isoformat()})+'\n')
