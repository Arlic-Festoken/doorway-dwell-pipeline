"""Readable doorway videos, preserving source PTS stamps and crop provenance."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REC=ROOT/'doorway_recovery_20261108';started=time.time();cv2.setNumThreads(2)
reports=json.loads((HERE/'results/assignment_summary.json').read_text());seq=json.loads((HERE/'sequences.json').read_text());frames={r['key']:r for r in json.loads((HERE/'frames.json').read_text())};events=json.loads((HERE/'events.json').read_text())
old_pack={r['event_id']:r for r in json.loads((REC/'results/evidence_manifest.json').read_text())['events']};needed=set()
for report in reports:
    records=json.loads((HERE/'variants'/report['variant']['directory']/'delivery_records.json').read_text())
    for r in records['confirmed']:needed.update(r['event_chain'])
record={'stage':'temporal_video_evidence','pid':os.getpid(),'cwd':str(HERE),'output':str(HERE/'videos'),'started_at':datetime.now().isoformat(),'stop_condition':'all selected endpoint clips rendered or 1800 seconds','gpu_ids':[]}
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
clocks=[json.loads((ROOT/'work/clocks'/f'VID{i:05d}.json').read_text())['offset_seconds'] for i in range(106)]
pts=[np.asarray([float(r['best_effort_timestamp_time']) for r in json.loads((ROOT/'work/native_crossings'/f'{i:03d}_pts.json').read_text())['frames']]) for i in range(106)]
boundary=np.array([[0,829],[450,807],[860,773],[1210,731],[1450,696],[1660,659],[1919,620]],np.int32)
manifest={};(HERE/'panels').mkdir(exist_ok=True)
for eid in sorted(needed):
    own=[s for s in seq if s['kind']=='event' and s['event_id']==eid];clips=[]
    for s in own:
        observations=[frames[k] for k in s['frames']];source_clips=sorted(set(r['clip'] for r in observations))
        for clip in source_clips:
            rows=[r for r in observations if r['clip']==clip];times=np.array([r['time'] for r in rows]);low,high=times[0],times[-1]
            if high-low<.4:continue
            relative=f'videos/{s["id"]}_c{clip:03d}.mp4';destination=HERE/relative
            metadata={'path':relative,'sequence_id':s['id'],'fragment':s['fragment'],'source_clip':clip,'interval':[float(low),float(high)],'display_sampling_hz':10,'native_pts_stamped':True}
            clips.append(metadata)
            if destination.exists():continue
            native=pts[clip]+clocks[clip];targets=np.arange(low,high+.001,.1);indices=[int(np.argmin(abs(native-t))) for t in targets]
            cap=cv2.VideoCapture(str(ROOT/'raw/DCIMA'/f'VID{clip:05d}.AVI'));cap.set(cv2.CAP_PROP_POS_FRAMES,indices[0]);position=indices[0]
            encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','1280x720','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE)
            for fi in indices:
                while position<fi:
                    if not cap.grab():raise ValueError(f'Decode failed {clip}:{fi}')
                    position+=1
                ok,source=cap.read();position+=1
                if not ok:raise ValueError(f'Decode failed {clip}:{fi}')
                at=float(native[fi]);j=int(np.argmin(abs(times-at)));observation=rows[j];box=np.array(observation['box']);height=box[3]-box[1]
                # Interpolation changes annotation position only, not pixels.
                hi=int(np.searchsorted(times,at));lo=hi-1
                if 0<=lo<hi<len(times) and times[hi]-times[lo]<=.25:
                    weight=(at-times[lo])/(times[hi]-times[lo]);box=(1-weight)*np.array(rows[lo]['box'])+weight*np.array(rows[hi]['box'])
                x1,y1,x2,y2=np.rint(box).astype(int);body=source[max(0,y1):min(1080,y2),max(0,x1):min(1920,x2)].copy()
                color=(75,215,100) if observation['visible'] else (0,170,255)
                cv2.polylines(source,[boundary],False,(0,230,250),2);cv2.rectangle(source,(x1,y1),(x2,y2),color,3)
                foot=(int((x1+x2)/2),int(y2-.04*height));cv2.circle(source,foot,6,(50,60,240),-1)
                canvas=np.full((720,1280,3),24,np.uint8);canvas[90:630,:960]=cv2.resize(source,(960,540))
                if body.size:
                    factor=min(300/body.shape[1],640/body.shape[0]);resized=cv2.resize(body,(max(1,int(body.shape[1]*factor)),max(1,int(body.shape[0]*factor))));bh,bw=resized.shape[:2];canvas[40+(640-bh)//2:40+(640-bh)//2+bh,970+(300-bw)//2:970+(300-bw)//2+bw]=resized
                seconds=int(at);stamp=f'{seconds//3600:02d}:{seconds%3600//60:02d}:{at%60:05.2f}'
                title=f'EVENT {eid:04d}  {events[eid]["direction"].upper()}  source PTS {stamp}'
                cv2.putText(canvas,title,(20,38),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),2)
                cv2.putText(canvas,f'Crossing offset {at-events[eid]["time"]:+.2f}s   fragment {s["fragment"]+1}',(20,675),cv2.FONT_HERSHEY_SIMPLEX,.6,(210,210,210),1)
                cv2.putText(canvas,'Green: usable crop   Orange: excluded visibility   Red point: box foot estimate',(20,705),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,180,180),1)
                encoder.stdin.write(canvas.tobytes())
            encoder.stdin.close();cap.release()
            if encoder.wait()!=0:raise RuntimeError(f'Encoding failed: {relative}')
    panel=old_pack[eid]['panel'];shutil.copy2(REC/panel,HERE/panel)
    manifest[eid]={'event_id':eid,'direction':events[eid]['direction'],'crossing_time':events[eid]['time'],'videos':clips,'panel':panel,'video_status':'continuous_snippet' if clips else 'static_evidence_only'}
    if len(manifest)%25==0:print(json.dumps({'stage':'video_evidence','endpoints':len(manifest),'total':len(needed),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
(HERE/'results/video_manifest.json').write_text(json.dumps(manifest,indent=2));record.update(stage='temporal_video_evidence_complete',endpoints=len(manifest),videos=sum(len(r['videos']) for r in manifest.values()),elapsed_seconds=time.time()-started)
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
print(json.dumps(record),flush=True)
