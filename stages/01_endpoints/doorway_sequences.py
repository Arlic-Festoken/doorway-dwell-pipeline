"""Rebuild doorway-only event sequences from native video PTS."""
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

root=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));old=root/'rebuild_20261107';new=Path(__file__).resolve().parent
for name in ['logs','sequences','crops','features','results']:(new/name).mkdir(exist_ok=True)
started=time.time();torch.set_num_threads(2);cv2.setNumThreads(2);torch.cuda.set_per_process_memory_fraction(.55)
source=old/'work/association_final';events=json.loads((source/'events_refined.json').read_text())
records=json.loads((source/'workbook_records_refined.json').read_text());nodes=json.loads((old/'work/bank/nodes.json').read_text())
paths=json.loads((source/'paths.json').read_text());obs=dict(np.load(old/'work/bank/observations.npz'))
manifest=json.loads((root/'source_inventory.json').read_text());boundary=np.asarray(json.loads((root/'scene.json').read_text())['boundary_pixels'])
observed={r['event_id']:r for r in records['candidates'] if r['native_crossing_observed']}
unverified={r['event_id'] for r in records['candidates'] if not r['native_crossing_observed']}
auto=[r for r in records['confirmed'] if r['source_pass'].startswith('automatic_')]
auto_entry={r['entry_event'] for r in auto};auto_exit={r['exit_event'] for r in auto}
historical=json.loads((source/'reference_recovery.json').read_text());locked=set()
for r in historical:
    if r['included_by_30s_rule']:
        locked.update(v['event_id'] for v in r['endpoints'].values() if v['event_id'] is not None)
for e in events:
    eid=e['event_id'];e['original_direction']=e['direction'];e['original_time']=e['time']
    e['source_category']='observed_unmatched_entry' if eid in observed and observed[eid]['entry_seconds'] is not None else 'observed_unmatched_exit' if eid in observed else 'unverified_event' if eid in unverified else 'automatic_entry' if eid in auto_entry else 'automatic_exit' if eid in auto_exit else 'historical_locked' if eid in locked else 'other_event'
(new/'events.json').write_text(json.dumps(events));(new/'previous_records.json').write_text(json.dumps(records,ensure_ascii=False))
(new/'reference_mapping.json').write_text(json.dumps(historical,ensure_ascii=False))
by_clip=defaultdict(list)
for e in events:
    rows=np.asarray([r for n in paths[e['path']] for r in nodes[n]['rows'] if abs(obs['time'][r]-e['time'])<=2.2],np.int64)
    rows=rows[np.argsort(obs['time'][rows])];boxes=obs['boxes'][rows,:4]
    roi=[max(0,int(boxes[:,0].min()-90)),max(0,int(boxes[:,1].min()-55)),min(1920,int(boxes[:,2].max()+90)),min(1080,int(boxes[:,3].max()+80))]
    for clip in [e['clip']-1,e['clip'],e['clip']+1]:
        if not 0<=clip<106:continue
        offset=json.loads((root/'work/clocks'/f'VID{clip:05d}.json').read_text())['offset_seconds']
        lo=max(0,e['time']-1.75-offset);hi=min(manifest[clip]['duration_seconds'],e['time']+1.75-offset)
        if hi>=lo:by_clip[clip].append({'event':e['event_id'],'low':lo,'high':hi,'clock':offset,'roi':roi,'rows':rows})
model=YOLO(str(root/'models/yolo11x-pose.pt'))
for clip,requests in sorted(by_clip.items()):
    target=new/'sequences'/f'clip_{clip:03d}.json'
    if target.exists():continue
    measurements=defaultdict(list);selected={}
    pts=[float(r['best_effort_timestamp_time']) for r in json.loads((root/'work/native_crossings'/f'{clip:03d}_pts.json').read_text())['frames']]
    chosen={round(t*10):i for i,t in enumerate(pts) if any(q['low']<=t<=q['high'] for q in requests)}
    indices=sorted(chosen.values());expression='+'.join(f'eq(n,{i})' for i in indices)
    command=['ffmpeg','-nostdin','-v','error','-threads','2','-filter_threads','2','-i',str(root/'raw/DCIMA'/manifest[clip]['file']),'-vf',f"select='{expression}'",'-frames:v',str(len(indices)),'-vsync','0','-an','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
    images=[];batch=[]
    def infer():
        predictions=model.predict(images,imgsz=640,conf=.08,iou=.75,rect=False,device=0,quantize='fp16',verbose=False)
        for result,q,im in zip(predictions,batch,images):
            if result.keypoints is None or len(result.boxes)==0:continue
            boxes=result.boxes.xyxy.cpu().numpy();pose=result.keypoints.data.cpu().numpy();confidence=result.boxes.conf.cpu().numpy()
            offset=np.asarray(q['roi'][:2]);boxes+=np.tile(offset,2);pose[:,:,:2]+=offset
            rows=q['rows'];stamp=q['time'];expected=np.asarray([np.interp(stamp,obs['time'][rows],obs['boxes'][rows,k]) for k in range(4)])
            inter=np.maximum(0,np.minimum(boxes[:,2:],expected[2:])-np.maximum(boxes[:,:2],expected[:2])).prod(1)
            union=(boxes[:,2:]-boxes[:,:2]).prod(1)+(expected[2:]-expected[:2]).prod()-inter
            overlap=inter/union.clip(1);order=np.argsort(-overlap);best=int(order[0])
            if overlap[best]<.25 or (len(order)>1 and overlap[best]-overlap[order[1]]<.10):continue
            p=pose[best];b=boxes[best];h=max(1,float(b[3]-b[1]));torso=p[[5,6,11,12],2]>=.4
            if torso.sum()<2:continue
            shoulders=p[[5,6]];hips=p[[11,12]];knees=p[[13,14]];ankles=p[[15,16]]
            valid=(ankles[:,2]>=.55)&(ankles[:,1]>knees[:,1])
            valid&=(knees[:,2]<.4)|((ankles[:,1]-knees[:,1])<3.2*np.maximum(15,knees[:,1]-hips[:,1]))
            foot=np.average(ankles[valid,:2],weights=ankles[valid,2],axis=0) if valid.any() else None
            d=float(foot[1]-np.interp(foot[0],boundary[:,0],boundary[:,1])) if foot is not None else None
            local=np.rint(b-np.tile(offset,2)).astype(int);local[[0,2]]=np.clip(local[[0,2]],0,im.shape[1]);local[[1,3]]=np.clip(local[[1,3]],0,im.shape[0])
            x1,y1,x2,y2=local
            if x2-x1<12 or y2-y1<24:continue
            crop=im[y1:y2,x1:x2];sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var())
            quality=float(overlap[best]*confidence[best]*(.3+.7*float(torso.mean()))*min(1,sharp/120)*min(1,h/180))
            shoulder_width=abs(float(shoulders[0,0]-shoulders[1,0]))/h
            face_visible=bool((p[:3,2]>=.5).sum()>=2)
            orientation='side' if min(shoulders[:,2])>=.4 and shoulder_width<.14 else 'front' if face_visible and min(shoulders[:,2])>=.4 else 'unknown'
            item={'time':stamp,'clip':clip,'frame_index':q['frame_index'],'box':b.tolist(),'pose':p.tolist(),'foot':None if foot is None else foot.tolist(),'distance':d,
                  'quality':quality,'match_iou':float(overlap[best]),'orientation':orientation,'orientation_is_heuristic':True,'height':h}
            eid=q['event'];measurements[eid].append(item)
            if abs(stamp-events[eid]['time'])<=1.15:
                slot=int(np.floor((stamp-events[eid]['time']+1.2)*4));key=(eid,slot)
                if key not in selected or quality>selected[key]['quality']:
                    path=new/'crops'/f'e{eid:04d}_c{clip:03d}_s{slot:02d}.jpg';cv2.imwrite(str(path),crop,[cv2.IMWRITE_JPEG_QUALITY,93])
                    selected[key]={**item,'event_id':eid,'path':str(path.relative_to(new))}
    with torch.inference_mode(), (new/'logs/decode.log').open('a') as log,subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log) as decoder:
        for i in indices:
            raw=decoder.stdout.read(1920*1080*3)
            if len(raw)!=1920*1080*3:raise RuntimeError(f'Incomplete native frame {clip}:{i}')
            frame=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
            for q in requests:
                if not q['low']<=pts[i]<=q['high']:continue
                x1,y1,x2,y2=q['roi'];images.append(frame[y1:y2,x1:x2].copy());batch.append({**q,'time':pts[i]+q['clock'],'frame_index':i})
                if len(images)==16:infer();images=[];batch=[]
        if decoder.wait():raise RuntimeError(f'Decode failed for clip {clip}')
        if images:infer()
    result={'measurements':dict(measurements),'crops':list(selected.values())}
    partial=target.with_suffix('.partial.json');partial.write_text(json.dumps(result));partial.replace(target)
    print(json.dumps({'stage':'doorway_sequences','clip':clip,'events':len(requests),'frames':len(indices),'pose_observations':sum(map(len,measurements.values())),'crops':len(selected),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
summary={'stage':'doorway_sequences_complete','pid':os.getpid(),'events':len(events),'elapsed_seconds':round(time.time()-started,1),'interior_identity_paths_reconstructed':False}
(new/'logs/sequences_status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
