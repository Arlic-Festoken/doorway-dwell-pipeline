import os
"""Native-PTS timing for newly recovered endpoints after identity assignment."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();stage=args.stage;root=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));started=time.time()
proposal=json.loads((stage/'results/assignment_proposal.json').read_text());events=json.loads((stage/'events_states.json').read_text());summary=json.loads((stage/'results/assignment_summary.json').read_text())
if not summary['accepted_for_delivery']:raise ValueError('Assignment did not pass historical recovery gate')
endpoint_ids={r[k] for r in proposal['automatic'] for k in ('entry_event','exit_event')};requested={i for i in endpoint_ids if not events[i]['native_previously_observed']};pending=set(requested);cached_refined={}
parent_cache=stage.parent/'results/native_delivery_times.json'
if parent_cache.exists():
    cached_refined={int(k):v for k,v in json.loads(parent_cache.read_text())['refined'].items() if int(k) in pending};pending-=set(cached_refined)
line=np.asarray(json.loads((root/'scene.json').read_text())['boundary_pixels']);manifest=json.loads((root/'source_inventory.json').read_text());measurements=defaultdict(list);by_clip=defaultdict(list)
for f in (stage/'sequences').glob('clip_*.json'):
    for i,rows in json.loads(f.read_text())['measurements'].items():
        if int(i) in pending:measurements[int(i)].extend(rows)
for i in pending:
    e=events[i];rows=sorted(measurements[i],key=lambda r:r['time']);b=np.asarray([r['box'] for r in rows]);roi=[max(0,int(b[:,0].min()-30)),max(0,int(b[:,1].min()-30)),min(1920,int(b[:,2].max()+30)),min(1080,int(b[:,3].max()+40))]
    for clip in range(max(0,e['clip']-1),min(106,e['clip']+2)):
        offset=json.loads((root/'work/clocks'/f'VID{clip:05d}.json').read_text())['offset_seconds'];lo=max(0,e['interval'][0]-.25-offset);hi=min(manifest[clip]['duration_seconds'],e['interval'][1]+.25-offset)
        if hi>=lo:by_clip[clip].append({'event_id':i,'low':lo,'high':hi,'clock':offset,'roi':roi,'rows':rows})
torch.set_num_threads(2);cv2.setNumThreads(2);torch.cuda.set_per_process_memory_fraction(.45);model=YOLO(str(root/'models/yolo11x-pose.pt'));native=defaultdict(list)
for clip,requests in sorted(by_clip.items()):
    pts=[float(r['best_effort_timestamp_time']) for r in json.loads((root/'work/native_crossings'/f'{clip:03d}_pts.json').read_text())['frames']];indices=[i for i,t in enumerate(pts) if any(q['low']<=t<=q['high'] for q in requests)];expr='+'.join(f'eq(n,{i})' for i in indices);images=[];batch=[]
    def infer():
        results=model.predict(images,imgsz=640,conf=.08,iou=.75,rect=False,device=0,verbose=False)
        for r,q in zip(results,batch):
            if not len(r.boxes):continue
            boxes=r.boxes.xyxy.cpu().numpy()+np.tile(q['roi'][:2],2);p=r.keypoints.data.cpu().numpy();p[:,:,:2]+=np.asarray(q['roi'][:2]);t=q['time'];rows=q['rows'];ts=[x['time'] for x in rows];bs=np.asarray([x['box'] for x in rows]);expected=np.array([np.interp(t,ts,bs[:,k]) for k in range(4)])
            inter=np.maximum(0,np.minimum(boxes[:,2:],expected[2:])-np.maximum(boxes[:,:2],expected[:2])).prod(1);ov=inter/((boxes[:,2:]-boxes[:,:2]).prod(1)+(expected[2:]-expected[:2]).prod()-inter+1e-6);order=np.argsort(-ov);j=order[0]
            if ov[j]<.3 or (len(order)>1 and ov[j]-ov[order[1]]<.1):continue
            pose=p[j];ankles=pose[[15,16]];knees=pose[[13,14]];hips=pose[[11,12]];ok=(ankles[:,2]>=.55)&(ankles[:,1]>knees[:,1]);ok&=(knees[:,2]<.4)|((ankles[:,1]-knees[:,1])<3.2*np.maximum(15,knees[:,1]-hips[:,1]))
            if not ok.any():continue
            foot=np.average(ankles[ok,:2],weights=ankles[ok,2],axis=0);d=float(foot[1]-np.interp(foot[0],line[:,0],line[:,1]));native[q['event_id']].append({'time':t,'distance':d,'foot':foot.tolist(),'height':float(boxes[j,3]-boxes[j,1]),'clip':clip,'frame_index':q['frame_index']})
    cmd=['ffmpeg','-nostdin','-v','error','-threads','2','-filter_threads','2','-i',str(root/'raw/DCIMA'/manifest[clip]['file']),'-vf',f"select='{expr}'",'-frames:v',str(len(indices)),'-vsync','0','-an','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
    with torch.inference_mode(),(stage/'logs/native_delivery_decode.log').open('a') as log,subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log) as decoder:
        for i in indices:
            raw=decoder.stdout.read(1920*1080*3)
            if len(raw)!=1920*1080*3:raise RuntimeError(f'Missing frame {clip}:{i}')
            frame=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
            for q in requests:
                if q['low']<=pts[i]<=q['high']:
                    x1,y1,x2,y2=q['roi'];images.append(frame[y1:y2,x1:x2].copy());batch.append({**q,'time':pts[i]+q['clock'],'frame_index':i})
                    if len(images)==16:infer();images=[];batch=[]
        if decoder.wait():raise RuntimeError(f'Decode failure {clip}')
        if images:infer()
    print(json.dumps({'stage':'native_delivery_times','clip':clip,'events':len(requests),'native_frames':len(indices),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
refined=dict(cached_refined)
for i in pending:
    rows=sorted(native[i],key=lambda r:r['time']);e=events[i];options=[];sign=1 if e['direction']=='entry' else -1
    for a,b in zip(rows[:-1],rows[1:]):
        if not (0<b['time']-a['time']<=.18 and sign*a['distance']>0 and sign*b['distance']<=0):continue
        if np.linalg.norm(np.asarray(a['foot'])-b['foot'])>.25*min(a['height'],b['height']):continue
        t=a['time']+a['distance']/(a['distance']-b['distance'])*(b['time']-a['time'])
        if not e['interval'][0]<=t<=e['interval'][1]:continue
        before=[r for r in rows if t-.25<=r['time']<t];after=[r for r in rows if t<r['time']<=t+.25]
        if len(before)>=3 and len(after)>=3 and np.mean([sign*r['distance']>0 for r in before])>=.75 and np.mean([sign*r['distance']<0 for r in after])>=.75:options.append({'time':float(t),'interval':[a['time'],b['time']],'method':'native_ankles_query_refinement'})
    if options:refined[i]=min(options,key=lambda r:abs(r['time']-e['time']))
automatic=[];new_short=[]
for r in proposal['automatic']:
    for side in ('entry','exit'):
        i=r[side+'_event']
        if i in refined:r[side+'_seconds']=refined[i]['time'];r[side+'_interval']=refined[i]['interval'];r[side+'_time_method']=refined[i]['method']
    r['duration_seconds']=r['exit_seconds']-r['entry_seconds'];(automatic if r['duration_seconds']>=30 else new_short).append(r)
candidates=[]
for e in proposal['unmatched']:
    candidates.append({'event_id':e['event_id'],'entry_seconds':e['time'] if e['direction']=='entry' else None,'exit_seconds':e['time'] if e['direction']=='exit' else None,'appearance':'',
        'direction':e['direction'],'native_crossing_observed':events[e['event_id']]['native_previously_observed'],'crossing_confirmed':True,'crossing_evidence':'native_or_two_side_state','interval':e['interval'],'source_category':e['source_category']})
duplicates={r['event_id'] for r in json.loads((stage/'results/duplicate_events.json').read_text())}
for e in events:
    if not e['eligible_crossing'] and e['event_id'] not in duplicates:candidates.append({'event_id':e['event_id'],'entry_seconds':e['time'] if e['direction']=='entry' else None,'exit_seconds':e['time'] if e['direction']=='exit' else None,'appearance':'','direction':e['direction'],'native_crossing_observed':False,'crossing_confirmed':False,'source_category':e['source_category'],'interval':e['interval']})
confirmed=sorted(automatic+proposal['historical'],key=lambda r:r['entry_seconds']);candidates.sort(key=lambda r:r['entry_seconds'] if r['entry_seconds'] is not None else r['exit_seconds'])
records={'mode':'automatic_doorway_20261108','confirmed':confirmed,'candidates':candidates,'unclosed':proposal['unmatched'],'minimum_duration_seconds':30,
         'automatic_records_are_independently_confirmed':False,'new_automatic_records':len(automatic),'historical_confirmed_records':len(proposal['historical'])}
(stage/'results/delivery_records.json').write_text(json.dumps(records,ensure_ascii=False));(stage/'results/native_delivery_times.json').write_text(json.dumps({'requested':sorted(requested),'reused_native_refinements':len(cached_refined),'refined':refined,'measurements':dict(native),'new_short_records':new_short}))
final={**summary,'complete_records':len(confirmed),'automatic_records':len(automatic),'historical_records':len(proposal['historical']),'native_refinement_requested':len(requested),'native_refinement_succeeded':len(refined),
       'newly_short_after_time_refinement':len(new_short),'approximate_complete_records':sum(any(r.get(s+'_interval',[0,0])[1]-r.get(s+'_interval',[0,0])[0]>.6 for s in ('entry','exit')) for r in confirmed),
       'unverified_candidates':sum(not r['crossing_confirmed'] for r in candidates),'native_refinement_elapsed_seconds':round(time.time()-started,1)}
(stage/'results/delivery_report.json').write_text(json.dumps(final,indent=2));print(json.dumps(final),flush=True)
