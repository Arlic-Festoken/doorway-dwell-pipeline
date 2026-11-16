import os
"""Query-selected doorway windows: DEIM + CrowdDet, local body association, pose."""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

root=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));old=root/'rebuild_20261107';base=Path(__file__).resolve().parent;new=base/'query_round';started=time.time()
for folder in ('logs','sequences','features','crops','results'):(new/folder).mkdir(parents=True,exist_ok=True)
requests=json.loads((base/'results/targeted_requests.json').read_text());events=json.loads((base/'events.json').read_text());first_id=len(events)
manifest=json.loads((root/'source_inventory.json').read_text());line=np.asarray(json.loads((root/'scene.json').read_text())['boundary_pixels'])
clocks=[json.loads((root/'work/clocks'/f'VID{i:05d}.json').read_text())['offset_seconds'] for i in range(106)]
by_clip=defaultdict(list)
for q in requests:
    eid=len(events);q['event_id']=eid;seed=q['seed_time'];ss=sorted(q['seed_samples'],key=lambda s:s['time']);q['samples']=ss
    b=np.asarray(min(ss,key=lambda s:abs(s['time']-seed))['box']);clip=max(0,min(105,int(np.searchsorted(clocks,seed,side='right')-1)));h=b[3]-b[1]
    boxes=np.asarray([s['box'] for s in ss]);q['roi']=[max(0,int(boxes[:,0].min()-.65*h)),max(0,int(boxes[:,1].min()-.5*h)),min(1920,int(boxes[:,2].max()+.65*h)),min(1080,int(boxes[:,3].max()+.65*h))]
    events.append({'event_id':eid,'path':-1,'node':q['node'],'direction':'entry','time':seed,'interval':[seed-.1,seed+.1],'rows':[],
       'clip':clip,'frame':0,'box':b.tolist(),'boundary_x':float((b[0]+b[2])/2),'time_method':'retrieval_seed_unconfirmed',
       'original_direction':'entry','original_time':seed,'source_category':'query_recovered_candidate','query_exit_events':q['exit_queries']})
    for c in range(max(0,clip-1),min(106,clip+2)):
        lo=max(0,seed-2.5-clocks[c]);hi=min(manifest[c]['duration_seconds'],seed+2.5-clocks[c])
        if hi>=lo:by_clip[c].append({**q,'low':lo,'high':hi,'clock':clocks[c]})
(new/'events.json').write_text(json.dumps(events));(new/'query_requests.json').write_text(json.dumps(requests))
for name in ('reference_mapping.json','previous_records.json'):(new/name).write_bytes((base/name).read_bytes())
torch.set_num_threads(2);cv2.setNumThreads(2);torch.cuda.set_per_process_memory_fraction(.75)
sys.path.insert(0,str(old/'repos/DEIM'))
from engine.core import YAMLConfig
cfg=YAMLConfig(str(old/'repos/DEIM/configs/deim_dfine/object365/deim_hgnetv2_x_obj2coco_24e.yml'),resume=str(old/'weights/deim_dfine_x_obj365.pth'));cfg.yaml_cfg['HGNetv2']['pretrained']=False
state=torch.load(old/'weights/deim_dfine_x_obj365.pth',map_location='cpu',weights_only=False);cfg.model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model']);deim=cfg.model.deploy().cuda().eval();post=cfg.postprocessor.deploy().cuda().eval();del state
sys.path[:0]=[str(old/'repos/CrowdDet/lib'),str(old/'repos/CrowdDet/model/rcnn_emd_refine')]
from network import Network
from utils.nms_utils import set_cpu_nms
crowd=Network();state=torch.load(old/'weights/crowddet_author_refine.pth',map_location='cpu',weights_only=False);crowd.load_state_dict(state['state_dict'],strict=True);crowd.cuda().eval();del state
pose_model=YOLO(str(root/'models/yolo11x-pose.pt'))

def overlap(boxes,box):
    boxes=np.asarray(boxes);box=np.asarray(box);inter=np.maximum(0,np.minimum(boxes[:,2:4],box[2:4])-np.maximum(boxes[:,:2],box[:2])).prod(1)
    return inter/((boxes[:,2:4]-boxes[:,:2]).prod(1)+(box[2:4]-box[:2]).prod()-inter+1e-6)

total_obs=0;total_crops=0
for clip in range(106):
    target=new/'sequences'/f'clip_{clip:03d}.json'
    if target.exists():continue
    qs=by_clip[clip];frames=defaultdict(list);measurements=defaultdict(list);selected={};images=[];batch=[]
    def infer():
        inputs=np.stack([cv2.resize(im,(640,640))[:,:,::-1] for im in images]).transpose(0,3,1,2).copy();sizes=torch.tensor([[im.shape[1],im.shape[0]] for im in images],device='cuda',dtype=torch.float32)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):labels,boxes,confidence=post(deim(torch.from_numpy(inputs).cuda().float()/255),sizes)
        poses=pose_model.predict(images,imgsz=640,conf=.05,iou=.75,rect=False,device=0,verbose=False)
        for k,(im,q,result) in enumerate(zip(images,batch,poses)):
            keep=(labels[k]==0)&(confidence[k]>=.05);db=torch.cat([boxes[k][keep].float(),confidence[k][keep,None].float()],1).cpu().numpy()
            hh,ww=im.shape[:2];scale=min(800/hh,1400/ww);h=int(round(hh*scale));w=int(round(ww*scale));x=torch.from_numpy(cv2.resize(im,(w,h)).transpose(2,0,1).copy()).float().cuda()[None]
            with torch.inference_mode():pred=crowd(x,torch.tensor([[h,w,scale,hh,ww,0]],device='cuda',dtype=torch.float32)).numpy()
            groups=np.repeat(np.arange(len(pred)),pred.shape[1]//6);pred=np.column_stack([pred.reshape(-1,6),groups]);pred=pred[pred[:,4]>=.1];pred=pred[set_cpu_nms(pred,.5)];pred[:,:4]/=scale
            combined=np.concatenate([db,pred[:,:5]]);combined[:,[0,2]]=combined[:,[0,2]].clip(0,ww-1);combined[:,[1,3]]=combined[:,[1,3]].clip(0,hh-1)
            size=combined[:,2:4]-combined[:,:2];combined=combined[(size[:,0]>12)&(size[:,1]>35)];keep=nms(torch.from_numpy(combined[:,:4]),torch.from_numpy(combined[:,4]),.7).numpy();combined=combined[keep]
            shift=np.tile(q['roi'][:2],2);combined[:,:4]+=shift
            pb=result.boxes.xyxy.cpu().numpy()+shift;kp=result.keypoints.data.cpu().numpy();kp[:,:,:2]+=np.asarray(q['roi'][:2])
            frames[q['event_id']].append({'time':q['time'],'frame_index':q['frame_index'],'detections':combined,'pose_boxes':pb,'poses':kp,'pose_confidence':result.boxes.conf.cpu().numpy(),
              'roi':q['roi'],'image':cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,94])[1]})
    if qs:
        pts=[float(r['best_effort_timestamp_time']) for r in json.loads((root/'work/native_crossings'/f'{clip:03d}_pts.json').read_text())['frames']]
        chosen={round(t*5):i for i,t in enumerate(pts) if any(q['low']<=t<=q['high'] for q in qs)};indices=sorted(chosen.values());expr='+'.join(f'eq(n,{i})' for i in indices)
        cmd=['ffmpeg','-nostdin','-v','error','-threads','2','-filter_threads','2','-i',str(root/'raw/DCIMA'/manifest[clip]['file']),'-vf',f"select='{expr}'",'-frames:v',str(len(indices)),'-vsync','0','-an','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
        with torch.inference_mode(),(new/'logs/decode.log').open('a') as log,subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log) as decoder:
            for i in indices:
                raw=decoder.stdout.read(1920*1080*3)
                if len(raw)!=1920*1080*3:raise RuntimeError(f'Decode missing {clip}:{i}')
                frame=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
                for q in qs:
                    if q['low']<=pts[i]<=q['high']:
                        x1,y1,x2,y2=q['roi'];images.append(frame[y1:y2,x1:x2].copy());batch.append({**q,'time':pts[i]+q['clock'],'frame_index':i})
                        if len(images)==8:infer();images=[];batch=[]
            if decoder.wait():raise RuntimeError(f'Decode failed {clip}')
            if images:infer()
    for q in qs:
        eid=q['event_id'];rows=sorted(frames[eid],key=lambda r:r['time'])
        if not rows:continue
        ss=q['samples'];st=np.asarray([s['time'] for s in ss]);sb=np.asarray([s['box'] for s in ss]);seed_index=min(range(len(rows)),key=lambda i:abs(rows[i]['time']-q['seed_time']))
        selected_body={}
        for traversal in (range(seed_index,len(rows)),range(seed_index,-1,-1)):
            last_box=None;last_time=None;velocity=np.zeros(4)
            for i in traversal:
                r=rows[i];d=r['detections']
                if not len(d):continue
                expected=np.asarray([np.interp(r['time'],st,sb[:,k]) for k in range(4)])
                if last_box is not None:
                    if abs(r['time']-last_time)>.85:break
                    predicted=last_box+velocity*(r['time']-last_time);ov=.75*overlap(d,predicted)+.25*overlap(d,expected)
                else:ov=overlap(d,expected)
                order=np.argsort(-ov);best=order[0]
                if ov[best]<.2 or (len(order)>1 and ov[best]-ov[order[1]]<.06):continue
                box=d[best,:4]
                if last_box is not None and abs(r['time']-last_time)>.01:velocity=.5*velocity+.5*(box-last_box)/(r['time']-last_time)
                last_box=box;last_time=r['time'];selected_body[i]=box
        for i,body in sorted(selected_body.items()):
            r=rows[i]
            if not len(r['pose_boxes']):continue
            ov=overlap(r['pose_boxes'],body);order=np.argsort(-ov);j=int(order[0])
            if ov[j]<.3 or (len(order)>1 and ov[j]-ov[order[1]]<.1):continue
            p=r['poses'][j];b=r['pose_boxes'][j];height=float(b[3]-b[1]);torso=p[[5,6,11,12],2]>=.4
            if torso.sum()<2:continue
            ankles=p[[15,16]];knees=p[[13,14]];hips=p[[11,12]];visible=(ankles[:,2]>=.55)&(ankles[:,1]>knees[:,1]);visible&=(knees[:,2]<.4)|((ankles[:,1]-knees[:,1])<3.2*np.maximum(15,knees[:,1]-hips[:,1]))
            foot=np.average(ankles[visible,:2],weights=ankles[visible,2],axis=0) if visible.any() else None;distance=float(foot[1]-np.interp(foot[0],line[:,0],line[:,1])) if foot is not None else None
            image=cv2.imdecode(r['image'],cv2.IMREAD_COLOR);local=np.rint(b-np.tile(r['roi'][:2],2)).astype(int);local[[0,2]]=local[[0,2]].clip(0,image.shape[1]);local[[1,3]]=local[[1,3]].clip(0,image.shape[0]);x1,y1,x2,y2=local
            if x2-x1<12 or y2-y1<24:continue
            crop=image[y1:y2,x1:x2];sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var());quality=float(ov[j]*r['pose_confidence'][j]*(.3+.7*torso.mean())*min(1,sharp/120)*min(1,height/180))
            width=abs(p[5,0]-p[6,0])/height;orientation='side' if min(p[[5,6],2])>=.4 and width<.14 else 'front' if (p[:3,2]>=.5).sum()>=2 else 'unknown'
            item={'time':r['time'],'clip':clip,'frame_index':r['frame_index'],'box':b.tolist(),'pose':p.tolist(),'foot':None if foot is None else foot.tolist(),'distance':distance,
                'quality':quality,'match_iou':float(ov[j]),'orientation':orientation,'orientation_is_heuristic':True,'height':height}
            measurements[eid].append(item)
            if abs(r['time']-q['seed_time'])<=1.15:
                slot=int(np.floor((r['time']-q['seed_time']+1.2)*4));key=(eid,slot)
                if key not in selected or quality>selected[key]['quality']:
                    path=new/'crops'/f'e{eid:04d}_c{clip:03d}_s{slot:02d}.jpg';cv2.imwrite(str(path),crop,[cv2.IMWRITE_JPEG_QUALITY,93]);selected[key]={**item,'event_id':eid,'path':str(path.relative_to(new))}
    payload={'measurements':dict(measurements),'crops':list(selected.values())};partial=target.with_suffix('.partial.json');partial.write_text(json.dumps(payload));partial.replace(target)
    total_obs+=sum(map(len,measurements.values()));total_crops+=len(selected)
    print(json.dumps({'stage':'query_redetection','clip':clip,'requests':len(qs),'observations':sum(map(len,measurements.values())),'crops':len(selected),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
summary={'stage':'query_sequences_complete','requests':len(requests),'new_event_first_id':first_id,'pose_observations':total_obs,'new_crops':total_crops,'elapsed_seconds':round(time.time()-started,1)}
(new/'logs/sequences_status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
