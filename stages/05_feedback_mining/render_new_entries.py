"""Render the two proposed entrances from native PTS and continuous observations."""
from datetime import datetime
import json,os,subprocess,time
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results';start=time.time();cv2.setNumThreads(2)
for folder in ('crops','contexts','panels','videos'):(HERE/folder).mkdir(exist_ok=True)
(OUT/'endpoints').mkdir(exist_ok=True)
ledger={'stage':'new_entry_evidence','pid':os.getpid(),'cwd':str(HERE),'gpu':None,'started_at':datetime.now().isoformat(),'output':str(HERE/'videos'),'stop_condition':'entrance clips complete or 600 seconds'}
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(ledger)+'\n')
boundary=np.array([[0,829],[450,807],[860,773],[1210,731],[1450,696],[1660,659],[1919,620]],np.int32)
for e in json.loads((OUT/'new_entry_proposals.json').read_text()):
    if (OUT/'endpoints'/f'{e["event_id"]:04d}.json').exists():continue
    eid=e['event_id'];clip=e['clip'];rows=e['measurements'];times=np.array([r['time'] for r in rows]);offset=json.loads((ROOT/'work/clocks'/f'VID{clip:05d}.json').read_text())['offset_seconds'];native=np.array([float(r['best_effort_timestamp_time']) for r in json.loads((ROOT/'work/native_crossings'/f'{clip:03d}_pts.json').read_text())['frames']])+offset
    chosen=sorted(set(np.rint(np.linspace(0,len(rows)-1,8)).astype(int).tolist()+[int(np.argmin(abs(times-e['time'])))]));views=[];tiles=[]
    cap=cv2.VideoCapture(str(ROOT/'raw/DCIMA'/f'VID{clip:05d}.AVI'))
    for n,k in enumerate(chosen,1):
        r=rows[k];fi=int(r['frame_index']);cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,frame=cap.read()
        if not ok:raise RuntimeError(f'Failed frame {clip}:{fi}')
        actual=float(native[fi]);x1,y1,x2,y2=np.rint(r['box']).astype(int);body=frame[max(0,y1):min(1080,y2),max(0,x1):min(1920,x2)].copy();crop=f'crops/new_e{eid:04d}_v{n}.jpg';context=f'contexts/new_e{eid:04d}_v{n}.jpg';cv2.imwrite(str(HERE/crop),body)
        cv2.polylines(frame,[boundary],False,(0,230,250),2);cv2.rectangle(frame,(x1,y1),(x2,y2),(70,220,90),3);cv2.circle(frame,tuple(np.rint(r['foot']).astype(int)),6,(30,50,250),-1);cv2.putText(frame,f'PROPOSED ENTRY {eid} V{n} PTS {actual:.3f}s',(30,40),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2);cv2.imwrite(str(HERE/context),cv2.resize(frame,(960,540)))
        views.append({'cell':f'V{n}','time':actual,'crop':crop,'context':context,'fragment':0,'side':'outside' if r['distance']>0 else 'inside','box':r['box'],'foot':r['foot'],'foot_source':r.get('foot_source','two_ankles'),'source_frame_index':fi})
        tile=np.full((400,200,3),28,np.uint8);scale=min(190/body.shape[1],340/body.shape[0]);body=cv2.resize(body,(round(body.shape[1]*scale),round(body.shape[0]*scale)));h,w=body.shape[:2];tile[40+(340-h)//2:40+(340-h)//2+h,(200-w)//2:(200-w)//2+w]=body;cv2.putText(tile,f'V{n} {actual-e["time"]:+.2f}s',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(tile)
    panel=np.full((800,1000,3),28,np.uint8)
    for k,tile in enumerate(tiles):panel[k//5*400:(k//5+1)*400,k%5*200:(k%5+1)*200]=tile
    panelpath=f'panels/new_e{eid:04d}.jpg';cv2.imwrite(str(HERE/panelpath),panel)
    relative=f'videos/new_e{eid:04d}_c{clip:03d}.mp4';encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','1280x720','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(HERE/relative)],stdin=subprocess.PIPE)
    indices=sorted(set(int(np.argmin(abs(native-t))) for t in np.arange(times[0],times[-1]+.001,.1)));cap.set(cv2.CAP_PROP_POS_FRAMES,indices[0]);position=indices[0]
    for fi in indices:
        while position<fi:
            if not cap.grab():raise RuntimeError(f'Failed decoding {clip}:{fi}')
            position+=1
        ok,frame=cap.read();position+=1
        if not ok:raise RuntimeError(f'Failed decoding {clip}:{fi}')
        at=float(native[fi]);hi=int(np.searchsorted(times,at));lo=max(0,hi-1);hi=min(hi,len(rows)-1);nearest=int(np.argmin(abs(times-at)));box=np.array(rows[nearest]['box']);foot=np.array(rows[nearest]['foot']);annotated=abs(times[nearest]-at)<=.15
        if hi>lo and times[hi]-times[lo]<=.3:
            w=(at-times[lo])/(times[hi]-times[lo]);box=(1-w)*np.array(rows[lo]['box'])+w*np.array(rows[hi]['box']);foot=(1-w)*np.array(rows[lo]['foot'])+w*np.array(rows[hi]['foot']);annotated=True
        canvas=np.full((720,1280,3),25,np.uint8)
        if annotated:
            x1,y1,x2,y2=np.rint(box).astype(int);body=frame[max(0,y1):min(1080,y2),max(0,x1):min(1920,x2)].copy();cv2.rectangle(frame,(x1,y1),(x2,y2),(70,220,90),3);cv2.circle(frame,tuple(np.rint(foot).astype(int)),6,(30,50,250),-1)
            scale=min(300/body.shape[1],640/body.shape[0]);body=cv2.resize(body,(round(body.shape[1]*scale),round(body.shape[0]*scale)));h,w=body.shape[:2];canvas[40+(640-h)//2:40+(640-h)//2+h,970+(300-w)//2:970+(300-w)//2+w]=body
        cv2.polylines(frame,[boundary],False,(0,230,250),2);canvas[90:630,:960]=cv2.resize(frame,(960,540));seconds=int(at);stamp=f'{seconds//3600:02d}:{seconds%3600//60:02d}:{at%60:05.2f}';cv2.putText(canvas,f'PROPOSED ENTRY {eid}   native PTS {stamp}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);cv2.putText(canvas,f'Crossing offset {at-e["time"]:+.2f}s; identity and entrance need review',(20,680),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1);encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close();cap.release()
    if encoder.wait()!=0:raise RuntimeError(relative)
    endpoint={'event_id':eid,'direction':'entry','time':e['time'],'interval':e['interval'],'source':e['source'],'crossing_confirmed_by_human':False,'inferred_foot_frames':e.get('inferred_foot_frames',0),'views':views,'panel':panelpath,'videos':[{'path':relative,'fragment':0,'source_clip':clip,'interval':[float(times[0]),float(times[-1])],'native_pts_stamped':True,'display_sampling_hz':10}]}
    (OUT/'endpoints'/f'{eid:04d}.json').write_text(json.dumps(endpoint));print(json.dumps({'event_id':eid,'views':len(views),'video':relative}),flush=True)
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps({**ledger,'stage':'new_entry_evidence_complete','elapsed_seconds':time.time()-start})+'\n')
