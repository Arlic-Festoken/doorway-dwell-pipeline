"""Render the recovered original crops and their source-frame locations."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;started=time.time();cv2.setNumThreads(2)
pack=json.loads((HERE/'results/evidence_manifest.json').read_text())['events'];inventory=json.loads((ROOT/'source_inventory.json').read_text());boundary=np.asarray(json.loads((ROOT/'scene.json').read_text())['boundary_pixels']);requests=defaultdict(lambda:defaultdict(list))
for e in pack:
    for f in e['frames']:
        shutil.copyfile(f['source_crop'],HERE/f['crop']);requests[f['clip']][f['native_frame_index']].append(f)
for clip,by_frame in sorted(requests.items()):
    indices=sorted(i for i,frames in by_frame.items() if any(not (HERE/f['context']).exists() for f in frames))
    if not indices:continue
    expression='+'.join(f'eq(n,{i})' for i in indices);command=['ffmpeg','-nostdin','-v','error','-threads','2','-filter_threads','2','-i',str(ROOT/'raw/DCIMA'/inventory[clip]['file']),'-vf',f"select='{expression}'",'-frames:v',str(len(indices)),'-vsync','0','-an','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
    with (HERE/'logs/decode.log').open('a') as log,subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log) as decoder:
        for index in indices:
            raw=decoder.stdout.read(1920*1080*3)
            if len(raw)!=1920*1080*3:raise RuntimeError(f'Incomplete frame {clip}:{index}')
            frame=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
            for f in by_frame[index]:
                thumb=cv2.resize(frame,(640,360));box=np.rint(np.asarray(f['box'])/3).astype(int)
                cv2.polylines(thumb,[np.rint(boundary/3).astype(np.int32)],False,(0,210,255),1);cv2.rectangle(thumb,tuple(box[:2]),tuple(box[2:]),(0,255,0),2);cv2.circle(thumb,tuple(np.rint(np.asarray(f['foot'])/3).astype(int)),4,(0,0,255),-1)
                cv2.imwrite(str(HERE/f['context']),thumb,[cv2.IMWRITE_JPEG_QUALITY,92])
        if decoder.wait():raise RuntimeError(f'Decode failed: {clip}')
    print(json.dumps({'clip':clip,'frames':len(indices),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
for e in pack:
    frames=e['frames'];image=Image.new('RGB',(1280,80+680*max(1,int(np.ceil(len(frames)/4)))),'#f5f5f5');draw=ImageDraw.Draw(image)
    draw.text((12,8),f"Event {e['event_id']} | {e['kind']} | {len(frames)} displayed views",fill='black',font=font)
    draw.text((12,36),'Green: target body. Red: foot. Yellow: doorway. Estimated feet are labelled.',fill='black',font=font)
    for k,f in enumerate(frames):
        x=(k%4)*320;y=80+(k//4)*680;t=f['time'];draw.text((x+4,y),f"{f['cell']}  {int(t)//3600:02}:{int(t)%3600//60:02}:{t%60:05.2f}",fill='black',font=font)
        crop=Image.open(HERE/f['crop']);crop.thumbnail((312,450));image.paste(crop,(x+(320-crop.width)//2,y+28+(450-crop.height)//2))
        context=Image.open(HERE/f['context']);context.thumbnail((320,180));image.paste(context,(x,y+494))
        if f['foot_method']=='box_estimate':draw.text((x+4,y+473),'FOOT ESTIMATED',fill='#9b2929',font=font)
    image.save(HERE/e['panel'],quality=92)
summary={'endpoints':len(pack),'frames':sum(len(e['frames']) for e in pack),'elapsed_seconds':round(time.time()-started,1)};(HERE/'results/render_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
