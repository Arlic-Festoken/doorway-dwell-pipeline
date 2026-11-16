import os
"""Encode newly sampled ordered RGB frames using the installed COPE backbone."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);args=parser.parse_args()
ROOT=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));HERE=Path(__file__).resolve().parent;OLD=ROOT/'rebuild_20261107';started=time.time()
torch.set_num_threads(2);cv2.setNumThreads(2);torch.cuda.set_per_process_memory_fraction(.45)
sys.path.insert(0,str(OLD/'repos/COPE'))
from config import cfg
from clip.model import CLIP
import cope.model as architecture
cfg=cfg.clone();cfg.merge_from_file(str(OLD/'repos/COPE/configs/OCC_Duke/cope.yml'))
state=torch.load(OLD/'weights/cope_occduke.pth',map_location='cpu',weights_only=True);state={k.removeprefix('module.'):v for k,v in state.items()}
def checkpoint_architecture(name,h,w,stride):
    return CLIP(embed_dim=512,image_resolution=224,vision_layers=12,vision_width=768,vision_patch_size=16,vision_stride_size=stride,context_length=77,vocab_size=49408,transformer_width=512,transformer_heads=8,transformer_layers=12,h_resolution=h,w_resolution=w)
architecture.load_clip_to_cpu=checkpoint_architecture
model=architecture.make_model(cfg,state['classifier.weight'].shape[0],camera_num=0,view_num=0);model.load_state_dict(state,strict=True);model.eval().cuda();del state
transform=T.Compose([T.Resize(cfg.INPUT.SIZE_TEST,interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD)])
frames=json.loads((HERE/'frames.json').read_text());by_clip=defaultdict(list)
for r in frames:
    if r['clip']%2==args.shard:by_clip[r['clip']].append(r)
for clip,rows in sorted(by_clip.items()):
    dest=HERE/'features'/f'clip_{clip:03d}.npz'
    if dest.exists():continue
    rows=sorted(rows,key=lambda r:r['native_frame_index']);by_frame=defaultdict(list)
    for r in rows:by_frame[r['native_frame_index']].append(r)
    cap=cv2.VideoCapture(str(ROOT/'raw/DCIMA'/f'VID{clip:05d}.AVI'));position=0;batch=[];batch_keys=[];keys=[];features=[];reliability=[];colors=[]
    def flush():
        if not batch:return
        images=torch.stack(batch).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):v,p=model(images);v=torch.nn.functional.normalize(v.float(),dim=1)
        features.append(v.cpu().numpy().astype(np.float16));reliability.extend(p.float().cpu().numpy().reshape(-1).tolist());keys.extend(batch_keys);batch.clear();batch_keys.clear()
    for fi,group in sorted(by_frame.items()):
        if fi-position>30:cap.set(cv2.CAP_PROP_POS_FRAMES,fi);position=fi
        while position<fi:
            if not cap.grab():raise ValueError(f'Cannot seek clip {clip} frame {fi}')
            position+=1
        ok,image=cap.read();position+=1
        if not ok:raise ValueError(f'Cannot decode clip {clip} frame {fi}')
        for r in group:
            x1,y1,x2,y2=np.rint(r['box']).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(image.shape[1],x2);y2=min(image.shape[0],y2)
            crop=image[y1:y2,x1:x2]
            if not crop.size:raise ValueError(f'Empty crop {r["key"]}')
            if not cv2.imwrite(str(HERE/r['path']),crop,[cv2.IMWRITE_JPEG_QUALITY,92]):raise OSError(r['path'])
            lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB);h,w=lab.shape[:2];parts=[]
            for low,high in ((.2,.48),(.55,.85)):
                p=lab[int(h*low):max(int(h*low)+1,int(h*high)),int(w*.2):max(int(w*.2)+1,int(w*.8))].reshape(-1,3)
                hist=np.concatenate([np.histogram(p[:,c],bins=16,range=(0,256))[0] for c in range(3)]).astype(np.float32);hist/=max(hist.sum(),1);parts.append(np.sqrt(hist))
            colors.append(parts);batch.append(transform(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))));batch_keys.append(r['key'])
            if len(batch)>=40:flush()
    flush();cap.release()
    np.savez_compressed(dest,keys=np.asarray(keys),features=np.concatenate(features),reliability=np.asarray(reliability),colors=np.asarray(colors,dtype=np.float16))
    status={'stage':'ordered_rgb_encoding','shard':args.shard,'clip':clip,'frames':len(keys),'elapsed_seconds':round(time.time()-started,1)}
    (HERE/'logs'/f'extract_{args.shard}_status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
print(json.dumps({'stage':'encoding_complete','shard':args.shard,'elapsed_seconds':time.time()-started}),flush=True)
