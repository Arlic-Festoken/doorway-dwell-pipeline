import os
"""Existing COPE and SapiensID models applied only to doorway sequence crops."""
from pathlib import Path
from types import MethodType
import fcntl
import json
import sys
import time

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from ultralytics import YOLO

root=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));old=root/'rebuild_20261107';new=Path(__file__).resolve().parent;started=time.time()
feature_lock=(new/'logs/features.lock').open('w')
fcntl.flock(feature_lock,fcntl.LOCK_EX)
if (new/'logs/features_status.json').exists():sys.exit(0)
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.55)
sys.path.insert(0,str(old/'repos/COPE'))
from config import cfg
from clip.model import CLIP
import cope.model as architecture
cfg=cfg.clone();cfg.merge_from_file(str(old/'repos/COPE/configs/OCC_Duke/cope.yml'))
state=torch.load(old/'weights/cope_occduke.pth',map_location='cpu',weights_only=True);state={k.removeprefix('module.'):v for k,v in state.items()}
def clip_architecture(name,h,w,stride):
    return CLIP(embed_dim=512,image_resolution=224,vision_layers=12,vision_width=768,vision_patch_size=16,vision_stride_size=stride,
        context_length=77,vocab_size=49408,transformer_width=512,transformer_heads=8,transformer_layers=12,h_resolution=h,w_resolution=w)
architecture.load_clip_to_cpu=clip_architecture
cope=architecture.make_model(cfg,state['classifier.weight'].shape[0],camera_num=0,view_num=0);cope.load_state_dict(state,strict=True);cope.eval().cuda();del state
ct=T.Compose([T.Resize(cfg.INPUT.SIZE_TEST,interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD)])
adapter=torch.nn.Sequential(torch.nn.Linear(1280,128,bias=False),torch.nn.ReLU(),torch.nn.Linear(128,1280,bias=False)).cuda()
adapter.load_state_dict(torch.load(old/'work/bank/scene_adapter.pt',map_location='cpu',weights_only=True)['state_dict']);adapter.eval()
repo=old/'repos/sapiensid';sys.path[:0]=[str(repo/'tasks/sapiensID'),str(repo)]
from src.models import get_model
import src.aligners.keypoint_predictor as aligner_module
config=OmegaConf.load(old/'weights/sapiensid_wb4m/model.yaml');sapiens=get_model(config)
sapiens.load_state_dict(torch.load(old/'weights/sapiensid_wb4m/model.pth',map_location='cpu',weights_only=True),strict=True);sapiens.eval().cuda()
ac=OmegaConf.load(repo/'tasks/sapiensID/src/aligners/configs/yolo_dfa.yaml');ac.rgb_mean=config.rgb_mean;ac.rgb_std=config.rgb_std;ac.use_cache=False
aligner_module.YOLO=lambda unused:YOLO(str(repo/'tasks/WebBody/c_detect/yolov8n-pose.pt'));aligner=aligner_module.KeypointPredictor(ac).cuda();aligner.eval()
def infer_pose(self,images):
    results=self.yolo_net(images,device=images.device,verbose=False);empty=torch.full((17,2),-1.,device=images.device)
    kps=torch.stack([r.keypoints.xyn[0] if len(r.keypoints.xyn)>0 else empty for r in results]);kps[kps==0]=-1;return kps
aligner.infer_using_yolo=MethodType(infer_pose,aligner);st=sapiens.make_test_transform()
total=0
for clip in range(106):
    target=new/'features'/f'clip_{clip:03d}.npz'
    if target.exists():continue
    path=new/'sequences'/f'clip_{clip:03d}.json'
    while not path.exists():
        if time.time()-started>3500:raise TimeoutError(str(path))
        time.sleep(3)
    crops=json.loads(path.read_text())['crops'];cf=[];sf=[];rel=[]
    with torch.inference_mode():
        for lo in range(0,len(crops),24):
            group=crops[lo:lo+24];images=[Image.open(new/r['path']).convert('RGB') for r in group]
            with torch.autocast('cuda',dtype=torch.float16):
                c,p=cope(torch.stack([ct(im) for im in images]).cuda());c=torch.nn.functional.normalize(c.float(),dim=1)
            c=torch.nn.functional.normalize(c+adapter(c),dim=1);cf.append(c.cpu().numpy().astype(np.float16));rel.append(p.float().cpu().numpy())
            for sub in range(0,len(images),8):
                x=torch.stack([st(im) for im in images[sub:sub+8]]).cuda();kps,masks=aligner(x)
                with torch.autocast('cuda',dtype=torch.float16):z=sapiens(x,foreground_masks=masks,ldmks=kps)
                if hasattr(z,'pooler_output'):z=z.pooler_output
                z=torch.nn.functional.normalize(z.float(),dim=1)
                if not torch.isfinite(z).all():raise ValueError(f'Nonfinite features in clip {clip}')
                sf.append(z.cpu().numpy().astype(np.float16))
    tmp=target.with_suffix('.partial.npz');np.savez_compressed(tmp,keys=np.asarray([r['path'] for r in crops]),cope=np.concatenate(cf) if cf else np.zeros((0,1280),np.float16),
        sapiens=np.concatenate(sf) if sf else np.zeros((0,4096),np.float16),reliability=np.concatenate(rel) if rel else np.zeros(0,np.float32));tmp.replace(target)
    total+=len(crops);print(json.dumps({'stage':'doorway_features','clip':clip,'crops':len(crops),'total':total,'elapsed_seconds':round(time.time()-started,1)}),flush=True)
(new/'logs/features_status.json').write_text(json.dumps({'complete':True,'crops':total,'elapsed_seconds':round(time.time()-started,1)}))
