from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from temporal_model import TemporalEncoder

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;started=time.time()
torch.set_num_threads(2);torch.manual_seed(1109);np.random.seed(1109);torch.cuda.set_per_process_memory_fraction(.4)
sequences=json.loads((HERE/'sequences.json').read_text());frames={r['key']:r for r in json.loads((HERE/'frames.json').read_text())}
feature={};color={};reliability={}
for path in sorted((HERE/'features').glob('clip_*.npz')):
    z=np.load(path)
    for k,f,c,r in zip(z['keys'],z['features'],z['colors'],z['reliability']):feature[str(k)]=f.astype(np.float32);color[str(k)]=c.astype(np.float32);reliability[str(k)]=float(r)
X=np.zeros((len(sequences),32,1280),np.float32);G=np.zeros((len(sequences),32,7),np.float32);M=np.zeros((len(sequences),32),np.float32)
colors=np.zeros((len(sequences),32,2,48),np.float32);times=np.zeros((len(sequences),32),np.float64);boxes=np.zeros((len(sequences),32,4),np.float32)
for i,s in enumerate(sequences):
    records=[frames[k] for k in s['frames']];n=len(records);b=np.array([r['box'] for r in records]);t=np.array([r['time'] for r in records]);h=b[:,3]-b[:,1];w=b[:,2]-b[:,0];foot=np.c_[(b[:,0]+b[:,2])/2,b[:,3]]
    times[i,:n]=t;boxes[i,:n]=b;dt=np.r_[.2,np.diff(t)];motion=np.vstack([np.zeros((1,2)),np.diff(foot,axis=0)])/h[:,None]/dt[:,None].clip(.03)
    G[i,:n]=np.c_[np.log(h/w),np.log(h/np.median(h)),np.clip(motion,-3,3),dt,t-t[0],np.ones(n)]
    for j,r in enumerate(records):
        k=r['key'];X[i,j]=feature[k];colors[i,j]=color[k];M[i,j]=r['visible'] and reliability[k]>=.6
    s['index']=i;s['fold']=int(hashlib.sha256(f'raw{ s["raw_track"] }'.encode()).hexdigest()[:8],16)%5
del feature,color,reliability
base=(X*M[:,:,None]).sum(1)/M.sum(1).clip(1)[:,None];base/=np.linalg.norm(base,axis=1,keepdims=True).clip(1e-8)

def distinct(i,j):
    a=sequences[i];b=sequences[j]
    if a['raw_track']==b['raw_track']:return False
    low=max(a['time_interval'][0],b['time_interval'][0]);high=min(a['time_interval'][1],b['time_interval'][1])
    if high-low<.5:return False
    at=(low+high)/2;ia=np.argmin(abs(times[i,:len(a['frames'])]-at));ib=np.argmin(abs(times[j,:len(b['frames'])]-at))
    if abs(times[i,ia]-times[j,ib])>.12 or not (M[i,ia] and M[j,ib]):return False
    aa=boxes[i,ia];bb=boxes[j,ib];inter=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])).prod();iou=inter/((aa[2:]-aa[:2]).prod()+(bb[2:]-bb[:2]).prod()-inter)
    return iou<.01 and np.linalg.norm((aa[:2]+aa[2:]-bb[:2]-bb[2:])/2)>.5*min(aa[3]-aa[1],bb[3]-bb[1])

training_ids=[i for i,s in enumerate(sequences) if s['kind']=='train' and M[i].sum()>=12]
pairs=[]
for i in training_ids:
    partition='fit' if sequences[i]['fold']<3 else 'calibration'
    n=len(sequences[i]['frames']);cut=n//2
    if min(M[i,:cut].sum(),M[i,cut:n].sum())>=5:pairs.append({'a':i,'b':i,'a_span':[0,cut],'b_span':[cut,n],'label':1,'partition':partition,'source':'nonoverlapping_continuous_halves'})
    others=[j for j in training_ids if j!=i and (sequences[j]['fold']<3)==(partition=='fit') and distinct(i,j)]
    for j in sorted(others,key=lambda j:float(base[i]@base[j]),reverse=True)[:8]:
        if i<j:pairs.append({'a':i,'b':j,'a_span':[0,n],'b_span':[0,len(sequences[j]['frames'])],'label':0,'partition':partition,'source':'simultaneous_spatially_separate_bodies'})
counts={p:dict(Counter(r['label'] for r in pairs if r['partition']==p)) for p in ('fit','calibration')}
if any(min(v.get(0,0),v.get(1,0))<20 for v in counts.values()):raise ValueError(f'Too few automatically justified training examples: {counts}')
(HERE/'results/temporal_sample_provenance.json').write_text(json.dumps(pairs))
print(json.dumps({'stage':'training_samples','counts':counts}),flush=True)

def batch_view(records,side,augment=False):
    xs=[];gs=[];ms=[]
    for record in records:
        i=record[side];lo,hi=record[side+'_span'];x=X[i,lo:hi].copy();g=G[i,lo:hi].copy();m=M[i,lo:hi].copy()
        g[:,5]-=g[0,5];g[0,2:4]=0;g[0,4]=.2
        if augment and np.random.rand()<.5:
            x=x[::-1].copy();g=g[::-1].copy();m=m[::-1].copy();g[:,2:4]*=-1;g[:,5]=g[0,5]-g[:,5]
        if augment:
            drop=(np.random.rand(len(m))<.08)&(m>0)
            if (m>0).sum()-drop.sum()>=4:m[drop]=0
        n=len(x);xs.append(np.pad(x,((0,32-n),(0,0))));gs.append(np.pad(g,((0,32-n),(0,0))));ms.append(np.pad(m,(0,32-n)))
    return [torch.from_numpy(np.stack(v)).cuda() for v in (xs,gs,ms)]

model=TemporalEncoder().cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
positive=[r for r in pairs if r['partition']=='fit' and r['label']==1];negative=[r for r in pairs if r['partition']=='fit' and r['label']==0]
training_log=[]
for epoch in range(30):
    model.train();losses=[]
    # Equal classes; only simultaneous identities form negative pairs.
    rounds=max(8,int(np.ceil(max(len(positive),len(negative))/24)))
    for _ in range(rounds):
        batch=[positive[i] for i in np.random.randint(len(positive),size=24)]+[negative[i] for i in np.random.randint(len(negative),size=24)]
        a=batch_view(batch,'a',True);b=batch_view(batch,'b',True);za=model(*a);zb=model(*b);sim=(za*zb).sum(-1);y=torch.tensor([r['label'] for r in batch],device='cuda',dtype=torch.float32)
        ba=F.normalize((a[0]*a[2][:,:,None]).sum(1),dim=1);bb=F.normalize((b[0]*b[2][:,:,None]).sum(1),dim=1);bs=(ba*bb).sum(1)
        loss=F.binary_cross_entropy_with_logits(12*(sim-.65),y)+.15*((sim-bs)**2).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
    row={'epoch':epoch+1,'loss':float(np.mean(losses)),'elapsed_seconds':round(time.time()-started,1)};training_log.append(row);print(json.dumps(row),flush=True)
torch.save({'model':model.state_dict(),'architecture':'COPE frozen frame encoder + masked residual multiscale TCN d1/d2/d4','seed':1109,'epochs':30},HERE/'temporal_encoder.pt');model.eval()

# Embed the whole clip and its two ordered halves in both temporal directions.
views=[];owners=[]
for i,s in enumerate(sequences):
    n=len(s['frames']);spans=[(0,n)]
    if min(M[i,:n//2].sum(),M[i,n//2:n].sum())>=4:spans.extend([(0,n//2),(n//2,n)])
    for lo,hi in spans:
        views.append({'a':i,'a_span':[lo,hi]});owners.append(i)
embeddings=[]
with torch.inference_mode():
    for off in range(0,len(views),64):
        b=batch_view(views[off:off+64],'a');forward=model(*b)
        xr=[];gr=[];mr=[]
        for r in views[off:off+64]:
            i=r['a'];lo,hi=r['a_span'];n=hi-lo;x=X[i,lo:hi][::-1].copy();g=G[i,lo:hi][::-1].copy();g[:,2:4]*=-1;g[:,5]=g[0,5]-g[:,5];m=M[i,lo:hi][::-1].copy()
            xr.append(np.pad(x,((0,32-n),(0,0))));gr.append(np.pad(g,((0,32-n),(0,0))));mr.append(np.pad(m,(0,32-n)))
        reverse=model(*[torch.from_numpy(np.stack(v)).cuda() for v in (xr,gr,mr)])
        embeddings.append(torch.stack([forward,reverse],1).cpu().numpy())
np.savez_compressed(HERE/'sequence_features.npz',appearance=X,geometry=G,mask=M,colors=colors,times=times,boxes=boxes,temporal=np.concatenate(embeddings).astype(np.float16),owners=np.asarray(owners),view_spans=np.asarray([r['a_span'] for r in views]))
(HERE/'sequences.json').write_text(json.dumps(sequences));(HERE/'results/temporal_training.json').write_text(json.dumps({'architecture':'masked residual multi-scale temporal convolution','dilations':[1,2,4],'input':'ordered COPE RGB descriptors + adjacent differences + normalized local movement + PTS time differences','training_counts':counts,'epochs':training_log,'historical_records_used':False,'whole_store_identity_labels_used':False,'independent_visit_accuracy_measured':False},indent=2))
print(json.dumps({'stage':'temporal_training_complete','sequences':len(sequences),'views':len(views),'elapsed_seconds':time.time()-started}),flush=True)
