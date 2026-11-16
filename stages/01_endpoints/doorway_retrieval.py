import os
"""Doorway-only retrieval; candidate provenance is not an identity conclusion."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

root=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));old=root/'rebuild_20261107';new=Path(__file__).resolve().parent;started=time.time()
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.45)
while not (new/'logs/features_status.json').exists() or not (new/'events_states.json').exists():
    if time.time()-started>3800:raise TimeoutError('Doorway inputs not ready')
    time.sleep(3)
events=json.loads((new/'events_states.json').read_text());reference=json.loads((new/'reference_mapping.json').read_text())
fusion=json.loads((old/'logs/fusion_selection.json').read_text());base=fusion['anchor_base']
line=np.asarray(json.loads((root/'scene.json').read_text())['boundary_pixels']);N=len(events);K=8
heldout={v['event_id'] for r in reference for v in r['endpoints'].values() if v['event_id'] is not None}
locked={v['event_id'] for r in reference if r['included_by_30s_rule'] for v in r['endpoints'].values() if v['event_id'] is not None}
newc=[];news=[];metadata=[];by_event=defaultdict(list)
for clip in range(106):
    records=json.loads((new/'sequences'/f'clip_{clip:03d}.json').read_text())['crops'];f=np.load(new/'features'/f'clip_{clip:03d}.npz')
    assert [r['path'] for r in records]==f['keys'].tolist()
    offset=len(metadata)
    for j,r in enumerate(records):
        r['index']=offset+j;by_event[r['event_id']].append(r);metadata.append(r)
    newc.append(f['cope']);news.append(f['sapiens'])
newc=np.concatenate(newc);news=np.concatenate(news)

def select(records,limit=K):
    if not records:return []
    # Retain both sides and time halves before filling the remaining quality slots.
    center=np.median([r['time'] for r in records]);chosen=[]
    groups=defaultdict(list)
    for r in records:
        d=r.get('distance');side='unknown' if d is None else 'inside' if d<0 else 'outside'
        groups[(side,r['time']>=center)].append(r)
    for group in groups.values():chosen.append(max(group,key=lambda r:r['quality']))
    for orientation in ('front','side'):
        group=[r for r in records if r.get('orientation')==orientation and r not in chosen]
        if group and len(chosen)<limit:chosen.append(max(group,key=lambda r:r['quality']))
    for r in sorted(records,key=lambda r:r['quality'],reverse=True):
        if len(chosen)>=limit:break
        if r not in chosen:chosen.append(r)
    return sorted(chosen[:limit],key=lambda r:r['time'])

units=[];crows=[];srows=[];sides=[];views=[];valid=[];svalid=[]
def add_unit(info,records,cf,sf,smap=None):
    records=select(records);c=np.zeros((K,1280),np.float16);s=np.zeros((K,4096),np.float16)
    side=np.zeros(K,np.int8);view=np.zeros(K,np.int8);v=np.zeros(K,bool);sv=np.zeros(K,bool)
    for j,r in enumerate(records):
        i=r['index'];c[j]=cf[i];v[j]=True
        si=i if smap is None else smap.get(i)
        if si is not None:s[j]=sf[si];sv[j]=True
        d=r.get('distance');side[j]=0 if d is None else (-1 if d<0 else 1)
        view[j]={'front':1,'side':2}.get(r.get('orientation'),0)
    units.append({**info,'samples':[{k:r[k] for k in ('path','time','box','quality','distance','orientation','row','track','clip') if k in r} for r in records],
                  'sample_count':int(v.sum()),'sapiens_count':int(sv.sum())})
    crows.append(c);srows.append(s);sides.append(side);views.append(view);valid.append(v);svalid.append(sv)

for e in events:
    records=[r for r in by_event[e['event_id']] if abs(r['time']-e['original_time'])<=1.15]
    add_unit({k:e[k] for k in ('event_id','time','direction','source_category','node','path','eligible_crossing')},records,newc,news)
bank=old/'work/bank';cached=json.loads((bank/'samples.json').read_text());cf=np.load(bank/'features_adapted.npz')['features']
sfdata=np.load(bank/'sapiens_aligned.npz');sf=sfdata['features'];smap={int(v):i for i,v in enumerate(sfdata['sample_indices'])}
node_events=defaultdict(list)
for e in events:
    if e['node']>=0:node_events[e['node']].append(e['time'])
groups=defaultdict(list)
for i,r in enumerate(cached):
    b=r['box'];d=b[3]-np.interp((b[0]+b[2])/2,line[:,0],line[:,1])
    if abs(d)>170:continue
    if any(abs(r['time']-t)<=2.5 for t in node_events[r['node']]):continue
    key=('node',r['node']) if r['node']>=0 else ('raw_track',r['clip'],r['track']) if r['track']>=0 else ('single_detection',i)
    groups[key].append({**r,'index':i,'distance':float(d),'orientation':'unknown'})
loose_count=0
for group_key,records in groups.items():
    node=records[0]['node']
    records.sort(key=lambda r:r['time']);segments=[];current=[]
    for r in records:
        jump=False
        if current:
            a=np.asarray(r['box']);b=np.asarray(current[-1]['box']);jump=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)>1.25*min(a[3]-a[1],b[3]-b[1])
        if current and (r['time']-current[-1]['time']>1.5 or r['time']-current[0]['time']>8 or jump):segments.append(current);current=[]
        current.append(r)
    if current:segments.append(current)
    for segment in segments:
        add_unit({'event_id':None,'time':float(np.median([r['time'] for r in segment])),
                  'start':segment[0]['time'],'end':segment[-1]['time'],'source_category':'loose_doorway_detection',
                  'node':node,'group_key':list(group_key),'direction':'unconfirmed','eligible_crossing':False,'cache_root':str(old)},segment,cf,sf,smap)
        loose_count+=1
del cf,sf,cached,groups,newc,news
C=torch.as_tensor(np.asarray(crows),device='cuda');S=torch.as_tensor(np.asarray(srows),device='cuda')
C=torch.nn.functional.normalize(C,dim=2);S=torch.nn.functional.normalize(S,dim=2)
del crows,srows
valid=np.asarray(valid);svalid=np.asarray(svalid);side=np.asarray(sides);view=np.asarray(views);M=len(units)

def stats(mat,va,vb):
    mask=va[:,:,None]&vb[:,None,:];m=np.where(mask,mat,-2);a=m.max(2);b=m.max(1)
    na=va.sum(1).clip(1);nb=vb.sum(1).clip(1);ra=np.sort(np.where(va,a,np.inf),axis=1);rb=np.sort(np.where(vb,b,np.inf),axis=1);ix=np.arange(len(mat))
    def quantile(x,n,q):
        at=(n-1)*q;l=at.astype(int);h=np.ceil(at).astype(int);return x[ix,l]+np.where(h==l,0,(x[ix,h]-x[ix,l])*(at-l))
    mean_a=np.where(va,a,0).sum(1)/na;mean_b=np.where(vb,b,0).sum(1)/nb
    ta=(ra[ix,na-1]+ra[ix,np.maximum(0,na-2)])/2;tb=(rb[ix,nb-1]+rb[ix,np.maximum(0,nb-2)])/2
    out=np.column_stack([np.minimum(mean_a,mean_b),np.maximum(mean_a,mean_b),np.minimum(quantile(ra,na,.5),quantile(rb,nb,.5)),
        np.minimum(quantile(ra,na,.25),quantile(rb,nb,.25)),np.minimum(ta,tb),m.max((1,2)),np.where(mask,mat,0).sum((1,2))/(na*nb)])
    return np.nan_to_num(out,nan=0,posinf=0,neginf=0)

scores=np.full((N,M),-1e4,np.float32);same_side=np.full((N,M),np.nan,np.float32);same_view=np.full((N,M),np.nan,np.float32)
for lo in range(0,N,12):
    n=min(12,N-lo)
    with torch.inference_mode():
        cm=(C[lo:lo+n].reshape(-1,1280)@C.reshape(-1,1280).T).reshape(n,K,M,K).permute(0,2,1,3).float().cpu().numpy().reshape(-1,K,K)
        sm=(S[lo:lo+n].reshape(-1,4096)@S.reshape(-1,4096).T).reshape(n,K,M,K).permute(0,2,1,3).float().cpu().numpy().reshape(-1,K,K)
    va=np.repeat(valid[lo:lo+n],M,axis=0);vb=np.tile(valid,(n,1));sa=np.repeat(svalid[lo:lo+n],M,axis=0);sb=np.tile(svalid,(n,1))
    cs=stats(cm,va,vb);ss=stats(sm,sa,sb);both=np.column_stack([cs,ss]);fused=((both-fusion['mean'])/fusion['scale'])@np.asarray(fusion['coefficient'])+fusion['intercept']
    raw=((cs-base['mean'])/base['scale'])@np.asarray(base['coefficient'])+base['intercept']
    covered=(sa.sum(1)>0)&(sb.sum(1)>0);raw[covered]=fused[covered]
    raw[(va.sum(1)==0)|(vb.sum(1)==0)]=-1e4;scores[lo:lo+n]=raw.reshape(n,M)
    # Explicit evidence only, not an uncalibrated bonus added to the ranking score.
    for values,out in [(side,same_side),(view,same_view)]:
        a=np.repeat(values[lo:lo+n],M,axis=0);b=np.tile(values,(n,1));mask=(a[:,:,None]==b[:,None,:])&(a[:,:,None]!=0)&va[:,:,None]&vb[:,None,:]
        sim=np.where(sa[:,:,None]&sb[:,None,:],(cm+sm)/2,cm);best=np.where(mask,sim,-2).max((1,2));best[~mask.any((1,2))]=np.nan
        out[lo:lo+n]=best.reshape(n,M)
    if lo%120==0:print(json.dumps({'stage':'doorway_retrieval_scores','events':lo+n,'gallery':M,'elapsed_seconds':round(time.time()-started,1)}),flush=True)
np.savez_compressed(new/'results/retrieval_scores.npz',scores=scores,same_side=same_side,same_view=same_view)
(new/'results/retrieval_units.json').write_text(json.dumps(units))

# Gallery eligibility here is broad and exploratory. Unverified candidates remain explicit.
times=np.asarray([u['time'] for u in units]);candidate=np.asarray([u['source_category'] in ('unverified_event','loose_doorway_detection') or u['direction']=='entry' for u in units])
exit_ids=np.asarray([e['event_id'] for e in events if e['direction']=='exit' or e['source_category']=='observed_unmatched_exit'])
legal=(times[None,:]<np.asarray([events[i]['time'] for i in exit_ids])[:,None])&candidate[None,:]
for qi,eid in enumerate(exit_ids):
    e=events[eid]
    for j,u in enumerate(units):
        if u['source_category']=='loose_doorway_detection' and (u['end']>=e['time']-2.5 or (u['node']==e['node'] and u['end']>=e['time']-8)):
            legal[qi,j]=False
ranked=np.where(legal,scores[exit_ids],-1e4)
order=np.argsort(-ranked,axis=1);reverse=np.argsort(-ranked,axis=0);rrank=np.empty_like(reverse);np.put_along_axis(rrank,reverse,np.arange(len(exit_ids))[:,None]+1,axis=0)
report=[];head=Counter();coverage=Counter();requests=[]
for qi,eid in enumerate(exit_ids):
    if events[eid]['source_category']!='observed_unmatched_exit':continue
    candidates=[]
    for rank,j in enumerate(order[qi,:20],1):
        if ranked[qi,j]<=-9999:continue
        u=units[int(j)];sid=u['event_id'];reverse_rank=int(rrank[qi,j])
        item={'gallery_id':int(j),'event_id':sid,'source_category':u['source_category'],'time':u['time'],'direction':u['direction'],
            'eligible_crossing':u['eligible_crossing'],'appearance_score':float(ranked[qi,j]),'exit_to_entrance_rank':rank,'entrance_to_exit_rank':reverse_rank,
            'mutual_top20':reverse_rank<=20,'same_side_evidence':float(same_side[eid,j]) if np.isfinite(same_side[eid,j]) else None,
            'same_orientation_evidence':float(same_view[eid,j]) if np.isfinite(same_view[eid,j]) else None,'historical_locked':sid in locked}
        candidates.append(item)
        if rank<=3 and u['source_category']=='loose_doorway_detection' and reverse_rank<=20:
            requests.append({'exit_event':int(eid),'gallery_id':int(j),'node':u['node'],'start':u['start'],'end':u['end'],'score':float(ranked[qi,j]),'samples':u['samples']})
    head[candidates[0]['source_category'] if candidates else 'no_earlier_searchable_candidate']+=1
    coverage.update(set(c['source_category'] for c in candidates))
    report.append({'exit_event':int(eid),'exit_time':events[eid]['time'],'candidates':candidates,'identity_status':'unresolved_retrieval_hypotheses'})
summary={'queried_original_exits':len(report),'event_sequences':N,'loose_doorway_episodes':loose_count,'gallery_units':M,
         'top1_candidate_source':dict(head),'top20_source_coverage_exit_count':dict(coverage),'targeted_redetection_requests':len(requests),
         'ranking':'frozen_COPE_Sapiens_fusion_with_explicit_mutual_ranks_not_full_Zhong_reranking',
         'candidate_similarity_threshold':None,'duration_penalty':None,'before_recording_start_claims':0,
         'results_are_candidate_provenance_not_confirmed_identity':True,'elapsed_seconds':round(time.time()-started,1)}
(new/'results/exit_300_retrieval.json').write_text(json.dumps(report,ensure_ascii=False))
(new/'results/redetection_requests.json').write_text(json.dumps(requests))
(new/'results/retrieval_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
