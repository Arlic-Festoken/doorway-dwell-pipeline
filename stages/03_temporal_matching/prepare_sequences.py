import os
"""Identity-local, ordered doorway clips. No visit labels or in-store stitching."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'))
HERE=Path(__file__).resolve().parent
BANK=ROOT/'rebuild_20261107/work/bank'
REC=ROOT/'doorway_recovery_20261108'
started=time.time()
for name in ('results','logs','features','crops','videos','variants'):(HERE/name).mkdir(exist_ok=True)
events=json.loads((REC/'events_recovered.json').read_text())
packs=json.loads((REC/'results/evidence_manifest.json').read_text())['events']
nodes=json.loads((BANK/'nodes.json').read_text())
z=np.load(BANK/'observations.npz');ot=z['time'];ob=z['boxes'];oc=z['clip'];on=z['node']
bc=np.load(BANK/'features_adapted.npz');bt=bc['time'];bq=bc['quality'];bo=bc['overlap'];br=bc['reliability'];cf=bc['features'].astype(np.float32)
cf/=np.linalg.norm(cf,axis=1,keepdims=True).clip(1e-8)
order=np.argsort(ot);ordered_times=ot[order]
clocks=[json.loads((ROOT/'work/clocks'/f'VID{i:05d}.json').read_text())['offset_seconds'] for i in range(106)]
pts=[np.asarray([float(r['best_effort_timestamp_time']) for r in json.loads((ROOT/'work/native_crossings'/f'{i:03d}_pts.json').read_text())['frames']]) for i in range(106)]
boundary=np.asarray([[0,829],[450,807],[860,773],[1210,731],[1450,696],[1660,659],[1919,620]])

def ious(a,b):
    b=np.asarray(b);a=np.asarray(a);inter=np.maximum(0,np.minimum(a[2:4],b[:,2:4])-np.maximum(a[:2],b[:,:2])).prod(1)
    return inter/((a[2:4]-a[:2]).prod()+(b[:,2:4]-b[:,:2]).prod(1)-inter).clip(1)

def matched_node(at,box):
    lo,hi=np.searchsorted(ordered_times,[at-.075,at+.075]);rows=order[lo:hi]
    if not len(rows):return None
    v=ious(box,ob[rows]);best=int(np.argmax(v));r=int(rows[best]);different=ious(ob[r],ob[rows])<.65
    if v[best]<.5 or (different.any() and v[best]-v[different].max()<.15) or on[r]<0:return None
    return int(on[r])

def segments(node):
    rows=np.asarray(nodes[node]['rows']);rows=rows[np.argsort(ot[rows])]
    # Multiple detector views at the same time are not additional video frames.
    _,unique=np.unique(ot[rows],return_index=True);rows=rows[unique]
    b=ob[rows,:4];t=ot[rows];h=b[:,3]-b[:,1];foot=np.c_[(b[:,0]+b[:,2])/2,b[:,3]]
    inter=np.maximum(0,np.minimum(b[1:,2:],b[:-1,2:])-np.maximum(b[1:,:2],b[:-1,:2])).prod(1);area=(b[:,2:]-b[:,:2]).prod(1)
    breaks=(np.diff(t)>.25)|(inter/(area[1:]+area[:-1]-inter).clip(1)<=.5)|(np.linalg.norm(np.diff(foot,axis=0),axis=1)>.35*np.minimum(h[1:],h[:-1]))
    # A low-occlusion appearance discontinuity splits a clip; its endpoints need
    # not resemble each other, so a gradual back-to-front turn remains intact.
    samples=nodes[node]['samples'];samples=sorted([i for i in samples if bq[i]>=.2 and bo[i]<=.3 and br[i]>=.7],key=lambda i:bt[i])
    for a,c in zip(samples[:-1],samples[1:]):
        if 0<bt[c]-bt[a]<=.55 and cf[a]@cf[c]<.48:
            j=int(np.searchsorted(t,(bt[a]+bt[c])/2))-1
            if 0<=j<len(breaks):breaks[j]=True
    return [r for r in np.split(rows,np.flatnonzero(breaks)+1) if len(r)>=4]

cache={}
def get_segments(node):
    if node not in cache:cache[node]=segments(node)
    return cache[node]

frames={};visibility_cache={}
def visible(row):
    row=int(row)
    if row not in visibility_cache:
        box=ob[row,:4];h=box[3]-box[1]
        lo,hi=np.searchsorted(ordered_times,[ot[row]-.025,ot[row]+.025]);others=order[lo:hi]
        overlaps=ious(box,ob[others]);other=(on[others]!=on[row])&(overlaps<.65)&(ob[others,4]>=.35)
        crowd=float(overlaps[other].max()) if other.any() else 0.
        visibility_cache[row]=(h>=100 and h/max(box[2]-box[0],1)>=1.2 and crowd<=.3,crowd)
    return visibility_cache[row]

def add_sequence(rows,node,kind,eid=None,fragment=0):
    # Fixed elapsed-time sampling, preserving frame order and missing intervals.
    rows=np.asarray(rows);selected=[]
    for row in rows:
        if selected and ot[row]-ot[selected[-1]]<.195:continue
        selected.append(int(row))
    if len(selected)>32:selected=selected[:32]
    valid=[r for r in selected if visible(r)[0]]
    if len(valid)<4 or ot[valid[-1]]-ot[valid[0]]<.6:return None
    sid=f'e{eid:04d}_f{fragment}' if kind=='event' else f't{node:05d}_{int(ot[rows[0]]*10)}'
    records=[]
    for r in selected:
        ok,crowd=visible(r);clip=int(oc[r]);fi=int(np.argmin(abs(pts[clip]+clocks[clip]-ot[r])))
        key=f'r{r}';t=float(pts[clip][fi]+clocks[clip])
        frames[key]={'key':key,'row':r,'node':node,'clip':clip,'native_frame_index':fi,'time':t,'box':ob[r,:4].tolist(),'visible':bool(ok),'overlap':crowd,'path':f'crops/{key}.jpg'}
        records.append(key)
    return {'id':sid,'kind':kind,'event_id':eid,'fragment':fragment,'node':node,'raw_track':nodes[node]['raw_track'],'frames':records,'valid_frames':len(valid),'time_interval':[frames[records[0]]['time'],frames[records[-1]]['time']]}

sequences=[];event_status=[]
for pack in packs:
    eid=pack['event_id'];e=events[eid];at=e['time'];added=[]
    if e.get('crossgap_nodes'):
        sources=[(n,e['interval'][f],f) for f,n in enumerate(e['crossgap_nodes'])]
    else:
        options=[]
        for frame in pack['frames']:
            n=matched_node(frame['time'],frame['box'])
            if n is not None:options.append((n,frame['time']))
        n=matched_node(at,e['crossing_box'])
        if n is not None:options.append((n,at))
        ranked=[]
        for n,anchor in set(options):
            for segment in get_segments(n):
                if np.min(abs(ot[segment]-anchor))>.15:continue
                rows=segment[abs(ot[segment]-at)<=3.2]
                if len(rows)<4:continue
                # Anchor association, then continuous pixels; no endpoint-wide
                # appearance threshold that would throw away a gradual turn.
                ranked.append((float(ot[rows[-1]]-ot[rows[0]])-.2*abs(anchor-at),n,anchor))
        sources=[(max(ranked)[1],max(ranked)[2],0)] if ranked else []
    for node,anchor,fragment in sources:
        candidates=get_segments(node)
        if not candidates:continue
        segment=min(candidates,key=lambda s:float(np.min(abs(ot[s]-anchor))))
        if np.min(abs(ot[segment]-anchor))>.2:continue
        rows=segment[abs(ot[segment]-anchor)<=3.2]
        s=add_sequence(rows,node,'event',eid,fragment)
        if s:sequences.append(s);added.append(s['id'])
    ready=len(added)==(2 if e.get('crossgap_nodes') else 1)
    event_status.append({'event_id':eid,'sequences':added,'temporal_ready':ready,'reason':'ready' if ready else 'insufficient_continuous_visible_frames'})
print(json.dumps({'stage':'event_sequences','events':len(packs),'usable':sum(s['temporal_ready'] for s in event_status),'sequences':len(sequences)}),flush=True)

# Training sources stay close to the doorway, cover all video times, and carry
# only local continuous-identity labels. Different track IDs are NOT negatives.
pool=[]
for n in nodes:
    if n['provisional'] or n['end']-n['start']<3.2:continue
    ids=np.asarray(n['samples'],int);good=ids[(bq[ids]>=.25)&(bo[ids]<=.3)&(br[ids]>=.7)]
    if len(good)<6:continue
    for seg in get_segments(n['id']):
        if ot[seg[-1]]-ot[seg[0]]<3.2:continue
        center=(ot[seg[0]]+ot[seg[-1]])/2;rows=seg[abs(ot[seg]-center)<=3.2]
        box=ob[rows,:4];foot=np.c_[(box[:,0]+box[:,2])/2,box[:,3]];dist=foot[:,1]-np.interp(foot[:,0],boundary[:,0],boundary[:,1])
        if np.mean(abs(dist)<=260)<.5:continue
        sg=good[(bt[good]>=ot[rows[0]])&(bt[good]<=ot[rows[-1]])]
        if len(sg)<6 or np.max(np.diff(np.sort(bt[sg])))>1.05:continue
        novelty=1-float(cf[sg[0]]@cf[sg[-1]])
        pool.append((int(center//20),novelty,n['id'],rows))
by_bin=defaultdict(list)
for r in pool:by_bin[r[0]].append(r)
chosen=[]
for bucket in by_bin.values():chosen.extend(sorted(bucket,key=lambda r:r[1],reverse=True)[:5])
chosen=sorted(chosen,key=lambda r:hashlib.sha256(str(r[2]).encode()).hexdigest())[:1000]
train_count=0
for _,novelty,node,rows in chosen:
    s=add_sequence(rows,node,'train')
    if s and s['valid_frames']>=12 and s['valid_frames']/len(s['frames'])>=.75:
        s['endpoint_appearance_change']=novelty;sequences.append(s);train_count+=1
used={k for s in sequences for k in s['frames']};frames={k:v for k,v in frames.items() if k in used}
(HERE/'sequences.json').write_text(json.dumps(sequences));(HERE/'frames.json').write_text(json.dumps(list(frames.values())))
(HERE/'events.json').write_text(json.dumps(events));(HERE/'results/event_sequence_status.json').write_text(json.dumps(event_status))
report={'active_endpoints':len(packs),'temporal_usable_endpoints':sum(s['temporal_ready'] for s in event_status),'event_sequences':sum(s['kind']=='event' for s in sequences),'training_sequences':train_count,'frames':len(frames),'nominal_sampling_hz':5,'maximum_sequence_seconds':6.4,'source_time':'native PTS + clock offset','historical_records_used':False,'elapsed_seconds':time.time()-started}
(HERE/'results/sequence_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
