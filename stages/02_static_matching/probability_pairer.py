import os
"""Clean-template probability assignment with explicit unmatched alternatives."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));OLD=ROOT/'doorway_balance_20261108';HERE=Path(__file__).resolve().parent;started=time.time()
parser=argparse.ArgumentParser();parser.add_argument('--scene-adapt',action='store_true');args=parser.parse_args()
events=json.loads((HERE/'events_recovered.json').read_text())
previous=json.loads((OLD/'results/delivery_records.json').read_text())
dispositions=json.loads((OLD/'results/event_dispositions.json').read_text())
evidence=json.loads((HERE/'results/evidence_manifest.json').read_text())
pack={r['event_id']:r for r in evidence['events']}
samples=json.loads((HERE/'results/clean_feature_samples.json').read_text())
z=np.load(HERE/'clean_features.npz');keys={str(k):i for i,k in enumerate(z['keys'])}
cf=z['cope'].astype(np.float32);sf=z['sapiens'].astype(np.float32)
cf/=np.linalg.norm(cf,axis=1,keepdims=True).clip(1e-8);sf/=np.linalg.norm(sf,axis=1,keepdims=True).clip(1e-8)
cv2.setNumThreads(2);part_cache={}
def color_parts(path):
    path=str(path)
    if path not in part_cache:
        image=cv2.imread(path)
        if image is None:raise ValueError('Missing person crop: '+path)
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB);h,w=lab.shape[:2];result=[]
        for low,high in ((.2,.48),(.55,.85)):
            pixels=lab[int(h*low):int(h*high),int(w*.2):max(int(w*.2)+1,int(w*.8))].reshape(-1,3)
            hist=np.concatenate([np.histogram(pixels[:,c],bins=16,range=(0,256))[0] for c in range(3)]).astype(np.float32);hist/=max(hist.sum(),1);result.append(np.sqrt(hist))
        part_cache[path]=np.stack(result)
    return part_cache[path]
source_paths={r['key']:r['original_crop'] for r in samples};parts=np.stack([color_parts(source_paths[str(key)]) for key in z['keys']])
by_event=defaultdict(list)
for r in samples:by_event[r['event_id']].append(r)
for rows in by_event.values():rows.sort(key=lambda r:r['time'])
template={eid:[keys[r['key']] for r in rows] for eid,rows in by_event.items() if len(rows)>=2}

def overlap(a,b):
    a=np.asarray(a);b=np.asarray(b);intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection+1e-8))

def separated(a,b):
    for x in pack[a]['frames']:
        for y in pack[b]['frames']:
            if abs(x['time']-y['time'])>.075:continue
            aa=np.asarray(x['box']);bb=np.asarray(y['box']);h=min(aa[3]-aa[1],bb[3]-bb[1])
            if overlap(aa,bb)<.01 and np.linalg.norm((aa[:2]+aa[2:]-bb[:2]-bb[2:])/2)>.5*h:return True
    return False

def compare_arrays(ca,cb,sa,sb,pa,pb):
    result=[]
    sa=sa[np.linalg.norm(sa,axis=1)>.1];sb=sb[np.linalg.norm(sb,axis=1)>.1]
    for fa,fb in ((ca,cb),(sa,sb)):
        if not len(fa) or not len(fb):result.extend([0.]*7);continue
        m=fa@fb.T;ra=np.sort(m.max(1));rb=np.sort(m.max(0))
        ranked=m.copy();matched=[]
        for _ in range(min(3,*m.shape)):
            i,j=np.unravel_index(np.argmax(ranked),ranked.shape);matched.append(float(ranked[i,j]));ranked[i,:]=-2;ranked[:,j]=-2
        result.extend([np.mean(matched[:2]),np.mean(matched),ra[-min(2,len(ra)):].mean(),rb[-min(2,len(rb)):].mean(),m.max(),np.quantile(m,.75),m.mean()])
    matrix=ca@cb.T;comparisons=[]
    for _ in range(min(2,*matrix.shape)):
        i,j=np.unravel_index(np.argmax(matrix),matrix.shape);comparisons.append(np.sum(pa[i]*pb[j],axis=1));matrix[i,:]=-2;matrix[:,j]=-2
    colors=np.stack(comparisons);means=colors.mean(0);result.extend([float(means[0]),float(means[1]),float(means.min()),float(colors.min(1).max())])
    return np.asarray(result,np.float64)

def vector(a,b):
    return compare_arrays(cf[a],cf[b],sf[a],sf[b],parts[a],parts[b])

# Calibration uses disjoint source endpoints. Old panel reviews are quarantined.
group={eid:f"node{events[eid]['node']}" if events[eid]['node']>=0 else f"event{events[eid].get('source_event',eid)}" for eid in template}
fold={eid:int(hashlib.sha256(str(group[eid]).encode()).hexdigest()[:8],16)%5 for eid in template}
training=[];calibration=[];sample_log=[]
for eid,rows in by_event.items():
    if pack[eid]['kind']=='two_fragment_interval':continue
    if eid not in template or len(rows)<4 or rows[-1]['time']-rows[0]['time']<.4:continue
    cut=len(rows)//2;a=[keys[r['key']] for r in rows[:cut]];b=[keys[r['key']] for r in rows[cut:]]
    target=training if fold[eid]<3 else calibration;target.append((vector(a,b),1));sample_log.append({'kind':'continuous_split','events':[eid],'partition':'fit' if target is training else 'calibration'})
ids=sorted(template)
for ai,a in enumerate(ids):
    for b in ids[ai+1:]:
        if pack[a]['kind']=='two_fragment_interval' or pack[b]['kind']=='two_fragment_interval':continue
        if abs(events[a]['time']-events[b]['time'])>3 or (fold[a]<3)!=(fold[b]<3) or not separated(a,b):continue
        target=training if fold[a]<3 else calibration;target.append((vector(template[a],template[b]),0));sample_log.append({'kind':'simultaneous_separated_bodies','events':[a,b],'partition':'fit' if target is training else 'calibration'})

# Expand supervision with simultaneous distractors and continuous multi-view
# fragments from the existing full-frame bank. No new detection or GPU work.
bank_dir=ROOT/'rebuild_20261107/work/bank';bc=np.load(bank_dir/'features_adapted.npz');bs=np.load(bank_dir/'sapiens_aligned.npz')
bank_cf=bc['features'].astype(np.float32);bank_sf=bs['features'].astype(np.float32)
bank_cf/=np.linalg.norm(bank_cf,axis=1,keepdims=True).clip(1e-8);bank_sf/=np.linalg.norm(bank_sf,axis=1,keepdims=True).clip(1e-8)
smap={int(v):i for i,v in enumerate(bs['sample_indices'])};bt=bc['time'];bn=bc['nodes'];bq=bc['quality'];bo=bc['overlap'];br=bc['reliability']
bank_meta=json.loads((bank_dir/'samples.json').read_text());boxes=np.asarray([r['box'] for r in bank_meta]);sorted_ids=np.argsort(bt);sorted_times=bt[sorted_ids]
by_node=defaultdict(list)
for i in smap:
    if bq[i]>=.3 and bo[i]<=.3 and br[i]>=.5:by_node[int(bn[i])].append(i)
def node_partition(node):return int(hashlib.sha256(f'node{node}'.encode()).hexdigest()[:8],16)%5<3
def bank_view(indices):return bank_cf[indices],bank_sf[[smap[i] for i in indices]],np.stack([color_parts(ROOT/'rebuild_20261107'/bank_meta[i]['path']) for i in indices])
additional_negative=0
for eid,own in template.items():
    if pack[eid]['kind']=='two_fragment_interval':continue
    target=training if fold[eid]<3 else calibration;seen=set()
    for source in by_event[eid][::max(1,len(by_event[eid])//3)]:
        at=source['time'];lo,hi=np.searchsorted(sorted_times,[at-.075,at+.075]);options=[]
        for bi in sorted_ids[lo:hi]:
            bi=int(bi);node=int(bn[bi])
            if node in seen or node==events[eid]['node'] or bi not in smap or bq[bi]<.3 or bo[bi]>.3 or br[bi]<.5 or node_partition(node)!=(fold[eid]<3):continue
            if overlap(source['box'],boxes[bi])>.01:continue
            h=min(source['box'][3]-source['box'][1],boxes[bi,3]-boxes[bi,1])
            if np.linalg.norm((np.asarray(source['box'][:2])+source['box'][2:]-boxes[bi,:2]-boxes[bi,2:])/2)<.5*h:continue
            ids_other=[i for i in by_node[node] if abs(bt[i]-at)<=.35]
            if ids_other:options.append((float(bank_cf[bi]@cf[own].mean(0)),node,ids_other))
        for similarity,node,other in sorted(options,reverse=True)[:3]:
            if node in seen:continue
            seen.add(node);c,s,p=bank_view(other);target.append((compare_arrays(cf[own],c,sf[own],s,parts[own],p),0));additional_negative+=1
            sample_log.append({'kind':'simultaneous_full_frame_distractor','events':[eid],'distractor_node':node,'time':at,'partition':'fit' if target is training else 'calibration'})

raw_obs=np.load(bank_dir/'observations.npz');raw_boxes=raw_obs['boxes'];raw_times=raw_obs['time'];nodes=json.loads((bank_dir/'nodes.json').read_text());additional_positive=0;seen_nodes=set()
for eid in template:
    if pack[eid]['kind']=='two_fragment_interval':continue
    node=events[eid]['node']
    if node<0 or node in seen_nodes:continue
    seen_nodes.add(node);at=events[eid]['time'];rows=np.asarray(nodes[node]['rows']);rows=rows[abs(raw_times[rows]-at)<=2.5];rows=rows[np.argsort(raw_times[rows])]
    if len(rows)<12:continue
    times=raw_times[rows];box=raw_boxes[rows,:4]
    if times[-1]-times[0]<1.5:continue
    intersection=np.maximum(0,np.minimum(box[1:,2:],box[:-1,2:])-np.maximum(box[1:,:2],box[:-1,:2])).prod(1);area=(box[:,2:]-box[:,:2]).prod(1)
    overlap_sequence=intersection/(area[1:]+area[:-1]-intersection).clip(1);feet=np.c_[(box[:,0]+box[:,2])/2,box[:,3]];height=box[:,3]-box[:,1]
    breaks=(np.diff(times)>.25)|(overlap_sequence<=.5)|(np.linalg.norm(np.diff(feet,axis=0),axis=1)>.35*np.minimum(height[:-1],height[1:]))
    segments=np.split(np.arange(len(rows)),np.flatnonzero(breaks)+1);segment=min(segments,key=lambda indices:float(np.min(abs(times[indices]-at))));times=times[segment]
    if times[-1]-times[0]<1.5:continue
    available=sorted([i for i in by_node[node] if times[0]<=bt[i]<=times[-1]],key=lambda i:bt[i])
    if len(available)<4 or bt[available[-1]]-bt[available[0]]<1.5:continue
    middle=(bt[available[0]]+bt[available[-1]])/2;a=[i for i in available if bt[i]<middle];b=[i for i in available if bt[i]>=middle]
    if min(len(a),len(b))<2:continue
    ca,sa,pa=bank_view(a);cb,sb,pb=bank_view(b);target=training if fold[eid]<3 else calibration;target.append((compare_arrays(ca,cb,sa,sb,pa,pb),1));additional_positive+=1
    sample_log.append({'kind':'continuous_multiview_bank_fragment','events':[eid],'node':node,'interval':[float(times[0]),float(times[-1])],'partition':'fit' if target is training else 'calibration'})
scene_positive=0;scene_negative=0
if args.scene_adapt:
    all_node_samples=defaultdict(list)
    for i,node in enumerate(bn):all_node_samples[int(node)].append(i)
    for node,available in by_node.items():
        if node in seen_nodes or len(available)<6 or scene_positive>=600:continue
        available=sorted(available,key=lambda i:bt[i]);rows=np.asarray(nodes[node]['rows']);rows=rows[np.argsort(raw_times[rows])]
        if len(rows)<20:continue
        times=raw_times[rows];box=raw_boxes[rows,:4];height=box[:,3]-box[:,1];feet=np.c_[(box[:,0]+box[:,2])/2,box[:,3]]
        intersection=np.maximum(0,np.minimum(box[1:,2:],box[:-1,2:])-np.maximum(box[1:,:2],box[:-1,:2])).prod(1);area=(box[:,2:]-box[:,:2]).prod(1)
        breaks=(np.diff(times)>.25)|(intersection/(area[1:]+area[:-1]-intersection).clip(1)<=.5)|(np.linalg.norm(np.diff(feet,axis=0),axis=1)>.35*np.minimum(height[:-1],height[1:]))
        best=None
        for segment in np.split(np.arange(len(rows)),np.flatnonzero(breaks)+1):
            low,high=times[segment[0]],times[segment[-1]]
            if high-low<2:continue
            candidates=[i for i in available if low<=bt[i]<=high]
            if len(candidates)<6:continue
            original=[i for i in all_node_samples[node] if low<=bt[i]<=high]
            if np.quantile(bo[original],.9)>.3:continue
            for ai,a in enumerate(candidates):
                for b in candidates[ai+1:]:
                    if not 1.5<=bt[b]-bt[a]<=12:continue
                    middle=[i for i in candidates if bt[a]<=bt[i]<=bt[b]]
                    if np.max(np.diff(bt[middle]))>1.5:continue
                    adj_c=np.sum(bank_cf[middle[1:]]*bank_cf[middle[:-1]],axis=1);adj_s=np.sum(bank_sf[[smap[i] for i in middle[1:]]]*bank_sf[[smap[i] for i in middle[:-1]]],axis=1)
                    if adj_c.min()<.55 or adj_s.min()<.25:continue
                    cc=float(bank_cf[a]@bank_cf[b]);ss=float(bank_sf[smap[a]]@bank_sf[smap[b]])
                    if not .4<=cc<=.8 or ss<.15:continue
                    novelty=1-.5*(cc+ss)
                    aa=[i for i in candidates if abs(bt[i]-bt[a])<=.75];bb=[i for i in candidates if abs(bt[i]-bt[b])<=.75]
                    if min(len(aa),len(bb))<2 or set(aa)&set(bb):continue
                    if best is None or novelty>best[0]:best=(novelty,aa,bb)
        if best is None:continue
        novelty,a,b=best;ca,sa,pa=bank_view(a);cb,sb,pb=bank_view(b);target=training if node_partition(node) else calibration
        target.append((compare_arrays(ca,cb,sa,sb,pa,pb),1));scene_positive+=1
        sample_log.append({'kind':'scene_continuous_view_change','node':node,'bank_samples':[a,b],'novelty':novelty,'partition':'fit' if target is training else 'calibration'})
        for anchor in (a[0],b[-1]):
            at=bt[anchor];lo,hi=np.searchsorted(sorted_times,[at-.075,at+.075]);other=[]
            for j in sorted_ids[lo:hi]:
                j=int(j);n=int(bn[j])
                if j not in smap or n==node or node_partition(n)!=node_partition(node) or bq[j]<.3 or bo[j]>.3 or br[j]<.5 or overlap(boxes[anchor],boxes[j])>.01:continue
                other.append((float(bank_cf[anchor]@bank_cf[j]),j))
            if not other:continue
            j=max(other)[1];cc,ss,pp=bank_view([j]);target.append((compare_arrays(bank_cf[[anchor]],cc,bank_sf[[smap[anchor]]],ss,np.stack([color_parts(ROOT/'rebuild_20261107'/bank_meta[anchor]['path'])]),pp),0));scene_negative+=1
            sample_log.append({'kind':'scene_simultaneous_hard_negative','nodes':[node,int(bn[j])],'bank_samples':[anchor,j],'partition':'fit' if target is training else 'calibration'})
    print(json.dumps({'stage':'scene_view_adaptation_samples','positives':scene_positive,'negatives':scene_negative}),flush=True)
del raw_obs,raw_boxes,raw_times,bank_cf,bank_sf,bank_meta,bc,bs
print(json.dumps({'stage':'expanded_supervision','additional_negatives':additional_negative,'additional_multiview_positives':additional_positive}),flush=True)
counts={name:dict(Counter(y for _,y in rows)) for name,rows in [('fit',training),('calibration',calibration)]}
print(json.dumps({'stage':'automatic_training_samples','counts':counts}),flush=True)
if any(min(Counter(y for _,y in rows).get(k,0) for k in (0,1))<8 for rows in (training,calibration)):
    raise ValueError(f'Not enough source-disjoint calibration examples: {counts}')
X=np.stack([x for x,y in training]);y=np.asarray([y for x,y in training]);full=np.any(abs(X[:,7:14])>1e-8,axis=1);scaler=StandardScaler().fit(X[full])
model=LogisticRegression(C=.1,class_weight='balanced',max_iter=400).fit(scaler.transform(X[full]),y[full])
cx=np.stack([x for x,y in calibration]);cy=np.asarray([y for x,y in calibration])
cfull=np.any(abs(cx[:,7:14])>1e-8,axis=1);platt=LogisticRegression(C=1,class_weight='balanced',max_iter=300).fit(model.decision_function(scaler.transform(cx[cfull]))[:,None],cy[cfull])
cope_columns=list(range(7))+list(range(14,18));cscaler=StandardScaler().fit(X[:,cope_columns]);cmodel=LogisticRegression(C=.1,class_weight='balanced',max_iter=400).fit(cscaler.transform(X[:,cope_columns]),y)
cplatt=LogisticRegression(C=1,class_weight='balanced',max_iter=300).fit(cmodel.decision_function(cscaler.transform(cx[:,cope_columns]))[:,None],cy)
def estimate(vectors):
    available=np.any(abs(vectors[:,7:14])>1e-8,axis=1);p=cplatt.predict_proba(cmodel.decision_function(cscaler.transform(vectors[:,cope_columns]))[:,None])[:,1]
    if available.any():p[available]=platt.predict_proba(model.decision_function(scaler.transform(vectors[available]))[:,None])[:,1]
    return p
(HERE/'results/probability_model.json').write_text(json.dumps({'feature':'comparable_multiview_COPE_Sapiens_plus_upper_lower_color','scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coefficient':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'platt_coefficient':platt.coef_.tolist(),'platt_intercept':platt.intercept_.tolist(),'cope_only':{'columns':cope_columns,'mean':cscaler.mean_.tolist(),'scale':cscaler.scale_.tolist(),'coefficient':cmodel.coef_.tolist(),'intercept':cmodel.intercept_.tolist(),'platt_coefficient':cplatt.coef_.tolist(),'platt_intercept':cplatt.intercept_.tolist()},'sample_counts':counts,'calibration_scope':'automatic_local_continuity_vs_simultaneous_distinct_bodies','real_visit_accuracy_evaluated':False,'old_panel_reviews_used':False,'historical_records_used':False},indent=2))
(HERE/'results/training_sample_provenance.json').write_text(json.dumps(sample_log))

active=sorted(pack);all_entries=[i for i in active if events[i]['direction']=='entry'];all_exits=[i for i in active if events[i]['direction']=='exit']
usable={i for i in template if pack[i]['ready_for_judge']}
entries=[i for i in all_entries if i in usable];exits=[i for i in all_exits if i in usable]
shape=(len(entries),len(exits));raw=np.full(shape,-1e4,np.float64);prob=np.zeros(shape,np.float64);legal=np.zeros(shape,bool);conflicts=[]
for ai,a in enumerate(entries):
    candidates=[];vectors=[]
    for bi,b in enumerate(exits):
        if events[b]['time']<=events[a]['time']:continue
        if separated(a,b):conflicts.append([a,b]);continue
        legal[ai,bi]=True;candidates.append(bi);vectors.append(vector(template[a],template[b]))
    if candidates:
        prob[ai,candidates]=estimate(np.stack(vectors));raw[ai,candidates]=logit(np.clip(prob[ai,candidates],1e-6,1-1e-6))
        if pack[a]['kind']=='two_fragment_interval':
            fragments=[[keys[r['key']] for r in by_event[a] if r['fragment']==k] for k in (0,1)]
            bridge_probability=float(estimate(vector(*fragments)[None,:])[0])
            for fragment in fragments:
                fragment_prob=estimate(np.stack([vector(fragment,template[exits[b]]) for b in candidates]))
                prob[ai,candidates]=np.minimum(prob[ai,candidates],fragment_prob)
            prob[ai,candidates]=np.minimum(prob[ai,candidates],bridge_probability)
order=np.argsort(-prob,axis=1);rank_a=np.empty_like(order);np.put_along_axis(rank_a,order,np.arange(len(exits))[None,:]+1,axis=1)
order=np.argsort(-prob,axis=0);rank_b=np.empty_like(order);np.put_along_axis(rank_b,order,np.arange(len(entries))[:,None]+1,axis=0)
margin_a=np.empty(shape);margin_b=np.empty(shape)
for a in range(len(entries)):
    best=np.sort(raw[a])[-2:]
    for b in range(len(exits)):margin_a[a,b]=raw[a,b]-(best[-2] if raw[a,b]==best[-1] else best[-1])
for b in range(len(exits)):
    best=np.sort(raw[:,b])[-2:]
    for a in range(len(entries)):margin_b[a,b]=raw[a,b]-(best[-2] if raw[a,b]==best[-1] else best[-1])
threshold=.9
supported=legal&(prob>=threshold)&((rank_a<=20)|(rank_b<=20))
# Sum of log odds relative to rejection; no cardinality bonus.
utility=np.where(supported,logit(np.clip(prob,1e-6,1-1e-6))-logit(threshold),-1e4)
objective=np.c_[utility,np.zeros((len(entries),len(entries)))];rr,cc=linear_sum_assignment(objective,maximize=True);optimum=float(objective[rr,cc].sum())
main=[];unstable=[]
for a,b in zip(rr,cc):
    if b<len(exits) and utility[a,b]>0:
        value=objective[a,b];objective[a,b]=-1e4;ar,ac=linear_sum_assignment(objective,maximize=True);gap=optimum-float(objective[ar,ac].sum());objective[a,b]=value
        row={'left':entries[a],'right':exits[b],'probability':float(prob[a,b]),'raw_score':float(raw[a,b]),'rank_forward':int(rank_a[a,b]),'rank_reverse':int(rank_b[a,b]),'margin_forward':float(margin_a[a,b]),'margin_reverse':float(margin_b[a,b]),'global_alternative_gap':gap,'method':'recovered_multiview_probability_assignment'}
        (main if gap>=.15 else unstable).append(row)

# Merge outside interruptions only when both clean local sequences share frames.
returns=[];seen_left=set();seen_right=set()
for link in dispositions['continuous_links']:
    if link['kind']!='short_return':continue
    a=int(dispositions['canonical'][link['left']]);b=int(dispositions['canonical'][link['right']])
    if a not in usable or b not in usable or a in seen_left or b in seen_right or not 0<events[b]['time']-events[a]['time']<=30:continue
    shared=sum(any(abs(x['time']-y['time'])<=.075 and overlap(x['box'],y['box'])>.7 for y in pack[b]['frames']) for x in pack[a]['frames'])
    if shared>=2 and not separated(a,b):
        returns.append({'left':a,'right':b,'method':'continuous_shared_clean_frames','shared_clean_frames':shared});seen_left.add(a);seen_right.add(b)
links=main+returns;succ={r['left']:r['right'] for r in links};pred={r['right']:r['left'] for r in links};meta={(r['left'],r['right']):r for r in links}
complete=[];short=[];open_chains=[];chains=[];unclosed=[]
for start in sorted(active,key=lambda i:events[i]['time']):
    if start in pred:continue
    chain=[start]
    while chain[-1] in succ:chain.append(succ[chain[-1]])
    first=events[chain[0]];last=events[chain[-1]];closed=first['direction']=='entry' and last['direction']=='exit'
    row={'entry_seconds':first['time'] if first['direction']=='entry' else None,'exit_seconds':last['time'] if last['direction']=='exit' else None,'entry_event':chain[0] if first['direction']=='entry' else None,'exit_event':chain[-1] if last['direction']=='exit' else None,'entry_interval':first['interval'] if first['direction']=='entry' else None,'exit_interval':last['interval'] if last['direction']=='exit' else None,'appearance':'','event_chain':chain,'source_pass':'automatic_recovered_probability_20261108','association_links':[meta[a,b] for a,b in zip(chain[:-1],chain[1:])],'outside_interruptions':[{'exit':events[a]['time'],'reentry':events[b]['time'],'gap_seconds':events[b]['time']-events[a]['time']} for a,b in zip(chain[:-1],chain[1:]) if events[a]['direction']=='exit']}
    row['duration_seconds']=row['exit_seconds']-row['entry_seconds'] if closed else None
    if closed:(complete if row['duration_seconds']>=30 else short).append(row)
    else:
        open_chains.append(row)
        for i in ([chain[0]] if first['direction']=='exit' else [])+([chain[-1]] if last['direction']=='entry' else []):
            e=events[i];unclosed.append({'event_id':i,'entry_seconds':e['time'] if e['direction']=='entry' else None,'exit_seconds':e['time'] if e['direction']=='exit' else None,'direction':e['direction'],'interval':e['interval'],'crossing_confirmed':True,'appearance':'','evidence_status':pack[i]['status'],'chain':chain})
    chains.append(row)
complete.sort(key=lambda r:r['entry_seconds'])
unverified=[r for r in previous['candidates'] if not r['crossing_confirmed']]
candidates=sorted(unclosed+unverified,key=lambda r:r['entry_seconds'] if r['entry_seconds'] is not None else r['exit_seconds'])
records={'mode':'automatic_doorway_recovery_20261108','confirmed':complete,'candidates':candidates,'unclosed':unclosed,'minimum_duration_seconds':30,'maximum_merged_outside_gap_seconds':30,'historical_records_used':False,'automatic_records_are_independently_confirmed':False,'probabilities_calibrated_on_independent_visit_labels':False}
summary={'complete_visits':len(complete),'unmatched_endpoints':len(unclosed),'unmatched_directions':dict(Counter(r['direction'] for r in unclosed)),'open_visit_chains':len(open_chains),'short_visits_excluded':len(short),'unverified_candidates_retained':len(unverified),'eligible_unique_entries':len(all_entries),'eligible_unique_exits':len(all_exits),'endpoints_with_ready_clean_features':len(usable),'store_visit_edges':len(main),'continuous_return_links':len(returns),'appearance_only_return_links':0,'geometry_conflicts_blocked':len(conflicts),'probability_threshold':threshold,'assignment_objective':'sum_log_odds_above_rejection_threshold_with_unmatched_options','requires_mutual_first_and_margin':False,'minimum_global_alternative_gap':.15,'ambiguous_global_edges_retained_unmatched':len(unstable),'historical_records_used':False,'old_panel_reviews_used_for_training':False,'accuracy_evaluation_performed':False,'calibration_scope':'automatic_local_sequence_samples_only','elapsed_seconds':round(time.time()-started,1)}
(HERE/'results/delivery_records.json').write_text(json.dumps(records,ensure_ascii=False));(HERE/'results/delivery_report.json').write_text(json.dumps(summary,indent=2))
(HERE/'results/event_dispositions.json').write_text(json.dumps({'main_edges':main,'return_edges':returns,'open_chains':open_chains,'chains':chains,'short_visits':short,'geometry_conflicts':conflicts,'unstable_edges':unstable}))
ranking=[]
for a,eid in enumerate(entries):
    for b in np.argsort(-prob[a])[:3]:
        if legal[a,b]:ranking.append({'entry_event':eid,'exit_event':exits[b],'probability':float(prob[a,b]),'rank_forward':int(rank_a[a,b]),'rank_reverse':int(rank_b[a,b]),'accepted':any(r['left']==eid and r['right']==exits[b] for r in main),'entry_panel':pack[eid]['panel'],'exit_panel':pack[exits[b]]['panel']})
(HERE/'results/top3_candidates.json').write_text(json.dumps(ranking));np.savez_compressed(HERE/'results/pair_probabilities.npz',entries=entries,exits=exits,probability=prob,raw_score=raw,legal=legal,rank_forward=rank_a,rank_reverse=rank_b)
print(json.dumps(summary),flush=True)
