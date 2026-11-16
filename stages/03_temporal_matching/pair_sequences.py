"""Compare ordered snippets, calibrate on local identities, assign full visits."""
from collections import Counter,defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REC=ROOT/'doorway_recovery_20261108';started=time.time()
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
record={'stage':'temporal_pair_assignment','pid':os.getpid(),'cwd':str(HERE),'gpu_ids':[0],'output':str(HERE/'variants'),'started_at':datetime.now().isoformat(),'stop_condition':'five assignments complete or 1800 seconds'}
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
seq=json.loads((HERE/'sequences.json').read_text());events=json.loads((HERE/'events.json').read_text());z=np.load(HERE/'sequence_features.npz')
X=z['appearance'];G=z['geometry'];M=z['mask'];C=z['colors'];T=z['times'];B=z['boxes'];Z=z['temporal'].astype(np.float32);owners=z['owners'];spans=z['view_spans']
old_pack={r['event_id']:r for r in json.loads((REC/'results/evidence_manifest.json').read_text())['events']};active=sorted(old_pack)
by_event=defaultdict(list)
for i,s in enumerate(seq):
    if s['kind']=='event' and M[i].sum()>=4:by_event[s['event_id']].append(i)
for eid,ids in list(by_event.items()):
    if len(ids)!=(2 if events[eid].get('crossgap_nodes') else 1):del by_event[eid]
bundle_cache={};bundles=[]
def bundle(i,span=None):
    span=tuple(span or (0,len(seq[i]['frames'])));key=(i,*span)
    if key in bundle_cache:return bundle_cache[key]
    lo,hi=span;valid=np.flatnonzero(M[i,lo:hi])+lo
    if len(valid)<2:raise ValueError(f'Insufficient sequence frames: {i}, {span}')
    ids=valid[np.rint(np.linspace(0,len(valid)-1,16)).astype(int)]
    views=np.flatnonzero((owners==i)&((spans[:,0]>=lo)&(spans[:,1]<=hi)))
    temporal=Z[views].reshape(-1,1280);temporal/=np.linalg.norm(temporal,axis=1,keepdims=True).clip(1e-8)
    k=len(bundles);bundle_cache[key]=k;bundles.append({'x':X[i,ids],'color':C[i,ids],'geometry':G[i,ids],'z':temporal,'source':i,'span':span});return k
pairs=json.loads((HERE/'results/temporal_sample_provenance.json').read_text())
cal=[r for r in pairs if r['partition']=='calibration']
cal_ids=[(bundle(r['a'],r['a_span']),bundle(r['b'],r['b_span'])) for r in cal]
event_bundle={eid:[bundle(i) for i in ids] for eid,ids in by_event.items()}
entry=[i for i in active if events[i]['direction']=='entry'];exits=[i for i in active if events[i]['direction']=='exit'];shape=(len(entry),len(exits))
old=np.load(REC/'variants/B_parts/pair_probabilities.npz');old_prob=np.zeros(shape);old_legal=np.zeros(shape,bool);entry_index={e:i for i,e in enumerate(entry)};exit_index={e:i for i,e in enumerate(exits)}
for ai,a in enumerate(old['entries']):
    for bi,b in enumerate(old['exits']):old_prob[entry_index[int(a)],exit_index[int(b)]]=old['probability'][ai,bi];old_legal[entry_index[int(a)],exit_index[int(b)]]=old['legal'][ai,bi]

def separated(a,b):
    for i in by_event.get(a,[]):
        for j in by_event.get(b,[]):
            va=np.flatnonzero(M[i]);vb=np.flatnonzero(M[j])
            for ia in va:
                jb=vb[np.argmin(abs(T[j,vb]-T[i,ia]))]
                if abs(T[j,jb]-T[i,ia])>.075:continue
                aa=B[i,ia];bb=B[j,jb];inter=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])).prod();iou=inter/((aa[2:]-aa[:2]).prod()+(bb[2:]-bb[:2]).prod()-inter)
                if iou<.01 and np.linalg.norm((aa[:2]+aa[2:]-bb[:2]-bb[2:])/2)>.5*min(aa[3]-aa[1],bb[3]-bb[1]):return True
    for x in old_pack[a]['frames']:
        for y in old_pack[b]['frames']:
            if abs(x['time']-y['time'])>.075:continue
            aa=np.asarray(x['box']);bb=np.asarray(y['box']);inter=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])).prod();iou=inter/((aa[2:]-aa[:2]).prod()+(bb[2:]-bb[:2]).prod()-inter)
            if iou<.01 and np.linalg.norm((aa[:2]+aa[2:]-bb[:2]-bb[2:])/2)>.5*min(aa[3]-aa[1],bb[3]-bb[1]):return True
    return False

legal=np.zeros(shape,bool);requests=[];request_owner=[]
for ai,a in enumerate(entry):
    for bi,b in enumerate(exits):
        if events[b]['time']<=events[a]['time'] or separated(a,b):continue
        legal[ai,bi]=True
        if a not in event_bundle or b not in event_bundle:continue
        for av in event_bundle[a]:
            for bv in event_bundle[b]:requests.append((av,bv));request_owner.append((ai,bi))
print(json.dumps({'stage':'legal_pair_queries','entry':len(entry),'exit':len(exits),'ordered_sequence_queries':len(requests),'calibration_examples':len(cal)}),flush=True)
xx=torch.tensor(np.stack([b['x'] for b in bundles]),device='cuda');cc=np.stack([b['color'] for b in bundles]);gg=np.stack([b['geometry'] for b in bundles]);zz=np.zeros((len(bundles),6,1280),np.float32);zm=np.zeros((len(bundles),6),bool)
for i,b in enumerate(bundles):zz[i,:len(b['z'])]=b['z'];zm[i,:len(b['z'])]=True
zt=torch.tensor(zz,device='cuda')

def dtw(sim):
    # Monotone path, with explicit costs for repeated frames. A whole-clip
    # reverse is allowed; arbitrary frame permutations are not.
    n=len(sim);cost=np.full((n,17,17),1e6,np.float32);length=np.zeros((n,17,17),np.int16);cost[:,0,0]=0
    for i in range(1,17):
        for j in range(1,17):
            options=np.stack([cost[:,i-1,j-1],cost[:,i-1,j]+.035,cost[:,i,j-1]+.035],1);choice=options.argmin(1)
            cost[:,i,j]=options[np.arange(n),choice]+1-sim[:,i-1,j-1]
            lengths=np.stack([length[:,i-1,j-1],length[:,i-1,j],length[:,i,j-1]],1);length[:,i,j]=lengths[np.arange(n),choice]+1
    return 1-cost[:,16,16]/length[:,16,16]

def vectors(ids):
    result=[]
    for offset in range(0,len(ids),768):
        batch=np.asarray(ids[offset:offset+768]);a=batch[:,0];b=batch[:,1]
        with torch.inference_mode():sim=torch.bmm(xx[a],xx[b].transpose(1,2)).cpu().numpy();tz=torch.bmm(zt[a],zt[b].transpose(1,2)).cpu().numpy()
        allowed=zm[a,:,None]&zm[b,None,:];tz=np.where(allowed,tz,-2);tsort=np.sort(tz.reshape(len(a),-1),axis=1)
        upper=(cc[a,:,0]*cc[b,:,0]).sum(-1).mean(1);lower=(cc[a,:,1]*cc[b,:,1]).sum(-1).mean(1)
        # Colors use nearest appearance-compatible views, not matching frame
        # numbers, because entry and exit motion can have opposite direction.
        match=sim.argmax(2);paired=cc[b[:,None],match];co=(cc[a]*paired).sum(-1).mean(1);upper=co[:,0];lower=co[:,1]
        aspect=np.abs(gg[a,:,0].mean(1)-gg[b,:,0].mean(1));forward=dtw(sim);reverse=dtw(sim[:,:,::-1]);both=np.maximum(forward,reverse)
        result.append(np.c_[sim.mean((1,2)),sim.max((1,2)),np.quantile(sim,.75,axis=(1,2)),sim.max(2).mean(1),sim.max(1).mean(1),upper,lower,aspect,
                            tz[:,0,0],tsort[:,-1],tsort[:,-2:].mean(1),tz[:,1,1],forward,reverse,both,both-sim.mean((1,2))])
    return np.concatenate(result)
cx=vectors(cal_ids);cy=np.array([r['label'] for r in cal]);columns={'F_dense_static':list(range(8)),'G_temporal':[5,6,7,8,9,10,11],'H_alignment':list(range(16))}
models={};model_meta={}
for name,cols in columns.items():
    scaler=StandardScaler().fit(cx[:,cols]);model=LogisticRegression(C=.1,class_weight='balanced',max_iter=500).fit(scaler.transform(cx[:,cols]),cy);models[name]=(scaler,model,cols)
    model_meta[name]={'columns':cols,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficient':model.coef_.tolist(),'intercept':model.intercept_.tolist()}
scores=vectors(requests);probs={name:old_prob.copy() for name in columns};has_temporal=np.zeros(shape,bool)
for name,(scaler,model,cols) in models.items():
    p=model.predict_proba(scaler.transform(scores[:,cols]))[:,1];groups=defaultdict(list)
    for (ai,bi),value in zip(request_owner,p):groups[ai,bi].append(float(value))
    for (ai,bi),values in groups.items():probs[name][ai,bi]=min(values);has_temporal[ai,bi]=True
    # A two-fragment inferred entrance must itself support one identity.
    for eid,bs in event_bundle.items():
        if len(bs)!=2:continue
        bridge=vectors([(bs[0],bs[1])]);bridge_p=float(model.predict_proba(scaler.transform(bridge[:,cols]))[0,1])
        if eid in entry_index:probs[name][entry_index[eid],has_temporal[entry_index[eid]]]=np.minimum(probs[name][entry_index[eid],has_temporal[entry_index[eid]]],bridge_p)
probs['I_fusion']=probs['H_alignment'].copy();shared=has_temporal&old_legal
probs['I_fusion'][shared]=expit(.65*logit(np.clip(probs['H_alignment'][shared],1e-6,1-1e-6))+.35*logit(np.clip(old_prob[shared],1e-6,1-1e-6)))
probs['J_coverage']=probs['I_fusion'].copy()
usable=has_temporal|old_legal;legal&=usable
(HERE/'results/pair_model.json').write_text(json.dumps({'models':model_meta,'calibration_samples':dict(Counter(cy.tolist())),'calibration_source':'source-track-disjoint from temporal encoder training, local continuous halves and simultaneous separate bodies','true_visit_accuracy_known':False,'feature_names':['appearance_mean','appearance_max','appearance_q75','row_best_mean','col_best_mean','upper_color','lower_color','aspect_difference','temporal_full_forward','temporal_best','temporal_best2','temporal_full_reverse','dtw_forward','dtw_reverse','dtw_best','dtw_gain'],'historical_records_used':False},indent=2))
np.savez_compressed(HERE/'results/pair_feature_vectors.npz',features=scores,owners=np.asarray(request_owner),calibration_features=cx,calibration_labels=cy)
old_dispositions=json.loads((REC/'variants/B_parts/event_dispositions.json').read_text());returns=old_dispositions['return_edges']
previous=json.loads((ROOT/'doorway_balance_20261108/results/delivery_records.json').read_text());unverified=[r for r in previous['candidates'] if not r['crossing_confirmed']]
configs=[('F_dense_static','连续帧静态对照',.75,.05),('G_temporal','时序网络',.75,.05),('H_alignment','时序与顺序对齐',.75,.05),('I_fusion','时序与原外观融合',.75,.05),('J_coverage','时序融合扩展',.5,.01)]
reports=[]
for name,title,threshold,gap_min in configs:
    prob=probs[name];prob[~legal]=0;raw=logit(np.clip(prob,1e-6,1-1e-6));order=np.argsort(-prob,axis=1);ra=np.empty_like(order);np.put_along_axis(ra,order,np.arange(len(exits))[None,:]+1,axis=1);order=np.argsort(-prob,axis=0);rb=np.empty_like(order);np.put_along_axis(rb,order,np.arange(len(entry))[:,None]+1,axis=0)
    utility=np.where(legal&(prob>=threshold),raw-logit(threshold),-1e4);objective=np.c_[utility,np.zeros((len(entry),len(entry)))];rr,cc0=linear_sum_assignment(objective,maximize=True);optimum=float(objective[rr,cc0].sum());main=[];unstable=[]
    for a,b in zip(rr,cc0):
        if b>=len(exits) or utility[a,b]<=0:continue
        value=objective[a,b];objective[a,b]=-1e4;ar,ac=linear_sum_assignment(objective,maximize=True);gap=optimum-float(objective[ar,ac].sum());objective[a,b]=value
        pair={'left':entry[a],'right':exits[b],'probability':float(prob[a,b]),'rank_forward':int(ra[a,b]),'rank_reverse':int(rb[a,b]),'global_alternative_gap':gap,'ordered_sequence_evidence':bool(has_temporal[a,b]),'method':name if has_temporal[a,b] else 'previous_static_fallback'}
        (main if gap>=gap_min else unstable).append(pair)
    links=main+returns;succ={r['left']:r['right'] for r in links};pred={r['right']:r['left'] for r in links};meta={(r['left'],r['right']):r for r in links};complete=[];short=[];unclosed=[];chains=[]
    for start in sorted(active,key=lambda i:events[i]['time']):
        if start in pred:continue
        chain=[start]
        while chain[-1] in succ:chain.append(succ[chain[-1]])
        first=events[chain[0]];last=events[chain[-1]];closed=first['direction']=='entry' and last['direction']=='exit'
        row={'entry_seconds':first['time'] if first['direction']=='entry' else None,'exit_seconds':last['time'] if last['direction']=='exit' else None,'entry_event':chain[0] if first['direction']=='entry' else None,'exit_event':chain[-1] if last['direction']=='exit' else None,'entry_interval':first['interval'] if first['direction']=='entry' else None,'exit_interval':last['interval'] if last['direction']=='exit' else None,'appearance':'','event_chain':chain,'source_pass':'automatic_'+name+'_20261109','association_links':[meta[a,b] for a,b in zip(chain[:-1],chain[1:])]}
        row['duration_seconds']=row['exit_seconds']-row['entry_seconds'] if closed else None
        if closed:(complete if row['duration_seconds']>=30 else short).append(row)
        else:
            for i in ([chain[0]] if first['direction']=='exit' else [])+([chain[-1]] if last['direction']=='entry' else []):
                e=events[i];unclosed.append({'event_id':i,'entry_seconds':e['time'] if e['direction']=='entry' else None,'exit_seconds':e['time'] if e['direction']=='exit' else None,'direction':e['direction'],'interval':e['interval'],'crossing_confirmed':True,'appearance':'','chain':chain})
        chains.append(row)
    complete.sort(key=lambda r:r['entry_seconds']);directory=HERE/'variants'/name;directory.mkdir(exist_ok=True)
    config={'id':name[0],'directory':name,'name':title,'threshold':threshold,'alternative_gap':gap_min,'workbook':f'store_dwell_{name[0]}_{title}.xlsx'}
    records={'mode':'automatic_doorway_temporal_20261109','variant':config,'confirmed':complete,'candidates':sorted(unclosed+unverified,key=lambda r:r['entry_seconds'] if r['entry_seconds'] is not None else r['exit_seconds']),'unclosed':unclosed,'historical_records_used':False}
    report={'variant':config,'complete_visits':len(complete),'unmatched_endpoints':len(unclosed),'unmatched_directions':dict(Counter(r['direction'] for r in unclosed)),'short_visits_excluded':len(short),'store_visit_edges':len(main),'temporal_supported_edges':sum(r['ordered_sequence_evidence'] for r in main),'static_fallback_edges':sum(not r['ordered_sequence_evidence'] for r in main),'continuous_return_links':len(returns),'ambiguous_edges':len(unstable),'eligible_entries':len(entry),'eligible_exits':len(exits),'unverified_candidates':len(unverified),'independent_accuracy_evaluated':False,'count_target_used':False,'historical_records_used':False}
    (directory/'delivery_records.json').write_text(json.dumps(records,ensure_ascii=False));(directory/'delivery_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(directory/'event_dispositions.json').write_text(json.dumps({'main_edges':main,'return_edges':returns,'short_visits':short,'unstable_edges':unstable,'chains':chains}));np.savez_compressed(directory/'pair_probabilities.npz',entries=entry,exits=exits,probability=prob,legal=legal,rank_forward=ra,rank_reverse=rb,temporal_support=has_temporal)
    reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
(HERE/'results/assignment_summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2));record.update(stage='temporal_pair_assignment_complete',elapsed_seconds=time.time()-started)
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
