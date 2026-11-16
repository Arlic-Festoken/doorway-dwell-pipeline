import os
"""Calibrate acceptance on simultaneous distinct bodies, then assign endpoints."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

root=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));new=Path(__file__).resolve().parent;old=root/'rebuild_20261107';started=time.time()
while not (new/'results/retrieval_summary.json').exists():
    if time.time()-started>3600:raise TimeoutError('Retrieval incomplete')
    time.sleep(3)
events=json.loads((new/'events_states.json').read_text());reference=json.loads((new/'reference_mapping.json').read_text());previous=json.loads((new/'previous_records.json').read_text())
units=json.loads((new/'results/retrieval_units.json').read_text());matrices=np.load(new/'results/retrieval_scores.npz');score=matrices['scores'][:,:len(events)].copy()
heldout={v['event_id'] for r in reference for v in r['endpoints'].values() if v['event_id'] is not None}
measurements=defaultdict(dict)
for f in sorted((new/'sequences').glob('clip_*.json')):
    for eid,rows in json.loads(f.read_text())['measurements'].items():
        for r in rows:measurements[int(eid)][(r['clip'],r['frame_index'])]=r

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/(((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection)+1e-8))

ordered_measurements={i:sorted(rows.values(),key=lambda r:r['time']) for i,rows in measurements.items()}
measurement_times={i:np.asarray([r['time'] for r in rows]) for i,rows in ordered_measurements.items()}
def aligned_observations(a,b):
    common=set(measurements[a])&set(measurements[b])
    if len(common)>=3:return [(measurements[a][k],measurements[b][k]) for k in common]
    # The original pass is 10 Hz, query windows are 5 Hz. Align within one native frame.
    aligned=[];right=ordered_measurements.get(b,[]);times=measurement_times.get(b,np.zeros(0))
    if not right:return aligned
    for row in ordered_measurements.get(a,[]):
        pos=int(np.searchsorted(times,row['time']));indices=[j for j in (pos-1,pos) if 0<=j<len(right)];j=min(indices,key=lambda j:abs(times[j]-row['time']))
        if right[j]['clip']==row['clip'] and abs(times[j]-row['time'])<=.04:aligned.append((row,right[j]))
    return aligned

# Collapse only duplicate observations of the same physical crossing, with shared-frame evidence.
canonical=np.arange(len(events));duplicate_evidence=[]
priority=sorted(range(len(events)),key=lambda i:(events[i]['native_previously_observed'],events[i]['eligible_crossing'],units[i]['sample_count'],-i),reverse=True)
priority_index={v:i for i,v in enumerate(priority)}
for a in priority:
    if canonical[a]!=a:continue
    for b in priority:
        if b==a or canonical[b]!=b or priority_index[b]<priority_index[a]:continue
        if events[a]['direction']!=events[b]['direction'] or abs(events[a]['time']-events[b]['time'])>1.2:continue
        overlap=[iou(x['box'],y['box']) for x,y in aligned_observations(a,b)]
        if len(overlap)>=3 and np.median(overlap)>=.7 and np.mean(np.asarray(overlap)>.6)>=.8:
            canonical[b]=a;duplicate_evidence.append({'event_id':b,'canonical_event':a,'shared_frames':len(overlap),'median_box_iou':float(np.median(overlap))})
heldout={int(canonical[i]) for i in heldout}

negatives=[];negative_pairs=[]
for a in range(len(events)):
    if a in heldout or canonical[a]!=a or units[a]['sample_count']<3:continue
    for b in range(a+1,len(events)):
        if b in heldout or canonical[b]!=b or units[b]['sample_count']<3 or abs(events[a]['time']-events[b]['time'])>2:continue
        separated=[]
        for left,right in aligned_observations(a,b):
            x=np.asarray(left['box']);y=np.asarray(right['box']);h=min(x[3]-x[1],y[3]-y[1]);dist=np.linalg.norm((x[:2]+x[2:]-y[:2]-y[2:])/2)
            separated.append(iou(x,y)<.02 and dist>.8*h)
        if len(separated)>=3 and np.mean(separated)>=.9 and score[a,b]>-9999:
            negatives.append(float(score[a,b]));negative_pairs.append([a,b])
if len(negatives)<100:raise ValueError('Not enough simultaneous independent doorway bodies for score calibration')
threshold=float(np.quantile(negatives,.995));strict=float(np.quantile(negatives,.999));spread=max(.5,float(np.std(negatives)))
calibration={'method':'frozen_fusion_threshold_on_same_frame_spatially_distinct_bodies','negative_examples':len(negatives),
    'negative_score_quantile':.995,'threshold':threshold,'strict_threshold':strict,'score_std':spread,
    'historical_labels_used':0,'pairs':negative_pairs,'historical_gate_baseline':8}
(new/'results/acceptance_calibration.json').write_text(json.dumps(calibration))
(new/'results/duplicate_events.json').write_text(json.dumps(duplicate_evidence))

def assign(reserved):
    entries=[i for i,e in enumerate(events) if canonical[i]==i and i not in reserved and e['eligible_crossing'] and e['direction']=='entry' and units[i]['sample_count']>0]
    exits=[i for i,e in enumerate(events) if canonical[i]==i and i not in reserved and e['eligible_crossing'] and e['direction']=='exit' and units[i]['sample_count']>0]
    # Scores are symmetric appearance evidence; no nearest-exit or dwell-time prior.
    raw=score[np.ix_(exits,entries)].T.copy();legal=np.asarray([events[j]['time'] for j in exits])[None,:]>np.asarray([events[i]['time'] for i in entries])[:,None]
    raw[~legal]=-1e4
    order=np.argsort(-raw,axis=1);rank_out=np.empty_like(order);np.put_along_axis(rank_out,order,np.arange(len(exits))[None,:]+1,axis=1)
    order=np.argsort(-raw,axis=0);rank_in=np.empty_like(order);np.put_along_axis(rank_in,order,np.arange(len(entries))[:,None]+1,axis=0)
    candidate=legal&((rank_out<=20)|(rank_in<=20));gain=(raw-threshold)/spread;gain[~candidate]=-1e4
    active_a=set(range(len(entries)));active_b=set(range(len(exits)));pairs=[]
    for round_id in range(3):
        if not active_a or not active_b:break
        ai=sorted(active_a);bi=sorted(active_b);sub=raw[np.ix_(ai,bi)];best_a=sub.argmax(1);best_b=sub.argmax(0);new_pairs=[]
        for row,local_col in enumerate(best_a):
            if best_b[local_col]!=row:continue
            a=ai[row];b=bi[local_col];value=raw[a,b]
            second_a=np.partition(sub[row],-2)[-2] if len(bi)>1 else -1e4
            second_b=np.partition(sub[:,local_col],-2)[-2] if len(ai)>1 else -1e4
            if candidate[a,b] and value>=strict and min(value-second_a,value-second_b)>=.25*spread:
                new_pairs.append((a,b,'mutual_clear_winner_round_'+str(round_id+1),0.))
        if not new_pairs:break
        for a,b,method,bonus in new_pairs:pairs.append((a,b,method,bonus));active_a.remove(a);active_b.remove(b)
    # A confirmed companion can resolve competition but cannot lift sub-threshold appearance.
    bonus=np.zeros_like(gain)
    for a in active_a:
        ea=events[entries[a]];ba=np.asarray(ea['box']);ha=ba[3]-ba[1]
        for b in active_b:
            if not candidate[a,b] or gain[a,b]<=0:continue
            eb=events[exits[b]];bb=np.asarray(eb['box']);hb=bb[3]-bb[1]
            for pa,pb,_,_ in pairs:
                p=events[entries[pa]];q=events[exits[pb]];dt=ea['time']-p['time'];du=eb['time']-q['time']
                if abs(dt)>3 or abs(du)>3 or abs(dt-du)>2:continue
                pbox=np.asarray(p['box']);qbox=np.asarray(q['box']);dp=(ba[:2]+ba[2:]-pbox[:2]-pbox[2:])/2;dq=(bb[:2]+bb[2:]-qbox[:2]-qbox[2:])/2
                if np.linalg.norm(dp)<1.5*ha and np.linalg.norm(dq)<1.5*hb and dp[0]*dq[0]>=0:bonus[a,b]=.15;break
    if active_a and active_b:
        ai=sorted(active_a);bi=sorted(active_b);sub=gain[np.ix_(ai,bi)]+bonus[np.ix_(ai,bi)]
        sub[gain[np.ix_(ai,bi)]<=0]=-1e4
        augmented=np.concatenate([sub,np.zeros((len(ai),len(ai)))],axis=1);rows,cols=linear_sum_assignment(augmented,maximize=True)
        for a,b in zip(rows,cols):
            if b<len(bi) and sub[a,b]>0:pairs.append((ai[a],bi[b],'global_one_to_one',float(bonus[ai[a],bi[b]])))
    result=[]
    for a,b,method,extra in pairs:
        i=entries[a];j=exits[b];e=events[i];x=events[j]
        result.append({'entry_seconds':e['time'],'exit_seconds':x['time'],'duration_seconds':x['time']-e['time'],'entry_event':i,'exit_event':j,
          'entry_interval':e['interval'],'exit_interval':x['interval'],'appearance':'','source_pass':'automatic_doorway_20261108',
          'association_method':method,'appearance_score':float(raw[a,b]),'companion_support_bonus':extra,
          'entry_to_exit_rank':int(rank_out[a,b]),'exit_to_entry_rank':int(rank_in[a,b]),'entry_time_method':e['time_method'],'exit_time_method':x['time_method']})
    return result

evaluation=assign(set());predicted={(v['entry_event'],v['exit_event']) for v in evaluation if v['duration_seconds']>=30};recovery=[]
for r in reference:
    if not r['included_by_30s_rule']:continue
    a=r['endpoints']['entry']['event_id'];b=r['endpoints']['exit']['event_id'];pair=(int(canonical[a]),int(canonical[b])) if a is not None and b is not None else None
    recovery.append({'reference_id':r['reference_id'],'mapped_pair':pair,'recovered':pair in predicted,
       'appearance_score':float(score[pair[1],pair[0]]) if pair is not None else None,'baseline_recovered':r['complete_pair_recovered']})
recovered=sum(r['recovered'] for r in recovery);accepted=recovered>=8
reserved={int(canonical[v['event_id']]) for r in reference if r['included_by_30s_rule'] for v in r['endpoints'].values() if v['event_id'] is not None}
automatic=assign(reserved);short=[r for r in automatic if r['duration_seconds']<30];complete=[r for r in automatic if r['duration_seconds']>=30]
historical=[r for r in previous['confirmed'] if not r['source_pass'].startswith('automatic_')]
used=reserved|{v[k] for v in automatic for k in ('entry_event','exit_event')};unmatched=[{k:e[k] for k in ('event_id','time','direction','source_category','interval','time_method')} for i,e in enumerate(events) if canonical[i]==i and e['eligible_crossing'] and i not in used]
old_observed=[e for e in events if e['source_category'].startswith('observed_unmatched_')];outcomes=Counter()
for e in old_observed:
    c=int(canonical[e['event_id']]);outcomes['duplicate_of_same_crossing' if c!=e['event_id'] else 'used_in_complete_visit' if any(c in (v['entry_event'],v['exit_event']) for v in complete) else 'used_in_short_visit' if any(c in (v['entry_event'],v['exit_event']) for v in short) else 'historical_locked' if c in reserved else 'still_unmatched']+=1
summary={'stage':'doorway_assignment_complete','accepted_for_delivery':accepted,'historical_recovered':recovered,'historical_total':12,'baseline_recovered':8,
    'automatic_complete_proposal':len(complete),'historical_locked_records':len(historical),'short_proposals_excluded':len(short),'duplicate_crossing_events':len(duplicate_evidence),
    'observed_unmatched_proposal':len(unmatched),'unmatched_directions':dict(Counter(e['direction'] for e in unmatched)),
    'original_382_outcomes_proposal':dict(outcomes),'mutual_first_fraction':float(np.mean([r['entry_to_exit_rank']==r['exit_to_entry_rank']==1 for r in complete])) if complete else None,
    'completed_over_20_minutes':sum(r['duration_seconds']>=1200 for r in complete),'elapsed_seconds':round(time.time()-started,1),
    'rollback_reason':None if accepted else 'Historical recovery declined; proposal not promoted and previous delivery retained'}
(new/'results/historical_recovery.json').write_text(json.dumps(recovery,indent=2))
(new/'results/assignment_proposal.json').write_text(json.dumps({'automatic':complete,'historical':historical,'short':short,'unmatched':unmatched},ensure_ascii=False))
if accepted:
    (new/'results/workbook_records.json').write_text(json.dumps({'mode':'doorway_endpoint_recovery','confirmed':complete+historical,'unclosed':unmatched,'candidates':unmatched,
       'new_automatic_records':len(complete),'historical_confirmed_records':len(historical),'minimum_duration_seconds':30},ensure_ascii=False))
(new/'results/assignment_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
