"""Mine previously unresolved crossings using contiguous, visible-ankle tracks."""
from collections import Counter,defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;TEMP=ROOT/'doorway_temporal_20261109';BASE=ROOT/'doorway_pairer_20261108';started=time.time()
events=json.loads((TEMP/'events.json').read_text());known=json.loads((ROOT/'doorway_recovery_20261108/results/evidence_manifest.json').read_text())['events'];active={r['event_id'] for r in known}
measurements=defaultdict(list)
for p in sorted((BASE/'query_round/sequences').glob('clip_*.json')):
    for key,values in json.loads(p.read_text())['measurements'].items():measurements[int(key)].extend(values)
boundary=np.array([[0,829],[450,807],[860,773],[1210,731],[1450,696],[1660,659],[1919,620]])
def iou(a,b):
    a=np.array(a);b=np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();return inter/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter+1e-8)
found=[];outcomes=Counter()
for eid,rows in measurements.items():
    if eid in active or eid>=len(events):continue
    e=events[eid];rows=sorted(rows,key=lambda r:r['time']);clean=[]
    for r in rows:
        p=np.array(r['pose']);h=r['box'][3]-r['box'][1]
        if r['foot'] is None or min(p[[15,16],2])<.4 or np.mean(p[[5,6,11,12],2])<.45 or r['match_iou']<.5 or h<100:continue
        d=float(r['foot'][1]-np.interp(r['foot'][0],boundary[:,0],boundary[:,1]));clean.append({**r,'distance':d})
    if len(clean)<8:outcomes['insufficient_visible_ankle_sequence']+=1;continue
    groups=[];group=[]
    for r in clean:
        if group:
            q=group[-1];h=min(r['box'][3]-r['box'][1],q['box'][3]-q['box'][1])
            if r['time']-q['time']>.3 or iou(r['box'],q['box'])<.5 or np.linalg.norm(np.array(r['foot'])-q['foot'])>.35*h:groups.append(group);group=[]
        group.append(r)
    if group:groups.append(group)
    proposals=[]
    for group in groups:
        for cut in range(3,len(group)-3):
            before=group[:cut];after=group[cut:]
            pre=[r for r in before if r['distance']>=12];post=[r for r in after if r['distance']<=-12]
            if min(len(pre),len(post))<3 or pre[-1]['time']-pre[0]['time']<.3 or post[-1]['time']-post[0]['time']<.3:continue
            if np.mean([r['distance']>=0 for r in before])<.85 or np.mean([r['distance']<=0 for r in after])<.85:continue
            low=pre[-1]['time'];high=post[0]['time']
            if high-low>1.0 or high<=low:continue
            anchor=min(group,key=lambda r:abs(r['time']-e['time']))
            if iou(anchor['box'],e['crossing_box'])<.4:continue
            crossing=(low+high)/2;foot=(np.array(pre[-1]['foot'])+post[0]['foot'])/2
            duplicate=[p['event_id'] for p in known if p['direction']=='entry' and abs(p['time']-crossing)<=1 and np.linalg.norm(np.array(events[p['event_id']]['crossing_foot'])-foot)<.4*(anchor['box'][3]-anchor['box'][1])]
            if duplicate:outcomes['duplicates_existing_entry']+=1;continue
            strength=min(pre[-1]['time']-pre[0]['time'],post[-1]['time']-post[0]['time']);proposals.append({'event_id':eid,'direction':'entry','time':crossing,'interval':[low,high],'crossing_box':anchor['box'],'crossing_foot':foot.tolist(),'clip':anchor['clip'],'source':'visible_ankle_continuous_two_sides','previous_direction':e['direction'],'previous_state_reason':e.get('state_reason'),'pre_duration':pre[-1]['time']-pre[0]['time'],'post_duration':post[-1]['time']-post[0]['time'],'strength':strength,'measurements':group,'crossing_confirmed_by_human':False})
    if proposals:found.append(max(proposals,key=lambda r:r['strength']));outcomes['new_entry_proposal']+=1
    else:outcomes['no_continuous_outside_to_inside_evidence']+=1
# Nearby candidates from duplicate detector tracks represent one hypothesis.
kept=[]
for r in sorted(found,key=lambda r:r['strength'],reverse=True):
    if any(abs(r['time']-s['time'])<=.6 and iou(r['crossing_box'],s['crossing_box'])>.5 for s in kept):continue
    kept.append(r)
(HERE/'results/new_entry_proposals.json').write_text(json.dumps(kept));summary={'proposals':len(kept),'outcomes':dict(outcomes),'direct_visible_ankles_only':True,'minimum_stable_side_seconds':.3,'historical_12_used':False,'elapsed_seconds':time.time()-started};(HERE/'results/new_entry_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
