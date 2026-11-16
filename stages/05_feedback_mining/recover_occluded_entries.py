"""Recover interval entrances when foot visibility interrupts an otherwise continuous body."""
from collections import Counter,defaultdict
import json,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results';started=time.time()
events=json.loads((ROOT/'doorway_temporal_20261109/events.json').read_text());known=json.loads((ROOT/'doorway_recovery_20261108/results/evidence_manifest.json').read_text())['events'];active={r['event_id'] for r in known};strict=json.loads((OUT/'new_entry_proposals.json').read_text());strict=[r for r in strict if r['source']=='visible_ankle_continuous_two_sides'];strict_ids={r['event_id'] for r in strict};measurements=defaultdict(list)
for p in sorted((ROOT/'doorway_pairer_20261108/query_round/sequences').glob('clip_*.json')):
    for key,rows in json.loads(p.read_text())['measurements'].items():measurements[int(key)].extend(rows)
boundary=np.array([[0,829],[450,807],[860,773],[1210,731],[1450,696],[1660,659],[1919,620]])
def iou(a,b):
    a=np.array(a);b=np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();return inter/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter+1e-8)
found=[];counts=Counter()
for eid,raw in measurements.items():
    if eid in active or eid in strict_ids or eid>=len(events):continue
    e=events[eid];rows=sorted(raw,key=lambda r:r['time']);usable=[]
    for r in rows:
        p=np.array(r['pose']);h=r['box'][3]-r['box'][1]
        if np.mean(p[[5,6,11,12],2])<.6 or r['match_iou']<.4 or h<100:continue
        valid=np.flatnonzero(p[[15,16],2]>=.45);foot=p[[15,16][valid[0]],:2] if len(valid)==1 else np.average(p[[15,16],:2],axis=0,weights=p[[15,16],2]) if len(valid)==2 else None
        usable.append({**r,'foot':None if foot is None else foot.tolist(),'foot_source':'two_ankles' if len(valid)==2 else 'one_ankle' if len(valid)==1 else 'occluded','height':h})
    # Keep each contiguous body chain separate; no appearance-only bridging.
    groups=[];g=[]
    for r in usable:
        if g:
            q=g[-1];gap=r['time']-q['time'];h=min(r['height'],q['height']);c=np.array([(r['box'][0]+r['box'][2])/2,(r['box'][1]+r['box'][3])/2]);d=np.array([(q['box'][0]+q['box'][2])/2,(q['box'][1]+q['box'][3])/2])
            if r['clip']!=q['clip'] or gap>.7 or gap<=0 or iou(r['box'],q['box'])<.35 or np.linalg.norm(c-d)>.35*h:groups.append(g);g=[]
        g.append(r)
    if g:groups.append(g)
    candidates=[]
    for group in groups:
        visible=[r for r in group if r['foot'] is not None]
        if len(visible)<4 or len(group)<7:continue
        offsets=np.array([(r['box'][3]-r['foot'][1])/r['height'] for r in visible]);correction=float(np.median(offsets));scatter=float(np.median(abs(offsets-correction)))
        if not (-.03<=correction<=.2) or scatter>.07:continue
        for r in group:
            if r['foot'] is None:r['foot']=[(r['box'][0]+r['box'][2])/2,r['box'][3]-correction*r['height']];r['foot_source']='local_ankle_calibrated_box'
            r['distance']=float(r['foot'][1]-np.interp(r['foot'][0],boundary[:,0],boundary[:,1]))
        for cut in range(2,len(group)-2):
            center=(group[cut-1]['time']+group[cut]['time'])/2;before=[r for r in group[:cut] if center-r['time']<=2.5];after=[r for r in group[cut:] if r['time']-center<=2.5];pre=[r for r in before if r['distance']>=12];post=[r for r in after if r['distance']<=-12]
            if min(len(pre),len(post))<2 or min(pre[-1]['time']-pre[0]['time'],post[-1]['time']-post[0]['time'])<.3:continue
            if np.mean([r['distance']>=0 for r in before])<.8 or np.mean([r['distance']<=0 for r in after])<.8:continue
            if not any(r['foot_source']!='local_ankle_calibrated_box' for r in pre) or not any(r['foot_source']!='local_ankle_calibrated_box' for r in post):continue
            low=pre[-1]['time'];high=post[0]['time']
            if high<=low or high-low>2.5:continue
            crossing=(low+high)/2;anchor=min(group,key=lambda r:abs(r['time']-crossing));old_anchor=min(group,key=lambda r:abs(r['time']-e['time']))
            if iou(old_anchor['box'],e['crossing_box'])<.3:continue
            foot=(np.array(pre[-1]['foot'])+post[0]['foot'])/2
            if any(abs(s['time']-crossing)<=1.0 and iou(s.get('crossing_box',events[s['event_id']]['crossing_box']),anchor['box'])>.25 for s in known+strict if s['direction']=='entry'):continue
            strength=min(pre[-1]['time']-pre[0]['time'],post[-1]['time']-post[0]['time']);candidates.append({'event_id':eid,'direction':'entry','time':crossing,'interval':[low,high],'crossing_box':anchor['box'],'crossing_foot':foot.tolist(),'clip':anchor['clip'],'source':'continuous_body_occluded_ankle_proposal','previous_direction':e['direction'],'previous_state_reason':e.get('state_reason'),'pre_duration':pre[-1]['time']-pre[0]['time'],'post_duration':post[-1]['time']-post[0]['time'],'strength':strength,'measurements':before+after,'inferred_foot_frames':sum(r['foot_source']=='local_ankle_calibrated_box' for r in before+after),'crossing_confirmed_by_human':False})
    if candidates:found.append(max(candidates,key=lambda r:(r['strength'],-abs(r['time']-e['time']))));counts['new_candidate']+=1
    else:counts['no_supported_body_transition']+=1
kept=[]
for r in sorted(found,key=lambda r:r['strength'],reverse=True):
    if any(abs(r['time']-s['time'])<=1 and iou(r['crossing_box'],s['crossing_box'])>.25 for s in kept):counts['duplicate_proposals']+=1;continue
    kept.append(r)
(OUT/'occluded_entry_proposals.json').write_text(json.dumps(kept));(OUT/'new_entry_proposals.json').write_text(json.dumps(strict+kept));summary={'strict_entries':len(strict),'additional_interval_entry_proposals':len(kept),'outcomes':dict(counts),'elapsed_seconds':time.time()-started,'automatically_confirmed_identity':False};(OUT/'occluded_entry_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
