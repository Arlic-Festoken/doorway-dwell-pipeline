import os
"""Confirm doorway state transitions, including uncertain occluded-foot intervals."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

root=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));old=root/'rebuild_20261107';new=Path(__file__).resolve().parent;started=time.time()
while not (new/'logs/sequences_status.json').exists():
    if time.time()-started>3700:raise TimeoutError('Doorway sequences did not finish')
    time.sleep(3)
events=json.loads((new/'events.json').read_text());reference=json.loads((new/'reference_mapping.json').read_text())
heldout={v['event_id'] for r in reference for v in r['endpoints'].values() if v['event_id'] is not None}
native=json.loads((old/'work/association_final/refined_events.json').read_text())
boundary=np.asarray(json.loads((root/'scene.json').read_text())['boundary_pixels'])
measurements=defaultdict(list)
for path in sorted((new/'sequences').glob('clip_*.json')):
    for k,values in json.loads(path.read_text())['measurements'].items():measurements[int(k)].extend(values)
training=[];targets=[];group_ids=[];projection_info={}
for eid,values in measurements.items():
    values.sort(key=lambda r:r['time'])
    for j,r in enumerate(values):
        p=np.asarray(r['pose']);b=np.asarray(r['box'])
        if min(p[[5,6,11,12],2])<.5:continue
        shoulder=p[[5,6],:2].mean(0);hip=p[[11,12],:2].mean(0);torso=hip[1]-shoulder[1]
        if torso<20 or torso>.7*r['height']:continue
        x=np.asarray([abs(p[5,0]-p[6,0])/torso,abs(p[11,0]-p[12,0])/torso,torso/r['height'],(hip[0]-shoulder[0])/torso])
        projection_info[(eid,j)]=(x,hip,torso)
        if r['foot'] is not None and r['match_iou']>=.6 and min(p[[15,16],2])>=.7 and eid not in heldout:
            training.append(x);targets.append((np.asarray(r['foot'])-hip)/torso);group_ids.append(eid)
X=np.asarray(training);Y=np.asarray(targets);validation=np.asarray(group_ids)%5==0
if min(sum(validation),sum(~validation))<100:raise ValueError('Insufficient visible-ankle samples for automatic hip projection')
scaler=StandardScaler().fit(X[~validation]);models=[HuberRegressor(max_iter=300).fit(scaler.transform(X[~validation]),Y[~validation,k]) for k in range(2)]
prediction=np.column_stack([m.predict(scaler.transform(X[validation])) for m in models]);errors=np.quantile(abs(prediction-Y[validation]),.9,axis=0)
model_data={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficient':[m.coef_.tolist() for m in models],'intercept':[float(m.intercept_) for m in models],
            'validation_absolute_error_q90_torso_units':errors.tolist(),'training_samples':int(sum(~validation)),'validation_samples':int(sum(validation)),'historical_event_samples_used':0}
(new/'results/hip_projection.json').write_text(json.dumps(model_data,indent=2))
results=[];counts=Counter();changes=[]
for e in events:
    eid=e['event_id'];values=measurements[eid];observations=[]
    for j,r in enumerate(values):
        method='ankles';sigma=6.;foot=r['foot'];weight=min(1,max(.2,r['quality']))
        if foot is None:
            info=projection_info.get((eid,j))
            if info is None:continue
            x,hip,torso=info;z=scaler.transform(x[None,:]);foot=hip+np.asarray([m.predict(z)[0] for m in models])*torso
            sigma=max(6,float((errors[1]+.3*errors[0])*torso));weight*=.35;method='hip_projection'
        foot=np.asarray(foot);distance=float(foot[1]-np.interp(foot[0],boundary[:,0],boundary[:,1]))
        observations.append({'time':r['time'],'distance':distance,'sigma':sigma,'weight':weight,'method':method,'foot':foot.tolist(),'box':r['box']})
    result={**e,'state_supported':False,'state_reason':'insufficient_observations','native_previously_observed':str(eid) in native,'observations':observations}
    if len(observations)>=12:
        t=np.asarray([r['time'] for r in observations]);d=np.asarray([r['distance'] for r in observations]);sigma=np.asarray([r['sigma'] for r in observations]);w=np.asarray([r['weight'] for r in observations])
        outside=ndtr(d/sigma).clip(1e-6,1-1e-6);log_out=np.log(outside)*w;log_in=np.log1p(-outside)*w
        co=np.r_[0,np.cumsum(log_out)];ci=np.r_[0,np.cumsum(log_in)];constant=max(co[-1],ci[-1]);candidates=[]
        boxes=np.asarray([r['box'] for r in observations]);centers=(boxes[:,:2]+boxes[:,2:])/2
        jump=np.linalg.norm(np.diff(centers,axis=0),axis=1)/np.maximum(36,np.minimum(boxes[:-1,3]-boxes[:-1,1],boxes[1:,3]-boxes[1:,1]))
        ambiguous_motion=bool(np.any((np.diff(t)<.3)&(jump>1+2*np.diff(t))))
        credible=np.where((d>=12)&(outside>=.84),1,np.where((d<=-12)&(outside<=.16),-1,0))
        stable_runs=[];run_start=0
        for j in range(1,len(t)+1):
            if j==len(t) or credible[j]!=credible[run_start] or t[j]-t[j-1]>.25:
                if credible[run_start] and t[j-1]-t[run_start]>=.5:stable_runs.append((run_start,j-1,int(credible[run_start])))
                run_start=j
        sides=[]
        for a,b,side in stable_runs:
            if not sides or side!=sides[-1]:sides.append(side)
        result['stable_state_runs']=[{'start':float(t[a]),'end':float(t[b]),'side':'outside' if s>0 else 'inside'} for a,b,s in stable_runs]
        if ambiguous_motion:result['state_reason']='ambiguous_body_motion'
        elif len(sides)>2:result['state_reason']='multiple_transitions_in_window'
        else:
            for direction,pre,post in [('entry',1,-1),('exit',-1,1)]:
                score=(co[:-1]+ci[-1]-ci[:-1]) if direction=='entry' else (ci[:-1]+co[-1]-co[:-1])
                for cut in range(1,len(t)):
                    before=[r for r in stable_runs if r[2]==pre and r[1]<cut];after=[r for r in stable_runs if r[2]==post and r[0]>=cut]
                    if not before or not after:continue
                    gain=float(score[cut]-constant-np.log(len(t)))
                    if gain<=0:continue
                    left=before[-1][1];right=after[0][0]
                    candidates.append({'direction':direction,'time':float((t[cut-1]+t[cut])/2),'interval':[float(t[left]),float(t[right])],
                        'state_log_evidence_gain':gain,'hip_observations':sum(r['method']=='hip_projection' for r in observations[left:right+1]),
                        'pre_state_duration':float(t[before[-1][1]]-t[before[-1][0]]),'post_state_duration':float(t[after[0][1]]-t[after[0][0]])})
            if candidates:
                best=max(candidates,key=lambda r:r['state_log_evidence_gain']);result.update(best,state_supported=True,state_reason='two_side_state_transition',time_method='doorway_state_interval')
                if best['direction']!=e['direction']:changes.append({'event_id':eid,'old':e['direction'],'new':best['direction'],'previous_native':str(eid) in native})
            else:result['state_reason']='no_supported_single_transition'
    if str(eid) in native:
        if result['state_supported'] and result['direction']!=e['direction']:
            result['state_direction_proposal']=result['direction'];result['direction']=e['direction'];result['state_reason']='direction_conflict_with_native';result['state_supported']=False
        result.update(time=native[str(eid)]['time'],interval=native[str(eid)]['interval'],time_method=native[str(eid)]['method'])
    result['eligible_crossing']=result['native_previously_observed'] or result['state_supported']
    if e['source_category']=='unverified_event':counts['unverified_'+('confirmed_'+result['direction'] if result['state_supported'] else result['state_reason'])]+=1
    results.append(result)
(new/'events_states.json').write_text(json.dumps(results))
summary={'stage':'doorway_states_complete','original_unverified_candidates':sum(e['source_category']=='unverified_event' for e in events),
         'new_entries_from_unverified':counts['unverified_confirmed_entry'],'new_exits_from_unverified':counts['unverified_confirmed_exit'],
         'unverified_outcomes':dict(counts),'direction_change_proposals':changes,'previous_native_crossings_retained':sum(r['native_previously_observed'] for r in results),
         'eligible_entry_events':sum(r['eligible_crossing'] and r['direction']=='entry' for r in results),'eligible_exit_events':sum(r['eligible_crossing'] and r['direction']=='exit' for r in results),
         'new_interval_width_median':float(np.median([r['interval'][1]-r['interval'][0] for r in results if r['source_category']=='unverified_event' and r['state_supported']])) if any(r['source_category']=='unverified_event' and r['state_supported'] for r in results) else None,
         'historical_identity_labels_used_for_training':0,'elapsed_seconds':round(time.time()-started,1)}
(new/'results/state_recovery_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
