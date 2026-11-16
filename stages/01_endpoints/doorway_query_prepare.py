import os
"""Deduplicate retrieval episodes and allocate query-driven local re-detection."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np

root=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));old=root/'rebuild_20261107';new=Path(__file__).resolve().parent
units=json.loads((new/'results/retrieval_units.json').read_text());events=json.loads((new/'events_states.json').read_text());f=np.load(new/'results/retrieval_scores.npz');scores=f['scores']
cal=json.loads((old/'logs/fusion_selection.json').read_text());nodes=json.loads((old/'work/bank/nodes.json').read_text());obs=dict(np.load(old/'work/bank/observations.npz'))
line=np.asarray(json.loads((root/'scene.json').read_text())['boundary_pixels']);exit_ids=np.asarray([e['event_id'] for e in events if e['direction']=='exit' or e['source_category']=='observed_unmatched_exit'])
times=np.asarray([u['time'] for u in units]);candidate=np.asarray([u['source_category'] in ('unverified_event','loose_doorway_detection') or u['direction']=='entry' for u in units]);covered=np.asarray([u['sapiens_count']>0 for u in units])
rank_scores=scores[exit_ids].copy()
for row,eid in enumerate(exit_ids):
    both=covered&covered[eid];rank_scores[row]=(rank_scores[row]-np.where(both,cal['threshold'],cal['anchor_base']['threshold']))/np.where(both,cal['negative_score_std'],cal['anchor_base']['negative_score_std'])
    legal=(times<events[eid]['time'])&candidate
    for j,u in enumerate(units):
        if u['source_category']=='loose_doorway_detection' and (u['end']>=events[eid]['time']-2.5 or (u['node']==events[eid]['node'] and u['end']>=events[eid]['time']-8)):legal[j]=False
    rank_scores[row,~legal]=-1e4
    if not units[eid]['sample_count']:rank_scores[row,:]=-1e4
order=np.argsort(-rank_scores,axis=0);reverse=np.empty_like(order);np.put_along_axis(reverse,order,np.arange(len(exit_ids))[:,None]+1,axis=0)
head=Counter();coverage=Counter();requests={};reports=[];hints=Counter()
for row,eid in enumerate(exit_ids):
    if events[eid]['source_category']!='observed_unmatched_exit':continue
    selected=[]
    for j in np.argsort(-rank_scores[row]):
        if rank_scores[row,j]<-9999 or len(selected)==20:break
        u=units[j]
        if not u['samples']:continue
        b=np.asarray(min(u['samples'],key=lambda r:abs(r['time']-u['time']))['box']);duplicate=False
        for k in selected:
            v=units[k]
            if abs(u['time']-v['time'])>2:continue
            c=np.asarray(min(v['samples'],key=lambda r:abs(r['time']-v['time']))['box']);inter=np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2])).prod()
            overlap=inter/((b[2:]-b[:2]).prod()+(c[2:]-c[:2]).prod()-inter+1e-6)
            if overlap>.6:duplicate=True;break
        if not duplicate:selected.append(int(j))
    candidates=[];loose_options=[]
    for rank,j in enumerate(selected,1):
        u=units[j];hint='not_applicable';seed=u['time'];details=None
        if u['source_category']=='loose_doorway_detection':
            ids=np.asarray(nodes[u['node']]['rows'] if u['node']>=0 else [s['row'] for s in u['samples']],int);ids=ids[(obs['time'][ids]>=u['start']-1.5)&(obs['time'][ids]<=u['end']+1.5)]
            ids=ids[np.argsort(obs['time'][ids])];t=obs['time'][ids];b=obs['boxes'][ids,:4];d=b[:,3]-np.interp((b[:,0]+b[:,2])/2,line[:,0],line[:,1]);cross=[]
            for q in range(1,len(ids)):
                if d[q-1]>0 and d[q]<=0 and t[q]-t[q-1]<=1.1 and np.any((t>=t[q]-1.5)&(t<t[q])&(d>=12)) and np.any((t>t[q])&(t<=t[q]+1.5)&(d<=-12)):cross.append(q)
            if cross:
                q=min(cross,key=lambda q:abs(t[q]-u['time']));seed=float(t[q-1]+d[q-1]/(d[q-1]-d[q])*(t[q]-t[q-1]));hint='box_motion_entry_hint'
            else:
                q=int(np.argmin(abs(d)));seed=float(t[q]);hint='near_line_without_entry_hint' if abs(d[q])<=45 else 'no_near_line_support'
            hints[hint]+=1
            if hint!='no_near_line_support' and seed<events[eid]['time']-2.5:
                loose_options.append((hint=='box_motion_entry_hint',-rank,j,seed))
        candidates.append({'gallery_id':j,'event_id':u['event_id'],'source_category':u['source_category'],'time':u['time'],'direction':u['direction'],
              'eligible_crossing':u['eligible_crossing'],'standardized_score':float(rank_scores[row,j]),'exit_to_entrance_rank':rank,'entrance_to_exit_rank':int(reverse[row,j]),
              'motion_hint':hint,'seed_time':seed,'identity_status':'candidate_only'})
    # At most two local episodes per exit, prioritize actual outside-to-inside motion.
    for entry_hint,negative_rank,j,seed in sorted(loose_options,reverse=True)[:2]:
        key=(tuple(units[j]['group_key']),round(seed))
        if key not in requests:requests[key]={'request_id':len(requests),'node':units[j]['node'],'seed_time':seed,'gallery_id':j,'exit_queries':[],
           'entry_motion_hint':entry_hint,'window_seconds':[-2.5,2.5],'seed_samples':units[j]['samples']}
        requests[key]['exit_queries'].append(int(eid))
    head[candidates[0]['source_category'] if candidates else 'no_earlier_searchable_candidate']+=1;coverage.update(set(c['source_category'] for c in candidates));reports.append({'exit_event':int(eid),'exit_time':events[eid]['time'],'candidates':candidates})
(new/'results/exit_300_retrieval_deduplicated.json').write_text(json.dumps(reports))
(new/'results/targeted_requests.json').write_text(json.dumps(list(requests.values())))
summary={'queried_original_exits':len(reports),'top1_candidate_source':dict(head),'top20_source_coverage_exit_count':dict(coverage),'top20_loose_motion_hints':dict(hints),
  'targeted_unique_windows':len(requests),'windows_with_entry_motion_hint':sum(r['entry_motion_hint'] for r in requests.values()),
  'candidate_scores_standardized_by_model_negative_calibration':True,'deduplicated_nearby_same_body_episodes':True,'retrieval_is_not_identity_confirmation':True}
(new/'results/retrieval_provenance_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
