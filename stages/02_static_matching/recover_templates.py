import os
"""Recover doorway views through continuous local bodies, not whole-store paths."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(os.environ.get('DWELL_PROJECT_ROOT', '.'));BANK=ROOT/'rebuild_20261107/work/bank';PURE=ROOT/'doorway_evidence_20261108';HERE=Path(__file__).resolve().parent;started=time.time()
parser=argparse.ArgumentParser();parser.add_argument('--allow-missing-sapiens',action='store_true');args=parser.parse_args()
for name in ('results','logs','panels','crops','contexts'):(HERE/name).mkdir(exist_ok=True)
events=json.loads((ROOT/'doorway_balance_20261108/events_final.json').read_text())
old_pack=json.loads((PURE/'results/evidence_manifest.json').read_text());old_samples=json.loads((PURE/'results/clean_feature_samples.json').read_text())
z=np.load(PURE/'clean_features.npz');pose_index={str(k):i for i,k in enumerate(z['keys'])};pc=z['cope'].astype(np.float32);ps=z['sapiens'].astype(np.float32)
pc/=np.linalg.norm(pc,axis=1,keepdims=True).clip(1e-8);ps/=np.linalg.norm(ps,axis=1,keepdims=True).clip(1e-8)
by_event=defaultdict(list)
for r in old_samples:by_event[r['event_id']].append(r)
pose_geometry={}
for path in sorted((ROOT/'doorway_pairer_20261108/query_round/sequences').glob('clip_*.json')):
    for r in json.loads(path.read_text())['crops']:pose_geometry[r['path']]=(r['clip'],r['foot'])
bc=np.load(BANK/'features_adapted.npz');bs=np.load(BANK/'sapiens_aligned.npz');cf=bc['features'].astype(np.float32);sf=bs['features'].astype(np.float32)
cf/=np.linalg.norm(cf,axis=1,keepdims=True).clip(1e-8);sf/=np.linalg.norm(sf,axis=1,keepdims=True).clip(1e-8)
smap={int(v):i for i,v in enumerate(bs['sample_indices'])};bt=bc['time'];bn=bc['nodes'];bq=bc['quality'];bo=bc['overlap'];br=bc['reliability']
metadata=json.loads((BANK/'samples.json').read_text());nodes=json.loads((BANK/'nodes.json').read_text());obs=np.load(BANK/'observations.npz');ot=obs['time'];ob=obs['boxes'];on=obs['node'];oc=obs['clip'];order=np.argsort(ot);ordered_times=ot[order]
by_node=defaultdict(list)
for i in np.flatnonzero((bq>=.2)&(bo<=.3)&(br>=.7)):
    i=int(i)
    if i not in smap and not args.allow_missing_sapiens:continue
    if bq[i]>=.2 and bo[i]<=.3 and br[i]>=.7:by_node[int(bn[i])].append(i)
clocks=[json.loads((ROOT/'work/clocks'/f'VID{i:05d}.json').read_text())['offset_seconds'] for i in range(106)]
pts=[np.asarray([float(r['best_effort_timestamp_time']) for r in json.loads((ROOT/'work/native_crossings'/f'{i:03d}_pts.json').read_text())['frames']]) for i in range(106)]

def ious(a,boxes):
    boxes=np.asarray(boxes);a=np.asarray(a);intersection=np.maximum(0,np.minimum(boxes[:,2:],a[2:])-np.maximum(boxes[:,:2],a[:2])).prod(1)
    return intersection/((boxes[:,2:]-boxes[:,:2]).prod(1)+(a[2:]-a[:2]).prod()-intersection).clip(1)

def matched_node(frame):
    lo,hi=np.searchsorted(ordered_times,[frame['time']-.075,frame['time']+.075]);rows=order[lo:hi]
    if not len(rows):return None
    values=ious(frame['box'],ob[rows,:4]);best=int(np.argmax(values));row=int(rows[best])
    if values[best]<.5:return None
    others=ious(ob[row,:4],ob[rows,:4])<.65
    if others.any() and values[best]-values[others].max()<.15:return None
    return int(on[row]),float(ot[row]),ob[row,:4]

def connected_samples(node,at):
    rows=np.asarray(nodes[node]['rows']);rows=rows[abs(ot[rows]-at)<=6];rows=rows[np.argsort(ot[rows])]
    if not len(rows):return []
    times=ot[rows];boxes=ob[rows,:4];height=boxes[:,3]-boxes[:,1];feet=np.c_[(boxes[:,0]+boxes[:,2])/2,boxes[:,3]]
    intersection=np.maximum(0,np.minimum(boxes[1:,2:],boxes[:-1,2:])-np.maximum(boxes[1:,:2],boxes[:-1,:2])).prod(1);area=(boxes[:,2:]-boxes[:,:2]).prod(1);overlap=intersection/(area[1:]+area[:-1]-intersection).clip(1)
    breaks=(np.diff(times)>.25)|(overlap<=.5)|(np.linalg.norm(np.diff(feet,axis=0),axis=1)>.35*np.minimum(height[1:],height[:-1]))
    segments=np.split(np.arange(len(rows)),np.flatnonzero(breaks)+1);segment=min(segments,key=lambda ix:np.min(abs(times[ix]-at)));low=float(times[segment[0]]);high=float(times[segment[-1]])
    return [i for i in by_node[node] if low<=bt[i]<=high]

all_features={};new_samples=[];packs=[];summary=Counter();trace=[]
for previous in old_pack['events']:
    eid=previous['event_id'];e=events[eid];at=e['time'];candidates=[];seed_features=[];linked={};sources=[]
    for r in by_event[eid]:
        i=pose_index[r['key']];key='pose:'+r['key'];source=ROOT/'doorway_pairer_20261108/query_round'/r['original_crop'];near=min(previous['frames'],key=lambda f:abs(f['time']-r['time']))
        crop_clip,crop_foot=pose_geometry[r['original_crop']]
        frame={'time':r['time'],'box':r['box'],'clip':crop_clip,'foot':crop_foot,'foot_method':'pose_ankles','source_crop':str(source),'key':key,'quality':1.0,'fragment':0}
        candidates.append(frame);all_features[key]=(pc[i],ps[i]);seed_features.append((r['time'],pc[i],ps[i]));sources.append('clean_pose')
    if e.get('crossgap_nodes'):
        for fragment,node in enumerate(e['crossgap_nodes']):linked[(node,fragment)]=e['interval'][fragment]
        correction=np.array([0.,-.04])
    else:
        for frame in previous['frames']:
            match=matched_node(frame)
            if match is not None:linked[(match[0],0)]=match[1]
        correction=np.median([[(f['foot'][0]-(f['box'][0]+f['box'][2])/2)/(f['box'][3]-f['box'][1]),(f['foot'][1]-f['box'][3])/(f['box'][3]-f['box'][1])] for f in previous['frames']],axis=0) if previous['frames'] else np.array([0.,-.04])
        if not linked:
            match=matched_node({'time':at,'box':e['crossing_box']})
            if match is not None:linked[(match[0],0)]=match[1];sources.append('crossing_box_anchor')
    accepted_bank=[]
    for (node,fragment),anchor_time in linked.items():
        possible=connected_samples(node,anchor_time)
        if not possible:continue
        if seed_features and not e.get('crossgap_nodes'):
            _,ac,ass=min(seed_features,key=lambda r:abs(r[0]-anchor_time))
        else:
            seed=min(possible,key=lambda i:abs(bt[i]-anchor_time));ac=cf[seed];ass=sf[smap[seed]] if seed in smap else np.zeros(4096,np.float32)
        for i in possible:
            cosine_c=float(cf[i]@ac);cosine_s=float(sf[smap[i]]@ass) if i in smap and np.linalg.norm(ass)>.1 else None
            if (cosine_s is not None and (cosine_c<.45 or cosine_s<.25)) or (cosine_s is None and cosine_c<.55):summary['appearance_inconsistent_bank_views_rejected']+=1;continue
            r=metadata[i];box=r['box'];h=box[3]-box[1]
            if h<100 or h/max(box[2]-box[0],1)<1.25:continue
            key=f'bank:{i}';all_features[key]=(cf[i],sf[smap[i]] if i in smap else np.zeros(4096,np.float32));foot=(np.asarray([(box[0]+box[2])/2,box[3]])+correction*h).tolist()
            candidates.append({'time':r['time'],'box':box,'clip':r['clip'],'foot':foot,'foot_method':'box_estimate','source_crop':str(ROOT/'rebuild_20261107'/r['path']),'key':key,'quality':float(bq[i]),'fragment':fragment,'bank_node':node})
            accepted_bank.append(i)
    # Keep complementary views of the same local body. Do not mix interval halves.
    selected=[]
    for fragment in sorted(set(r['fragment'] for r in candidates)):
        pool=[r for r in candidates if r['fragment']==fragment];unique=[]
        for r in sorted(pool,key=lambda r:r['quality'],reverse=True):
            if not any(abs(r['time']-s['time'])<.075 and ious(r['box'],[s['box']])[0]>.7 for s in unique):unique.append(r)
        if not unique:continue
        part=[min(unique,key=lambda r:abs(r['time']-(e['interval'][fragment] if e.get('crossgap_nodes') else at)))];remaining=[r for r in unique if r not in part]
        while remaining and len(part)<(6 if e.get('crossgap_nodes') else 12):
            def novelty(r):
                c,s=all_features[r['key']];similarity=max(.5*float(c@all_features[q['key']][0])+.5*float(s@all_features[q['key']][1]) if np.linalg.norm(s)>.1 and np.linalg.norm(all_features[q['key']][1])>.1 else float(c@all_features[q['key']][0]) for q in part)
                return (1-similarity)*np.sqrt(max(.1,r['quality']))
            chosen=max(remaining,key=novelty);part.append(chosen);remaining.remove(chosen)
        selected.extend(part)
    selected.sort(key=lambda r:r['time'])
    if not e.get('crossgap_nodes') and linked:events[eid]['node']=min(node for node,fragment in linked)
    kind='two_fragment_interval' if e.get('crossgap_nodes') else 'continuous_local_body'
    usable=len(selected)>=3 and (not e.get('crossgap_nodes') or all(sum(r['fragment']==f for r in selected)>=2 for f in (0,1)))
    # Display at most eight views while preserving all clean feature templates.
    display=selected if len(selected)<=8 else [selected[i] for i in np.rint(np.linspace(0,len(selected)-1,8)).astype(int)]
    frames=[]
    for k,r in enumerate(display):
        clip=r['clip'];frame_index=int(np.argmin(abs(pts[clip]+clocks[clip]-r['time'])))
        frames.append({**r,'cell':f'V{k+1}','native_frame_index':frame_index,'time':float(pts[clip][frame_index]+clocks[clip]),'crop':f'crops/e{eid:04d}_v{k+1}.jpg','context':f'contexts/e{eid:04d}_v{k+1}.jpg'})
    for r in selected:new_samples.append({'event_id':eid,'key':r['key'],'time':r['time'],'box':r['box'],'original_crop':r['source_crop'],'fragment':r['fragment'],'source':'bank' if r['key'].startswith('bank:') else 'pose'})
    status='ready' if usable else 'insufficient_clean_body_views'
    packs.append({'event_id':eid,'direction':e['direction'],'time':at,'interval':e['interval'],'source_category':e['source_category'],'kind':kind,'continuous_crossing':bool(selected) and not bool(e.get('crossgap_nodes')),'ready_for_judge':usable,'status':status,'frames':frames,'panel':f'panels/e{eid:04d}.jpg','template_count':len(selected),'linked_local_nodes':sorted(set(n for n,f in linked)),'previous_status':previous['status']})
    summary['endpoints']+=1;summary['usable_endpoints']+=usable;summary['newly_usable_endpoints']+=usable and not (len(previous['frames'])>=6 and previous['continuous_crossing']);summary['endpoints_with_bank_recovery']+=bool(accepted_bank)
    trace.append({'event_id':eid,'linked_nodes':packs[-1]['linked_local_nodes'],'accepted_bank_views':len(accepted_bank),'clean_templates':len(selected),'sources':sources})
used=sorted(set(r['key'] for r in new_samples));np.savez_compressed(HERE/'clean_features.npz',keys=np.asarray(used),cope=np.stack([all_features[k][0] for k in used]),sapiens=np.stack([all_features[k][1] for k in used]))
(HERE/'events_recovered.json').write_text(json.dumps(events));(HERE/'results/clean_feature_samples.json').write_text(json.dumps(new_samples));(HERE/'results/evidence_manifest.json').write_text(json.dumps({'events':packs,'historical_records_used':False,'raw_local_tracking_window_seconds':6,'minimum_clean_views':3}))
(HERE/'results/template_recovery_trace.json').write_text(json.dumps(trace));summary.update({'templates':len(new_samples),'unique_crops':len(used),'elapsed_seconds':round(time.time()-started,1),'historical_records_used':False})
(HERE/'results/template_recovery_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
