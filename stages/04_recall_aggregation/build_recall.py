"""High recall retrieval. Alternatives remain alternatives until user review."""
from collections import Counter,defaultdict
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REC=ROOT/'doorway_recovery_20261108';TEMP=ROOT/'doorway_temporal_20261109';started=time.time()
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
ledger={'stage':'high_recall_retrieval','pid':os.getpid(),'cwd':str(HERE),'gpu':0,'started_at':datetime.now().isoformat(),'output':str(HERE/'results'),'stop_condition':'candidate exports complete or 1200 seconds'}
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(ledger)+'\n')
events=json.loads((HERE/'events.json').read_text());packs={r['event_id']:r for r in json.loads((REC/'results/evidence_manifest.json').read_text())['events']}
active=sorted(packs);entries=[i for i in active if events[i]['direction']=='entry'];exits=[i for i in active if events[i]['direction']=='exit'];ei={e:i for i,e in enumerate(entries)};oi={e:i for i,e in enumerate(exits)};shape=(len(entries),len(exits))
duration=np.array([events[b]['time'] for b in exits])[None,:]-np.array([events[a]['time'] for a in entries])[:,None]
legal=duration>0;long=duration>=30
review_rows=list(csv.DictReader((HERE/'results/已有审核快照.csv').open(encoding='utf-8-sig')));static_reviews=list(csv.DictReader((HERE/'results/旧五方案审核快照.csv').open(encoding='utf-8-sig')));review_sources=defaultdict(list)
for source,data in [('temporal',review_rows),('static',static_reviews)]:
    for r in data:review_sources[r['配对编号']].append({'source':source,'label':r['审核结果_0异人_1同人_2不足'],'note':r.get('备注','')})
review={}
for key,judgments in review_sources.items():
    labels={j['label'] for j in judgments};chosen='1' if '1' in labels else '2' if '2' in labels else '0'
    review[key]={'配对编号':key,'审核结果_0异人_1同人_2不足':chosen,'备注':'；'.join(j['note'] for j in judgments if j['note']),'source_judgments':judgments,'judgment_disagreement':len(labels)>1}
previous={r['pair_id']:r for r in json.loads((HERE/'results/previous_pair_manifest.json').read_text())}
channels={};provenance={}
for label,path in [('A_multiview',REC/'variants/A_multiview/pair_probabilities.npz'),('B_parts',REC/'variants/B_parts/pair_probabilities.npz'),('C_scene',REC/'variants/F_scene_adapt/pair_probabilities.npz'),('F_dense',TEMP/'variants/F_dense_static/pair_probabilities.npz'),('G_temporal',TEMP/'variants/G_temporal/pair_probabilities.npz'),('H_ordered',TEMP/'variants/H_alignment/pair_probabilities.npz'),('I_fusion',TEMP/'variants/I_fusion/pair_probabilities.npz')]:
    z=np.load(path);scores=np.full(shape,np.nan);aa=np.array([ei[int(i)] for i in z['entries']]);bb=np.array([oi[int(i)] for i in z['exits']]);scores[np.ix_(aa,bb)]=np.where(z['legal'],z['probability'],np.nan);channels[label]=scores;provenance[label]=str(path)

# Reuse even one or two clean views, which previous assignment stages excluded.
# For 101 endpoints without a clean view, add ONE explicitly unvalidated
# crossing-anchor crop. These frames are retrieval proposals, not ID evidence.
samples=json.loads((REC/'results/clean_feature_samples.json').read_text());z=np.load(REC/'clean_features.npz');keys={str(k):i for i,k in enumerate(z['keys'])};cf=list(z['cope'].astype(np.float32));sf=list(z['sapiens'].astype(np.float32));by_event=defaultdict(list)
for r in samples:by_event[r['event_id']].append(keys[r['key']])
anchors=json.loads((HERE/'results/anchor_sources.json').read_text());anchor_by_key={r['key']:r for r in anchors}
for path in sorted((HERE/'features').glob('clip_*.npz')):
    z=np.load(path)
    for key,feature in zip(z['keys'],z['features']):
        row=anchor_by_key[str(key)];by_event[row['event_id']].append(len(cf));cf.append(feature.astype(np.float32));sf.append(np.zeros(4096,np.float32))
cf=np.stack(cf);sf=np.stack(sf);cf/=np.linalg.norm(cf,axis=1,keepdims=True).clip(1e-8);sf/=np.linalg.norm(sf,axis=1,keepdims=True).clip(1e-8);has_s=np.linalg.norm(sf,axis=1)>.1
with torch.inference_mode():
    c=torch.tensor(cf,device='cuda');cm=(c@c.T).cpu().numpy();del c
    s=torch.tensor(sf,device='cuda');sm=(s@s.T).cpu().numpy();del s
raw_max=np.full(shape,np.nan);raw_coverage=raw_max.copy();sapiens=raw_max.copy()
for ai,a in enumerate(entries):
    av=by_event[a]
    if not av:continue
    for bi,b in enumerate(exits):
        bv=by_event[b]
        if not bv or not legal[ai,bi]:continue
        m=cm[np.ix_(av,bv)];raw_max[ai,bi]=m.max();raw_coverage[ai,bi]=.5*(np.sort(m.max(1))[-min(3,len(av)):].mean()+np.sort(m.max(0))[-min(3,len(bv)):].mean())
        sa=[i for i in av if has_s[i]];sb=[i for i in bv if has_s[i]]
        if sa and sb:
            m=sm[np.ix_(sa,sb)];sapiens[ai,bi]=.5*(np.sort(m.max(1))[-min(3,len(sa)):].mean()+np.sort(m.max(0))[-min(3,len(sb)):].mean())
channels.update(COPE_best_view=raw_max,COPE_multiple_views=raw_coverage,Sapiens_views=sapiens)
del cm,sm,cf,sf

# The old pair classifiers can down-rank a useful channel. Retrieve by its
# raw temporal alignment score as an independent source as well.
raw=np.load(TEMP/'results/pair_feature_vectors.npz');owners=raw['owners'];features=raw['features'];dtw=np.full(shape,np.nan)
for pair,v in zip(owners,features):
    a,b=map(int,pair);dtw[a,b]=max(dtw[a,b],v[14]) if np.isfinite(dtw[a,b]) else v[14]
channels['DTW_alignment']=dtw

geometry_conflicts=set()
def separated(a,b):
    aa=packs[a]['frames'];bb=packs[b]['frames']
    if not aa or not bb:return False
    if min(f['time'] for f in bb)-max(f['time'] for f in aa)>.075 or min(f['time'] for f in aa)-max(f['time'] for f in bb)>.075:return False
    for x in aa:
        for y in bb:
            if abs(x['time']-y['time'])>.075:continue
            u=np.array(x['box']);v=np.array(y['box']);inter=np.maximum(0,np.minimum(u[2:],v[2:])-np.maximum(u[:2],v[:2])).prod();iou=inter/((u[2:]-u[:2]).prod()+(v[2:]-v[:2]).prod()-inter)
            if iou<.01 and np.linalg.norm((u[:2]+u[2:]-v[:2]-v[2:])/2)>.5*min(u[3]-u[1],v[3]-v[1]):return True
    return False
for ai,a in enumerate(entries):
    for bi,b in enumerate(exits):
        if legal[ai,bi] and separated(a,b):legal[ai,bi]=False;geometry_conflicts.add((a,b))
available=np.zeros(shape,bool);priority=np.zeros(shape);ranks={};best_rank=np.full(shape,999999,int)
for name,scores in channels.items():
    valid=legal&np.isfinite(scores);available|=valid
    # Main ranking covers >=30 seconds. Short alternatives are exported too,
    # so review can later consume their endpoints before final visit export.
    scores=np.where(valid&long,scores,-np.inf);f=np.full(shape,999999,int);r=f.copy()
    for a in range(len(entries)):
        idx=np.flatnonzero(np.isfinite(scores[a]));idx=idx[np.argsort(-scores[a,idx],kind='stable')];f[a,idx]=np.arange(len(idx))+1
    for b in range(len(exits)):
        idx=np.flatnonzero(np.isfinite(scores[:,b]));idx=idx[np.argsort(-scores[idx,b],kind='stable')];r[idx,b]=np.arange(len(idx))+1
    ranks[name]=(f,r);best_rank=np.minimum(best_rank,np.minimum(f,r));priority+=np.where(f<999999,1/(60+f),0)+np.where(r<999999,1/(60+r),0)
video=json.loads((HERE/'results/video_manifest.json').read_text());anchor_ids={r['event_id'] for r in anchors};endpoint_evidence={}
for eid in active:
    p=packs[eid];v=video[str(eid)];views=[]
    for row in p['frames']:
        for field,folder in [('crop','crops'),('context','contexts')]:
            source=REC/row[field];target=HERE/row[field]
            if not target.exists():target.symlink_to(source)
        views.append({'cell':row['cell'],'time':row['time'],'crop':row['crop'],'context':row['context'],'fragment':row['fragment']})
    if eid in anchor_ids:
        anchor=next(r for r in anchors if r['event_id']==eid);views.append({'cell':'ANCHOR','time':anchor['time'],'crop':anchor['path'],'context':f'contexts/anchor_{eid:04d}.jpg','fragment':0,'identity_sequence_confirmed':False})
    endpoint_evidence[eid]={'event_id':eid,'direction':events[eid]['direction'],'time':events[eid]['time'],'interval':events[eid]['interval'],'views':views,'videos':v['videos'],'panel':v['panel'],'source':'unvalidated_crossing_anchor' if eid in anchor_ids else 'clean_local_views','template_count':len(by_event[eid])}

def stamp(t):
    t=round(t);return f'{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}'
def stamped(t,interval):return ('约 ' if interval[1]-interval[0]>.6 else '')+'2026-01-26 '+stamp(t)
# Learn only review order, using unambiguous existing 0/1 labels. IDs, clock
# time and dwell duration are not inputs. No candidate is deleted by this model.
features=[];feature_names=[]
for name,(forward,reverse) in ranks.items():
    value=channels[name];present=np.isfinite(value)
    features.extend([np.where(present,value,0),present.astype(float),np.where(forward<999999,1/(1+np.log1p(forward)),0),np.where(reverse<999999,1/(1+np.log1p(reverse)),0)])
    feature_names.extend([name+'_score',name+'_available',name+'_forward_rank_feature',name+'_reverse_rank_feature'])
feedback_features=np.stack(features,-1).reshape(-1,len(features));train_idx=[];train_labels=[]
for key,judgments in review_sources.items():
    labels={j['label'] for j in judgments}
    if labels not in ({'0'},{'1'}):continue
    a,b=map(int,[key.split('_')[0][1:],key.split('_')[1][1:]])
    if a not in ei or b not in oi or not (available[ei[a],oi[b]] and long[ei[a],oi[b]]):continue
    train_idx.append(ei[a]*len(exits)+oi[b]);train_labels.append(int(next(iter(labels))))
scaler=StandardScaler().fit(feedback_features[train_idx]);reranker=LogisticRegression(C=.1,class_weight='balanced',max_iter=500).fit(scaler.transform(feedback_features[train_idx]),train_labels)
feedback_priority=reranker.predict_proba(scaler.transform(feedback_features))[:,1].reshape(shape)
(HERE/'results/反馈排序模型.json').write_text(json.dumps({'features':feature_names,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficient':reranker.coef_.tolist(),'intercept':reranker.intercept_.tolist(),'label_counts':dict(Counter(train_labels)),'use':'review_order_only','deletes_candidates':False,'independent_accuracy_measured':False,'historical_12_used':False},ensure_ascii=False,indent=2))
rows=[]
for ai,a in enumerate(entries):
    for bi,b in enumerate(exits):
        if not available[ai,bi] or not long[ai,bi]:continue
        key=f'I{a:04d}_O{b:04d}';old=review.get(key,{});label=old.get('审核结果_0异人_1同人_2不足','');support=[]
        for name,(f,r) in ranks.items():
            if min(f[ai,bi],r[ai,bi])<=30:support.append({'source':name,'entry_to_exit_rank':int(f[ai,bi]),'exit_to_entry_rank':int(r[ai,bi])})
        rank=int(best_rank[ai,bi]);tier=1 if rank<=3 else 2 if rank<=10 else 3 if rank<=30 else 4
        rows.append({'pair_id':key,'entry_event':a,'exit_event':b,'entry_seconds':events[a]['time'],'exit_seconds':events[b]['time'],'duration_seconds':float(duration[ai,bi]),'entry_interval':events[a]['interval'],'exit_interval':events[b]['interval'],'tier':tier,'best_directional_rank':rank,'retrieval_priority':float(priority[ai,bi]),'feedback_priority':float(feedback_priority[ai,bi]),'retrieval_sources':support,'existing_review':label,'existing_review_sources':old.get('source_judgments',[]),'judgment_disagreement':old.get('judgment_disagreement',False),'existing_review_note':old.get('备注',''),'appearance':'','endpoint_competition_groups':[f'I{a:04d}',f'O{b:04d}'],'entry_evidence':f'endpoints/{a:04d}.json','exit_evidence':f'endpoints/{b:04d}.json','identity_accepted':False,'contains_unvalidated_anchor':a in anchor_ids or b in anchor_ids})
rows.sort(key=lambda r:(r['tier'],-r['feedback_priority'],-r['retrieval_priority'],r['entry_seconds'],r['exit_seconds']))
for i,r in enumerate(rows,1):r['candidate_index']=i
sets={'01_优先候选_双向Top3':[r for r in rows if r['tier']==1],'02_扩展候选_双向Top10':[r for r in rows if r['tier']<=2],'03_宽召回候选_双向Top30':[r for r in rows if r['tier']<=3],'04_全部可检索组合':rows}
for name,data in sets.items():
    with (HERE/'results'/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['进入时间','出店时间','停留时间'])
        for r in data:w.writerow([stamped(r['entry_seconds'],r['entry_interval']),stamped(r['exit_seconds'],r['exit_interval']),('约 ' if max(r['entry_interval'][1]-r['entry_interval'][0],r['exit_interval'][1]-r['exit_interval'][0])>.6 else '')+stamp(r['duration_seconds'])])
    for excel_row,r in enumerate(data,2):r.setdefault('dataset_rows',{})[name]=excel_row
pending=[r for r in rows if r['existing_review']==''];insufficient=[r for r in rows if r['existing_review']=='2'];already_same=[r for r in rows if r['existing_review']=='1']
out=HERE/'results/endpoints';out.mkdir(exist_ok=True)
for eid,evidence in endpoint_evidence.items():(out/f'{eid:04d}.json').write_text(json.dumps(evidence,ensure_ascii=False))
for name,data in [('全部候选',rows),('仅新增待审核',pending),('已有判同',already_same),('已有证据不足',insufficient)]:
    with (HERE/'results'/f'{name}.jsonl').open('w') as f:
        for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (HERE/'results'/f'{name}_审核表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['配对编号','进入时间','出店时间','停留时间','召回层级','最佳单向排名','已有审核_0异人_1同人_2不足','新审核_0异人_1同人_2不足','备注','入口证据','出口证据'])
        for r in data:w.writerow([r['pair_id'],stamped(r['entry_seconds'],r['entry_interval']),stamped(r['exit_seconds'],r['exit_interval']),stamp(r['duration_seconds']),r['tier'],r['best_directional_rank'],r['existing_review'],'',r['existing_review_note'],r['entry_evidence'],r['exit_evidence']])
with (HERE/'results/不足30秒的竞争组合.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['配对编号','进入时间','出店时间','停留秒数','是否出现在主停留表'])
    for ai,bi in zip(*np.where(available&~long)):
        a=entries[ai];b=exits[bi];w.writerow([f'I{a:04d}_O{b:04d}',stamped(events[a]['time'],events[a]['interval']),stamped(events[b]['time'],events[b]['interval']),float(duration[ai,bi]),'否'])
with (HERE/'results/无候选端点.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['事件编号','方向','时间','原因'])
    covered={r['entry_event'] for r in rows}|{r['exit_event'] for r in rows}
    for eid in active:
        if eid not in covered:w.writerow([eid,events[eid]['direction'],stamp(events[eid]['time']),'视频内没有满足时间先后和30秒条件的可检索另一端'])
report={'stage':'high_recall_candidates_complete','eligible_entries':len(entries),'eligible_exits':len(exits),'retrieval_channels':list(channels),'candidate_pairs':len(rows),'tiers':{name:{'pairs':len(data),'new_unreviewed':sum(r['existing_review']=='' for r in data),'entry_coverage':len({r['entry_event'] for r in data}),'exit_coverage':len({r['exit_event'] for r in data})} for name,data in sets.items()},'existing_review_counts':dict(Counter(r['existing_review'] for r in rows if r['existing_review']!='')),'all_previous_reviewed_pairs_preserved':set(review)<=set(r['pair_id'] for r in rows),'unreviewed_pairs':len(pending),'unvalidated_anchor_endpoints':len(anchor_ids),'pairs_using_unvalidated_anchor':sum(r['contains_unvalidated_anchor'] for r in rows),'geometry_conflicts_excluded':len(geometry_conflicts),'short_competition_pairs':int(np.sum(available&~long)),'uncovered_endpoints':sorted(set(active)-covered),'one_to_one_assignment_performed':False,'new_model_review_performed':False,'historical_12_used':False,'elapsed_seconds':time.time()-started}
(HERE/'results/recall_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));np.savez_compressed(HERE/'results/retrieval_ranks.npz',entries=entries,exits=exits,available=available,priority=priority,best_rank=best_rank,**{name+'_forward':f for name,(f,r) in ranks.items()},**{name+'_reverse':r for name,(f,r) in ranks.items()})
# One compact workbook will contain the priority tier; broad tiers remain CSV
# and JSONL for inexpensive model batch review, without very large XLSX files.
(HERE/'results/priority_workbook.json').write_text(json.dumps({'rows':[r for r in pending if r['tier']==1]},ensure_ascii=False))
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps({**ledger,'stage':'high_recall_retrieval_complete','elapsed_seconds':time.time()-started})+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
