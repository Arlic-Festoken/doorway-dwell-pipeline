"""Use returned labels and accepted-pair templates to mine new review examples."""
from collections import Counter,defaultdict
from datetime import datetime
import json,os,time,pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results';PREV=ROOT/'doorway_recall_20261109';REC=ROOT/'doorway_recovery_20261108';TEMP=ROOT/'doorway_temporal_20261109';QUERY=ROOT/'doorway_pairer_20261108/query_round'
start=time.time();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
ledger={'stage':'feedback_and_seed_mining','pid':os.getpid(),'cwd':str(HERE),'gpu':0,'started_at':datetime.now().isoformat(),'output':str(OUT),'stop_condition':'mining complete or timeout 1200 seconds'}
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(ledger)+'\n')
events=json.loads((TEMP/'events.json').read_text());ranks=np.load(PREV/'results/retrieval_ranks.npz');entries=ranks['entries'].tolist();exits=ranks['exits'].tolist();ei={e:i for i,e in enumerate(entries)};oi={e:i for i,e in enumerate(exits)};shape=(len(entries),len(exits))
judgments=json.loads((OUT/'judgments.json').read_text());rows=[json.loads(line) for line in (PREV/'results/全部候选.jsonl').open()];by_pair={r['pair_id']:r for r in rows}
proposals=json.loads((OUT/'new_entry_proposals.json').read_text());active=sorted(set(entries+exits+[r['event_id'] for r in proposals]));index={e:i for i,e in enumerate(active)}
for r in proposals:events[r['event_id']]={**events[r['event_id']],**{k:v for k,v in r.items() if k!='measurements'}}
samples=json.loads((REC/'results/clean_feature_samples.json').read_text());z=np.load(REC/'clean_features.npz');keys={str(k):i for i,k in enumerate(z['keys'])};cf=list(z['cope'].astype(np.float32));sf=list(z['sapiens'].astype(np.float32));by_event=defaultdict(list)
for r in samples:by_event[r['event_id']].append(keys[r['key']])
clean_counts={e:len(by_event[e]) for e in active}
for proposal in proposals:
    eid=proposal['event_id'];clip=proposal['clip'];measure=proposal['measurements'];cached=json.loads((QUERY/'sequences'/f'clip_{clip:03d}.json').read_text())['crops'];z=np.load(QUERY/'features'/f'clip_{clip:03d}.npz');lookup={str(k):i for i,k in enumerate(z['keys'])}
    for r in cached:
        if r['event_id']!=eid or r['quality']<.25 or min(abs(r['time']-m['time']) for m in measure)>.08:continue
        k=lookup[r['path']];by_event[eid].append(len(cf));cf.append(z['cope'][k].astype(np.float32));sf.append(z['sapiens'][k].astype(np.float32))
    clean_counts[eid]=len(by_event[eid])
anchors=json.loads((PREV/'results/anchor_sources.json').read_text());anchor_by_key={r['key']:r for r in anchors}
for path in sorted((PREV/'features').glob('clip_*.npz')):
    z=np.load(path)
    for k,c in zip(z['keys'],z['features']):
        e=anchor_by_key[str(k)]['event_id'];by_event[e].append(len(cf));cf.append(c.astype(np.float32));sf.append(np.zeros(4096,np.float32))
cf=np.stack(cf);sf=np.stack(sf);cf/=np.linalg.norm(cf,axis=1,keepdims=True).clip(1e-8);sf/=np.linalg.norm(sf,axis=1,keepdims=True).clip(1e-8);has_s=np.linalg.norm(sf,axis=1)>.1
with torch.inference_mode():
    a=torch.tensor(cf,device='cuda');cm=(a@a.T).cpu().numpy();del a
    a=torch.tensor(sf,device='cuda');sm=(a@a.T).cpu().numpy();del a
M=len(active);cmax=np.full((M,M),np.nan,np.float32);ccov=cmax.copy();scov=cmax.copy()
for i,e in enumerate(active):
    a=by_event[e]
    if not a:continue
    for j in range(i,M):
        b=by_event[active[j]]
        if not b:continue
        m=cm[np.ix_(a,b)];cmax[i,j]=cmax[j,i]=m.max();ccov[i,j]=ccov[j,i]=.5*(np.sort(m.max(1))[-min(3,len(a)):].mean()+np.sort(m.max(0))[-min(3,len(b)):].mean())
        sa=[k for k in a if has_s[k]];sb=[k for k in b if has_s[k]]
        if sa and sb:
            m=sm[np.ix_(sa,sb)];scov[i,j]=scov[j,i]=.5*(np.sort(m.max(1))[-min(3,len(sa)):].mean()+np.sort(m.max(0))[-min(3,len(sb)):].mean())
del cm,sm,cf,sf
np.savez_compressed(OUT/'endpoint_similarities.npz',event_ids=active,cope_best=cmax,cope_coverage=ccov,sapiens_coverage=scov,clean_counts=[clean_counts[e] for e in active])
print(json.dumps({'stage':'endpoint_similarities_complete','endpoints':M,'seconds':time.time()-start}),flush=True)
channels={}
for name,path in [('A_multiview',REC/'variants/A_multiview/pair_probabilities.npz'),('B_parts',REC/'variants/B_parts/pair_probabilities.npz'),('C_scene',REC/'variants/F_scene_adapt/pair_probabilities.npz'),('F_dense',TEMP/'variants/F_dense_static/pair_probabilities.npz'),('G_temporal',TEMP/'variants/G_temporal/pair_probabilities.npz'),('H_ordered',TEMP/'variants/H_alignment/pair_probabilities.npz'),('I_fusion',TEMP/'variants/I_fusion/pair_probabilities.npz')]:
    z=np.load(path);s=np.full(shape,np.nan);a=[ei[int(e)] for e in z['entries']];b=[oi[int(e)] for e in z['exits']];s[np.ix_(a,b)]=np.where(z['legal'],z['probability'],np.nan);channels[name]=s
ix=np.ix_([index[e] for e in entries],[index[e] for e in exits]);channels.update(COPE_best_view=cmax[ix],COPE_multiple_views=ccov[ix],Sapiens_views=scov[ix])
z=np.load(TEMP/'results/pair_feature_vectors.npz');temporal=np.full((*shape,16),np.nan);temporal[z['owners'][:,0],z['owners'][:,1]]=z['features'];channels['DTW_alignment']=temporal[:,:,14]
features=[];names=[]
for name,s in channels.items():
    f=ranks[name+'_forward'];r=ranks[name+'_reverse'];features.extend([np.nan_to_num(s),np.isfinite(s).astype(float),np.where(f<999999,1/(1+np.log1p(f)),0),np.where(r<999999,1/(1+np.log1p(r)),0)]);names.extend([name+'_score',name+'_available',name+'_forward',name+'_reverse'])
for k in range(16):features.extend([np.nan_to_num(temporal[:,:,k]),np.isfinite(temporal[:,:,k]).astype(float)]);names.extend([f'temporal_{k}',f'temporal_{k}_available'])
features.extend([np.broadcast_to(np.log1p([clean_counts[e] for e in entries])[:,None],shape),np.broadcast_to(np.log1p([clean_counts[e] for e in exits])[None,:],shape)]);names+=['entry_clean_count_log','exit_clean_count_log'];X=np.stack(features,-1).reshape(-1,len(features))
train=[];labels=[];train_keys=[]
for key,v in judgments.items():
    if set(v) not in ({'0'},{'1'}) or key not in by_pair:continue
    r=by_pair[key];train.append(ei[r['entry_event']]*len(exits)+oi[r['exit_event']]);labels.append(int(v[0]));train_keys.append(key)
train=np.array(train);y=np.array(labels);weights=np.where(y==1,15.,1.);scaler=StandardScaler().fit(X[train]);lr=LogisticRegression(C=.025,max_iter=1000).fit(scaler.transform(X[train]),y,sample_weight=weights)
hgb=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=7,max_depth=3,min_samples_leaf=30,l2_regularization=12,learning_rate=.06,early_stopping=False,random_state=0).fit(X[train],y,sample_weight=weights)
sl=lr.predict_proba(scaler.transform(X))[:,1];sh=hgb.predict_proba(X)[:,1];score=np.sqrt(sl*sh)
# These are screening cutoffs, not independently measured precision estimates.
cut_lr=float(max(np.quantile(sl[train[y==1]],.2),np.quantile(sl[train[y==0]],.985)))
cut_hgb=float(max(np.quantile(sh[train[y==1]],.2),np.quantile(sh[train[y==0]],.985)))
recall_lr=float(max(np.quantile(sl[train[y==1]],.05),np.quantile(sl[train[y==0]],.95)))
recall_hgb=float(max(np.quantile(sh[train[y==1]],.05),np.quantile(sh[train[y==0]],.95)))
np.savez_compressed(OUT/'feedback_scores.npz',entries=entries,exits=exits,linear=sl.reshape(shape),tree=sh.reshape(shape),combined=score.reshape(shape),train_indices=train,train_labels=y)
positive_idx=np.array([[index[by_pair[k]['entry_event']],index[by_pair[k]['exit_event']]] for k,v in judgments.items() if set(v)=={'1'} and k in by_pair])
negative_idx=np.array([[index[by_pair[k]['entry_event']],index[by_pair[k]['exit_event']]] for k,v in judgments.items() if set(v)=={'0'} and k in by_pair])
combined=np.where(np.isfinite(scov),.65*ccov+.35*scov,ccov)
pv=combined[positive_idx[:,0],positive_idx[:,1]];nv=combined[negative_idx[:,0],negative_idx[:,1]]
appearance_cut=float(max(np.nanquantile(pv,.10),np.nanquantile(nv,.90)))
seed_cut=float(max(np.nanquantile(pv,.35),np.nanquantile(nv,.97)))
seeds=[]
for key,v in judgments.items():
    if set(v)!={'1'} or key not in by_pair:continue
    r=by_pair[key];a=r['entry_event'];b=r['exit_event']
    if min(clean_counts[a],clean_counts[b])>=3:seeds.append(r)
qseed=[]
for r in seeds:
    a=combined[:,index[r['entry_event']]];b=combined[:,index[r['exit_event']]];qseed.append(.7*np.maximum(a,b)+.3*np.minimum(a,b))
qseed=np.stack(qseed,1);selected=[];reasons=Counter();saved=[]
preserved=json.loads((OUT/'positive_union.json').read_text());main=[r for r in preserved if r['included_in_base_preserving_merge']]
for r in rows:
    key=r['pair_id'];a=r['entry_event'];b=r['exit_event'];flat=ei[a]*len(exits)+oi[b];clean=min(clean_counts[a],clean_counts[b]);direct=float(combined[index[a],index[b]])
    if key in judgments:continue
    support=[]
    if clean>=2 and direct>=appearance_cut:
        support_scores=np.minimum(qseed[index[a]],qseed[index[b]])
        for si in np.argsort(-np.nan_to_num(support_scores,nan=-1))[:5]:
            s=seeds[si];shared=bool({a,b}&{s['entry_event'],s['exit_event']});disjoint=r['exit_seconds']<=s['entry_seconds'] or r['entry_seconds']>=s['exit_seconds']
            if support_scores[si]<seed_cut or not (shared or disjoint):continue
            blocked=False
            for x in (a,b):
                for t in (s['entry_event'],s['exit_event']):
                    if events[x]['direction']==events[t]['direction']:continue
                    en,ex=(x,t) if events[x]['direction']=='entry' else (t,x);label=judgments.get(f'I{en:04d}_O{ex:04d}')
                    if label and set(label)=={'0'}:blocked=True
            if not blocked:support.append({'seed_pair_id':s['pair_id'],'score':float(support_scores[si]),'shared_endpoint':shared})
    why=[]
    if clean>=2 and direct>=appearance_cut and sl[flat]>=cut_lr and sh[flat]>=cut_hgb:why.append('feedback_two_models')
    elif clean>=2 and direct>=appearance_cut and sl[flat]>=recall_lr and sh[flat]>=recall_hgb:why.append('feedback_recall')
    if support and sl[flat]>=cut_lr*.6 and sh[flat]>=cut_hgb*.6:why.append('accepted_pair_template')
    if not why:continue
    row={**r,'candidate_sources':why,'feedback_linear_score':float(sl[flat]),'feedback_tree_score':float(sh[flat]),'feedback_sort_score':float(score[flat]),'appearance_evidence_score':direct,'seed_support':support,'entry_clean_frames':clean_counts[a],'exit_clean_frames':clean_counts[b],'new_entry_proposal':False,'model_score_is_probability':False,'existing_review':'','existing_review_sources':[]}
    row['competes_with_main']=[s['pair_id'] for s in main if {a,b}&set(s['event_chain'])];selected.append(row);reasons.update(why)
# Newly recovered entrances have no legacy channel scores. Require matching
# multi-view evidence above the same appearance floor before proposing exits.
new_stats=[]
for p in proposals:
    eid=p['event_id'];candidates=[]
    for b in exits:
        if events[b]['time']-p['time']<30 or min(clean_counts[eid],clean_counts[b])<2:continue
        i=index[eid];j=index[b];direct=float(combined[i,j]);cs=float(ccov[i,j]);ss=float(scov[i,j])
        if not np.isfinite(direct) or direct<appearance_cut:continue
        candidates.append((direct,b,cs,ss))
    candidates.sort(reverse=True);new_stats.append({'event_id':eid,'clean_templates':clean_counts[eid],'above_appearance_floor':len(candidates),'exported':min(3,len(candidates))})
    for rank,(direct,b,cs,ss) in enumerate(candidates[:3],1):
        selected.append({'pair_id':f'I{eid:04d}_O{b:04d}','entry_event':eid,'exit_event':b,'entry_seconds':p['time'],'exit_seconds':events[b]['time'],'duration_seconds':events[b]['time']-p['time'],'entry_interval':p['interval'],'exit_interval':events[b]['interval'],'candidate_sources':['new_entry_sequence'],'appearance_evidence_score':direct,'cope_coverage':cs,'sapiens_coverage':ss,'new_entry_exit_rank':rank,'entry_clean_frames':clean_counts[eid],'exit_clean_frames':clean_counts[b],'new_entry_proposal':True,'seed_support':[],'feedback_sort_score':0.,'identity_accepted':False,'existing_review':'','competes_with_main':[s['pair_id'] for s in main if b in s['event_chain']]});reasons['new_entry_sequence']+=1
selected.sort(key=lambda r:(r['new_entry_proposal'],-r['feedback_sort_score'],-r['appearance_evidence_score']))
for n,r in enumerate(selected,1):r['review_index']=n
groups={'反馈双模型筛选':[r for r in selected if 'feedback_two_models' in r['candidate_sources']],'反馈扩展筛选':[r for r in selected if 'feedback_recall' in r['candidate_sources']],'判1人物扩展':[r for r in selected if 'accepted_pair_template' in r['candidate_sources']],'补入口候选':[r for r in selected if r['new_entry_proposal']]}
(OUT/'new_candidates.json').write_text(json.dumps(selected,ensure_ascii=False));(OUT/'priority_workbook.json').write_text(json.dumps({'rows':selected},ensure_ascii=False));(OUT/'candidate_groups.json').write_text(json.dumps({'groups':groups},ensure_ascii=False))
with (OUT/'feedback_models.pkl').open('wb') as f:pickle.dump({'scaler':scaler,'lr':lr,'hgb':hgb,'names':names,'train_keys':train_keys},f)
report={'new_candidates':len(selected),'source_counts':dict(reasons),'candidate_group_sizes':{k:len(v) for k,v in groups.items()},'already_submitted_but_blank':sum(r['pair_id'] in {x['pair_id'] for x in rows if x['tier']==1} for r in selected),'clean_training_label_counts':dict(Counter(labels)),'positive_seed_pairs':len(seeds),'cutoffs':{'linear':cut_lr,'tree':cut_hgb,'recall_linear':recall_lr,'recall_tree':recall_hgb,'appearance':appearance_cut,'seed_similarity':seed_cut},'new_entry_matching':new_stats,'competing_with_main':sum(bool(r['competes_with_main']) for r in selected),'old_reviewed_pairs_resubmitted':0,'historical_12_used':False,'new_model_review_performed':False,'independent_accuracy_measured':False,'elapsed_seconds':time.time()-start}
(OUT/'mining_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps({**ledger,'stage':'feedback_and_seed_mining_complete','elapsed_seconds':time.time()-start})+'\n')
