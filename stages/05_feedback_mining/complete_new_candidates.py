"""Keep every new-entrance match passing the evidence floor, including short competitors."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results';report=json.loads((OUT/'mining_summary.json').read_text());floor=report['cutoffs']['appearance'];z=np.load(OUT/'endpoint_similarities.npz');ids=z['event_ids'].tolist();lookup={e:i for i,e in enumerate(ids)};count={e:int(z['clean_counts'][i]) for i,e in enumerate(ids)};similarity=np.where(np.isfinite(z['sapiens_coverage']),.65*z['cope_coverage']+.35*z['sapiens_coverage'],z['cope_coverage'])
events=json.loads((ROOT/'doorway_temporal_20261109/events.json').read_text());exits=np.load(ROOT/'doorway_recall_20261109/results/retrieval_ranks.npz')['exits'];proposals=json.loads((OUT/'new_entry_proposals.json').read_text());main=[r for r in json.loads((OUT/'positive_union.json').read_text()) if r['included_in_base_preserving_merge']];rows=[r for r in json.loads((OUT/'new_candidates.json').read_text()) if not r['new_entry_proposal']];short=[];matching=[]
for p in proposals:
    eid=p['event_id'];matches=[]
    for b in exits:
        b=int(b);dt=events[b]['time']-p['time'];aidx=lookup[eid];bidx=lookup[b];score=float(similarity[aidx,bidx])
        if dt<=0 or min(count[eid],count[b])<2 or not np.isfinite(score) or score<floor:continue
        matches.append({'pair_id':f'I{eid:04d}_O{b:04d}','entry_event':eid,'exit_event':b,'entry_seconds':p['time'],'exit_seconds':events[b]['time'],'duration_seconds':dt,'entry_interval':p['interval'],'exit_interval':events[b]['interval'],'candidate_sources':['new_entry_sequence'],'new_entry_source':p['source'],'appearance_evidence_score':score,'cope_coverage':float(z['cope_coverage'][aidx,bidx]),'sapiens_coverage':float(z['sapiens_coverage'][aidx,bidx]),'entry_clean_frames':count[eid],'exit_clean_frames':count[b],'new_entry_proposal':True,'approximate_entry':p.get('inferred_foot_frames',0)>0,'seed_support':[],'feedback_sort_score':0.,'identity_accepted':False,'existing_review':'','competes_with_main':[s['pair_id'] for s in main if b in s['event_chain']]})
    matches.sort(key=lambda r:-r['appearance_evidence_score'])
    for n,r in enumerate(matches,1):r['new_entry_exit_rank']=n;(rows if r['duration_seconds']>=30 else short).append(r)
    matching.append({'event_id':eid,'clean_templates':count[eid],'retained_long_matches':sum(r['duration_seconds']>=30 for r in matches),'retained_short_competitors':sum(r['duration_seconds']<30 for r in matches)})
rows.sort(key=lambda r:(r['new_entry_proposal'],-r['feedback_sort_score'],-r['appearance_evidence_score']))
for n,r in enumerate(rows,1):r['review_index']=n
groups={'反馈及判1扩展':[r for r in rows if not r['new_entry_proposal']],'新入口待审核':[r for r in rows if r['new_entry_proposal']]}
report.update(new_candidates=len(rows),source_counts={'existing_endpoint_feedback_and_seed':len(groups['反馈及判1扩展']),'new_entry_sequence':len(groups['新入口待审核'])},candidate_group_sizes={k:len(v) for k,v in groups.items()},new_entry_matching=matching,new_entry_top_k_cap=None,new_entry_short_competitors=len(short),competing_with_main=sum(bool(r['competes_with_main']) for r in rows))
(OUT/'new_candidates.json').write_text(json.dumps(rows,ensure_ascii=False));(OUT/'priority_workbook.json').write_text(json.dumps({'rows':rows},ensure_ascii=False));(OUT/'candidate_groups.json').write_text(json.dumps({'groups':groups},ensure_ascii=False));(OUT/'short_competitors.json').write_text(json.dumps(short,ensure_ascii=False));(OUT/'mining_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'new_candidates':len(rows),'short_competitors':len(short),'new_entry_proposals':len(proposals),'matched_new_entries':sum(r['retained_long_matches']>0 for r in matching),'new_entry_matches':len(groups['新入口待审核'])}),flush=True)
