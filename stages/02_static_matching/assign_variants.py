"""Run explicit quality/coverage and companion alternatives on fixed pair evidence."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;events=json.loads((HERE/'events_recovered.json').read_text());pack={r['event_id']:r for r in json.loads((HERE/'results/evidence_manifest.json').read_text())['events']}
z=np.load(HERE/'variants/B_parts/pair_probabilities.npz');entries=z['entries'].astype(int).tolist();exits=z['exits'].astype(int).tolist();prob=z['probability'];legal=z['legal'];rank_a=z['rank_forward'];rank_b=z['rank_reverse'];raw=z['raw_score']
baseline=json.loads((HERE/'variants/B_parts/event_dispositions.json').read_text());previous=json.loads((ROOT/'doorway_balance_20261108/results/delivery_records.json').read_text());unverified=[r for r in previous['candidates'] if not r['crossing_confirmed']]
returns=baseline['return_edges'];active=sorted(pack);anchors=[r for r in baseline['main_edges'] if r['probability']>=.97 and r['global_alternative_gap']>=.5 and max(r['rank_forward'],r['rank_reverse'])<=3]
group_bonus=np.zeros(prob.shape);group_source={}
def near(a,b):
    ea=events[a];eb=events[b];ha=ea['crossing_box'][3]-ea['crossing_box'][1];hb=eb['crossing_box'][3]-eb['crossing_box'][1]
    return a!=b and abs(ea['time']-eb['time'])<=3 and np.linalg.norm(np.asarray(ea['crossing_foot'])-eb['crossing_foot'])<1.5*min(ha,hb)
for anchor in anchors:
    a0=anchor['left'];b0=anchor['right'];aa=[i for i,a in enumerate(entries) if near(a,a0)];bb=[j for j,b in enumerate(exits) if near(b,b0)]
    h0a=events[a0]['crossing_box'][3]-events[a0]['crossing_box'][1];h0b=events[b0]['crossing_box'][3]-events[b0]['crossing_box'][1]
    for i in aa:
        a=entries[i];ha=events[a]['crossing_box'][3]-events[a]['crossing_box'][1]
        for j in bb:
            b=exits[j]
            if not legal[i,j] or prob[i,j]<.65:continue
            hb=events[b]['crossing_box'][3]-events[b]['crossing_box'][1]
            if abs(np.log((ha/h0a)/(hb/h0b)))>.25:continue
            if abs(abs(events[a]['time']-events[a0]['time'])-abs(events[b]['time']-events[b0]['time']))>2:continue
            entry_offset=(events[a]['crossing_foot'][0]-events[a0]['crossing_foot'][0])/min(ha,h0a);exit_offset=(events[b]['crossing_foot'][0]-events[b0]['crossing_foot'][0])/min(hb,h0b)
            if abs(entry_offset-exit_offset)>.75:continue
            bonus=.25*float(logit(min(anchor['probability'],.999)))
            if bonus>group_bonus[i,j]:group_bonus[i,j]=bonus;group_source[(i,j)]=[a0,b0]

configs=[{'id':'C_coverage','name':'概率覆盖版','threshold':.75,'gap':.05,'companions':False},
         {'id':'D_companions','name':'同行辅助版','threshold':.85,'gap':.10,'companions':True},
         {'id':'E_exploration','name':'候选扩展版','threshold':.50,'gap':.01,'companions':False}]
(HERE/'results/variant_configurations.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2))
for config in configs:
    directory=HERE/'variants'/config['id'];directory.mkdir(exist_ok=True)
    bonus=group_bonus if config['companions'] else np.zeros(prob.shape)
    utility=logit(np.clip(prob,1e-6,1-1e-6))-logit(config['threshold'])+bonus
    allowed=legal&((rank_a<=20)|(rank_b<=20))&(utility>0)&((prob>=.65) if config['companions'] else True)
    utility=np.where(allowed,utility,-1e4);objective=np.c_[utility,np.zeros((len(entries),len(entries)))];rr,cc=linear_sum_assignment(objective,maximize=True);optimum=float(objective[rr,cc].sum());main=[];unstable=[]
    for a,b in zip(rr,cc):
        if b>=len(exits) or utility[a,b]<=0:continue
        value=objective[a,b];objective[a,b]=-1e4;ar,ac=linear_sum_assignment(objective,maximize=True);gap=optimum-float(objective[ar,ac].sum());objective[a,b]=value
        pair={'left':entries[a],'right':exits[b],'probability':float(prob[a,b]),'rank_forward':int(rank_a[a,b]),'rank_reverse':int(rank_b[a,b]),'global_alternative_gap':gap,'group_support_log_odds':float(bonus[a,b]),'group_anchor':group_source.get((a,b)) if config['companions'] else None,'method':config['id']}
        (main if gap>=config['gap'] else unstable).append(pair)
    links=main+returns;succ={r['left']:r['right'] for r in links};pred={r['right']:r['left'] for r in links};meta={(r['left'],r['right']):r for r in links};complete=[];short=[];unclosed=[];chains=[];open_chains=[]
    for start in sorted(active,key=lambda i:events[i]['time']):
        if start in pred:continue
        chain=[start]
        while chain[-1] in succ:chain.append(succ[chain[-1]])
        first=events[chain[0]];last=events[chain[-1]];closed=first['direction']=='entry' and last['direction']=='exit'
        row={'entry_seconds':first['time'] if first['direction']=='entry' else None,'exit_seconds':last['time'] if last['direction']=='exit' else None,'entry_event':chain[0] if first['direction']=='entry' else None,'exit_event':chain[-1] if last['direction']=='exit' else None,'entry_interval':first['interval'] if first['direction']=='entry' else None,'exit_interval':last['interval'] if last['direction']=='exit' else None,'appearance':'','event_chain':chain,'source_pass':'automatic_'+config['id']+'_20261108','association_links':[meta[a,b] for a,b in zip(chain[:-1],chain[1:])],'outside_interruptions':[{'exit':events[a]['time'],'reentry':events[b]['time'],'gap_seconds':events[b]['time']-events[a]['time']} for a,b in zip(chain[:-1],chain[1:]) if events[a]['direction']=='exit']}
        row['duration_seconds']=row['exit_seconds']-row['entry_seconds'] if closed else None
        if closed:(complete if row['duration_seconds']>=30 else short).append(row)
        else:
            open_chains.append(row)
            for i in ([chain[0]] if first['direction']=='exit' else [])+([chain[-1]] if last['direction']=='entry' else []):
                e=events[i];unclosed.append({'event_id':i,'entry_seconds':e['time'] if e['direction']=='entry' else None,'exit_seconds':e['time'] if e['direction']=='exit' else None,'direction':e['direction'],'interval':e['interval'],'crossing_confirmed':True,'appearance':'','evidence_status':pack[i]['status'],'chain':chain})
        chains.append(row)
    complete.sort(key=lambda r:r['entry_seconds']);candidates=sorted(unclosed+unverified,key=lambda r:r['entry_seconds'] if r['entry_seconds'] is not None else r['exit_seconds'])
    records={'mode':'automatic_doorway_recovery_20261108','variant':config,'confirmed':complete,'candidates':candidates,'unclosed':unclosed,'minimum_duration_seconds':30,'maximum_merged_outside_gap_seconds':30,'historical_records_used':False,'automatic_records_are_independently_confirmed':False,'probabilities_calibrated_on_independent_visit_labels':False}
    report={'variant':config,'complete_visits':len(complete),'unmatched_endpoints':len(unclosed),'unmatched_directions':dict(Counter(r['direction'] for r in unclosed)),'short_visits_excluded':len(short),'store_visit_edges':len(main),'continuous_return_links':len(returns),'appearance_only_return_links':0,'ambiguous_global_edges_retained_unmatched':len(unstable),'companion_supported_pairs':sum(r['group_support_log_odds']>0 for r in main),'eligible_unique_entries':278,'eligible_unique_exits':472,'unverified_candidates_retained':len(unverified),'accuracy_evaluation_performed':False,'historical_records_used':False,'count_target_used':False}
    (directory/'delivery_records.json').write_text(json.dumps(records,ensure_ascii=False));(directory/'delivery_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(directory/'event_dispositions.json').write_text(json.dumps({'main_edges':main,'return_edges':returns,'open_chains':open_chains,'chains':chains,'short_visits':short,'unstable_edges':unstable}));print(json.dumps(report,ensure_ascii=False),flush=True)
