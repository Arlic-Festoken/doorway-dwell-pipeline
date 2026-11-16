"""Keep the earlier E48 base intact; union positive judgments by stable pair."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REC=ROOT/'doorway_recovery_20261108';TEMP=ROOT/'doorway_temporal_20261109';OUT=HERE/'results'
old_reviews={r['配对编号']:r for r in csv.DictReader((OUT/'旧五方案审核快照.csv').open(encoding='utf-8-sig'))};new_reviews={r['配对编号']:r for r in csv.DictReader((OUT/'已有审核快照.csv').open(encoding='utf-8-sig'))}
old=json.loads((REC/'results/unique_pair_manifest.json').read_text());new=json.loads((TEMP/'results/unique_pair_manifest.json').read_text())
positive={};judgments={}
for source,rows,labels in [('temporal',new,new_reviews),('static',old,old_reviews)]:
    for row in rows:
        key=row['pair_id'];label=labels[key]['审核结果_0异人_1同人_2不足'];judgments.setdefault(key,[]).append({'source':source,'label':label,'scheme_rows':row['scheme_rows']})
        if label=='1':positive[key]={k:v for k,v in row.items() if k not in ['entry_evidence','exit_evidence']}
base={r['pair_id']:positive[r['pair_id']] for r in old if old_reviews[r['pair_id']]['审核结果_0异人_1同人_2不足']=='1' and 'E' in r['scheme_rows']}
occupied={eid for r in base.values() for eid in r['event_chain']};extras={k:r for k,r in positive.items() if k not in base};blocked={k for k,r in extras.items() if occupied&set(r['event_chain'])};remaining={k:r for k,r in extras.items() if k not in blocked}
mutual={k for k,r in remaining.items() for q,s in remaining.items() if k!=q and set(r['event_chain'])&set(s['event_chain'])};additions={k:r for k,r in remaining.items() if k not in mutual};accepted={**base,**additions};conflicts=[]
for key in sorted(blocked|mutual):
    r=extras[key];competing=[k for k,s in positive.items() if k!=key and set(r['event_chain'])&set(s['event_chain'])]
    conflicts.append({'pair_id':key,'conflict_with':competing,'reason':'shares_endpoint_with_preserved_base' if key in blocked else 'mutually_competing_new_positive'})
for key,r in positive.items():
    r.update({'positive_judgments':judgments[key],'base_E48':key in base,'included_in_base_preserving_merge':key in accepted,'judgment_disagreement':len({j['label'] for j in judgments[key]})>1})
order=lambda r:(r['entry_seconds'],r['exit_seconds'])
groups={'保留基础的融合表':sorted(accepted.values(),key=order),'所有判1候选':sorted(positive.values(),key=order),'判1但端点冲突':sorted([positive[k] for k in blocked|mutual],key=order),'跨轮标签分歧':sorted([r for r in positive.values() if r['judgment_disagreement']],key=order)}
def stamp(t):
    t=round(t);return f'{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}'
def date(t,interval):return ('约 ' if interval[1]-interval[0]>.6 else '')+'2026-01-26 '+stamp(t)
for name,rows in groups.items():
    with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['进入时间','出店时间','停留时间'])
        for r in rows:w.writerow([date(r['entry_seconds'],r['entry_interval']),date(r['exit_seconds'],r['exit_interval']),('约 ' if max(r['entry_interval'][1]-r['entry_interval'][0],r['exit_interval'][1]-r['exit_interval'][0])>.6 else '')+stamp(r['duration_seconds'])])
with (OUT/'判1融合来源对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['配对编号','属于48条基础','合并进入主表','判定分歧','各轮原始标签','竞争配对'])
    for r in groups['所有判1候选']:w.writerow([r['pair_id'],int(r['base_E48']),int(r['included_in_base_preserving_merge']),int(r['judgment_disagreement']),json.dumps(r['positive_judgments'],ensure_ascii=False),'、'.join(next((c['conflict_with'] for c in conflicts if c['pair_id']==r['pair_id']),[]))])
(OUT/'positive_workbook.json').write_text(json.dumps({'groups':groups},ensure_ascii=False));(OUT/'positive_union.json').write_text(json.dumps(groups['所有判1候选'],ensure_ascii=False));(OUT/'positive_conflicts.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2))
summary={'base':'previous E variant judged 1','base_rows':len(base),'static_all_variants_positive':sum(r['审核结果_0异人_1同人_2不足']=='1' for r in old_reviews.values()),'all_rounds_positive_union':len(positive),'base_preserving_merged_rows':len(accepted),'new_noncompeting_positive_rows':len(additions),'competing_positive_candidates':len(conflicts),'conflict_with_base':len(blocked),'mutual_new_positive_conflicts':len(mutual),'judgment_disagreements_in_positive_union':sum(r['judgment_disagreement'] for r in positive.values()),'base_rows_removed':0,'judgments_are_model_labels_not_independent_truth':True}
(OUT/'positive_merge_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
