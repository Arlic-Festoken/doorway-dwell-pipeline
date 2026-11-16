"""Consume user-supplied labels; preserve the accepted earlier base."""
from collections import Counter
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PREV=ROOT/'doorway_recall_20261109';OUT=HERE/'results'
for name in ('results','logs','endpoints','media'):(HERE/name).mkdir(exist_ok=True)
rows=list(csv.DictReader((HERE/'新审核.csv').open(encoding='utf-8-sig')));column='新审核_0异人_1同人_2不足';labels={r['配对编号']:r[column].strip() for r in rows}
if len(labels)!=len(rows) or any(v not in ('','0','1','2') for v in labels.values()):raise ValueError('Duplicate pair IDs or unsupported review labels')
all_rows={r['pair_id']:r for r in map(json.loads,(PREV/'results/全部候选.jsonl').open())}
if not set(labels)<=set(all_rows):raise ValueError('Review contains IDs absent from the previous candidate manifest')
prior=json.loads((PREV/'results/positive_union.json').read_text());positive={r['pair_id']:r for r in prior};preserved={r['pair_id']:r for r in prior if r['included_in_base_preserving_merge']}
for key,label in labels.items():
    if label!='1':continue
    r=all_rows[key].copy();r['event_chain']=[r['entry_event'],r['exit_event']];r['positive_judgments']=[{'source':'user_returned_recall_round','label':'1'}];r['base_E48']=False;r['judgment_disagreement']=False;positive[key]=r
used={eid for r in preserved.values() for eid in r['event_chain']};extras={k:r for k,r in positive.items() if k not in preserved};base_conflicts={k for k,r in extras.items() if set(r['event_chain'])&used};remaining={k:r for k,r in extras.items() if k not in base_conflicts};mutual={k for k,r in remaining.items() for q,s in remaining.items() if k!=q and set(r['event_chain'])&set(s['event_chain'])};additions={k:r for k,r in remaining.items() if k not in mutual};merged={**preserved,**additions}
for key,r in positive.items():r['included_in_base_preserving_merge']=key in merged
order=lambda r:(r['entry_seconds'],r['exit_seconds'])
groups={'保留基础的融合表':sorted(merged.values(),key=order),'所有判1候选':sorted(positive.values(),key=order),'判1但端点冲突':sorted([r for k,r in positive.items() if k not in merged],key=order),'跨轮标签分歧':sorted([r for r in positive.values() if r.get('judgment_disagreement')],key=order)}
judgments={}
for k,r in all_rows.items():
    values=[s['label'] for s in r.get('existing_review_sources',[])]
    if labels.get(k):values.append(labels[k])
    if values:judgments[k]=values
(OUT/'judgments.json').write_text(json.dumps(judgments));(OUT/'positive_union.json').write_text(json.dumps(list(positive.values()),ensure_ascii=False));(OUT/'positive_workbook.json').write_text(json.dumps({'groups':groups},ensure_ascii=False));(OUT/'preserved_record_ids.json').write_text(json.dumps(list(merged)))
with (OUT/'判1来源对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['配对编号','旧48条基础','此前56条主表','本轮新审核判1','进入融合主表','竞争配对'])
    for r in groups['所有判1候选']:
        key=r['pair_id'];conflicts=[k for k,s in positive.items() if k!=key and set(r['event_chain'])&set(s['event_chain'])];w.writerow([key,int(r.get('base_E48',False)),int(key in preserved),int(labels.get(key)=='1'),int(key in merged),'、'.join(conflicts)])
summary={'new_review_rows':len(rows),'new_review_counts':dict(Counter(labels.values())),'previous_main_rows':len(preserved),'old_E48_retained':sum(r.get('base_E48',False) for r in merged.values()),'positive_union':len(positive),'merged_main_rows':len(merged),'new_noncompeting_positive_rows':len(additions),'competing_positive_rows':len(positive)-len(merged),'reviewed_pair_total':len(judgments),'new_model_review_performed':False}
(OUT/'ingest_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
