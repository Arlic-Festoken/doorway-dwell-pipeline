"""One candidate dataset: collect method outputs, deduplicate, retain labels and evidence."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent;BUNDLE=HERE.parent;INPUT=HERE/'inputs';DATA=BUNDLE/'dataset';CONFIG=json.loads((HERE/'config.json').read_text())
HEADER=['配对编号','进入时间','出店时间','停留时间','已有标注','本轮标注','进店动作成立','备注','来源方法','同进店其他配法','同出店其他配法','证据','记录用途']
def read_json(path):return json.loads(path.read_text())
def json_out(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
def jsonl_out(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def pair_id(r):return f'I{int(r["entry_event"]):04d}_O{int(r["exit_event"]):04d}'
def normalize(r,source):
    a=int(r['entry_event']);b=int(r['exit_event']);start=float(r['entry_seconds']);end=float(r['exit_seconds'])
    if end<=start:raise ValueError(f'Invalid time order: {a}, {b}')
    return {'pair_id':pair_id(r),'entry_event':a,'exit_event':b,'entry_seconds':start,'exit_seconds':end,'entry_interval':r['entry_interval'],'exit_interval':r['exit_interval'],'event_chain':r.get('event_chain',[a,b]),'method':source,'new_entry_proposal':r.get('new_entry_proposal',False),'approximate_entry':r.get('approximate_entry',False),'details':{k:r[k] for k in ('candidate_sources','association_links','retrieval_sources','seed_support','appearance_evidence_score','new_entry_source') if k in r}}

def collect(root):
    INPUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True);sources=[];catalog=[];fingerprints=[]
    def source_rows(path,rows,method):
        sources.extend(normalize(r,method) for r in rows);catalog.append({'id':method,'file':str(path.relative_to(root)),'rows':len(rows)});fingerprints.append({'file':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for m in CONFIG['methods']:
        p=root/m['phase']/'variants'/m['directory']/'delivery_records.json';source_rows(p,read_json(p)['confirmed'],m['id'])
        q=p.parent/'event_dispositions.json';source_rows(q,read_json(q).get('short_visits',[]),m['id'])
    prev=root/'doorway_recall_20261109';ex=root/'doorway_examples_20261109';reviews=read_json(ex/'results/judgments.json');all_recall=[json.loads(line) for line in (prev/'results/全部候选.jsonl').open()]
    source_rows(prev/'results/全部候选.jsonl',[r for r in all_recall if r['tier']==1],'R_多通道已提交候选')
    source_rows(prev/'results/全部候选.jsonl',[r for r in all_recall if r['pair_id'] in reviews],'历史已审核记录')
    source_rows(ex/'results/new_candidates.json',read_json(ex/'results/new_candidates.json'),'N_反馈与补入口')
    source_rows(ex/'results/short_competitors.json',read_json(ex/'results/short_competitors.json'),'N_短停留竞争')
    positives=read_json(ex/'results/positive_union.json');source_rows(ex/'results/positive_union.json',positives,'各轮判1保留')
    jsonl_out(INPUT/'method_candidates.jsonl',sources);json_out(INPUT/'method_catalog.json',catalog);json_out(INPUT/'source_fingerprints.json',fingerprints)
    # Preserve the actual returned review sources, including conflicting labels.
    labels=defaultdict(list)
    review_files=[('static',prev/'results/旧五方案审核快照.csv','审核结果_0异人_1同人_2不足'),('temporal',prev/'results/已有审核快照.csv','审核结果_0异人_1同人_2不足'),('recall',ex/'新审核.csv','新审核_0异人_1同人_2不足')]
    for name,p,column in review_files:
        for r in csv.DictReader(p.open(encoding='utf-8-sig')):
            label=r[column].strip()
            if label:
                if label not in ('0','1','2'):raise ValueError(f'Unknown label in {p}')
                labels[r['配对编号']].append({'source':name,'label':label,'note':r.get('备注','')})
        target=INPUT/'original_reviews'/f'{name}.csv';target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    json_out(INPUT/'label_history.json',labels);json_out(INPUT/'base_E48.json',[r['pair_id'] for r in positives if r.get('base_E48')]);json_out(INPUT/'previous_main70.json',[r['pair_id'] for r in positives if r.get('included_in_base_preserving_merge')])
    proposals=read_json(ex/'results/new_entry_proposals.json');json_out(INPUT/'new_entry_proposals.json',proposals);proposal_ids={r['event_id'] for r in proposals}
    needed={eid for r in sources for eid in r['event_chain']};endpoints={}
    for eid in sorted(needed):
        owner=ex if eid in proposal_ids else prev;ep=read_json(owner/'results/endpoints'/f'{eid:04d}.json');endpoints[str(eid)]=ep
        media=[ep['panel']]+[v['path'] for v in ep['videos']]+[v[k] for v in ep['views'] for k in ('crop','context') if v.get(k)]
        for rel in media:
            dst=DATA/rel;src=(owner/rel).resolve();dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.is_symlink() and dst.resolve()!=src:dst.unlink()
            if not dst.exists():dst.symlink_to(src)
    json_out(INPUT/'endpoints.json',endpoints)
    # Bundle the source used to generate each method; models/video remain on
    # the server. run_methods.py provides the cache-based production entry.
    for phase in ('doorway_recovery_20261108','doorway_temporal_20261109','doorway_recall_20261109','doorway_examples_20261109'):
        target=HERE/'upstream_source'/phase;target.mkdir(parents=True,exist_ok=True)
        for p in (root/phase).glob('*.py'):
            if not p.name.startswith('._'):shutil.copy2(p,target/p.name)
    json_out(INPUT/'collection.json',{'project_root':str(root),'collected_at':datetime.now(timezone.utc).isoformat(),'config':CONFIG,'raw_method_rows':len(sources),'endpoint_count':len(endpoints)})

def load_annotations(path):
    if path.suffix.lower()=='.csv':return list(csv.DictReader(path.open(encoding='utf-8-sig')))
    if path.suffix.lower()!='.xlsx':raise ValueError('Returned annotations must be CSV or XLSX')
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings=[''.join(si.itertext()) for si in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si',ns)]
        rows=[]
        for row in ET.fromstring(archive.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',ns):
            data={}
            for cell in row.findall('s:c',ns):
                column=''.join(c for c in cell.attrib['r'] if c.isalpha());v=cell.find('s:v',ns);value='' if v is None else v.text or ''
                if cell.attrib.get('t')=='s':value=strings[int(value)]
                if cell.attrib.get('t')=='inlineStr':value=''.join(cell.find('s:is',ns).itertext())
                data[column]=value
            rows.append(data)
    if not rows:return []
    head=rows[0];return [{name:r.get(col,'') for col,name in head.items()} for r in rows[1:] if r.get('A')]

def import_labels(path):
    candidates={json.loads(line)['pair_id'] for line in (INPUT/'method_candidates.jsonl').open()};rows=load_annotations(path);updates=[];seen=set();stamp=datetime.now(timezone.utc).isoformat();digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for r in rows:
        key=r['配对编号'].strip();label=str(r.get('本轮标注','')).strip();entrance=str(r.get('进店动作成立','')).strip();note=str(r.get('备注','')).strip()
        if key in seen or key not in candidates:raise ValueError(f'Duplicate or unknown pair ID: {key}')
        seen.add(key)
        if label not in ('','0','1','2') or entrance not in ('','是','否','不确定'):raise ValueError(f'Invalid annotation: {key}')
        if label or entrance or note:updates.append({'pair_id':key,'label':label,'entrance_supported':entrance,'note':note,'source':path.name,'file_sha256':digest,'imported_at':stamp})
    ledger=INPUT/'annotation_updates.jsonl';old=[json.loads(l) for l in ledger.open()] if ledger.exists() else [];done={(r['file_sha256'],r['pair_id']) for r in old};added=[r for r in updates if (r['file_sha256'],r['pair_id']) not in done];jsonl_out(ledger,old+added)
    print(json.dumps({'imported_annotations':len(added)},ensure_ascii=False))

def build():
    sources=[json.loads(line) for line in (INPUT/'method_candidates.jsonl').open()];history=read_json(INPUT/'label_history.json');endpoints=read_json(INPUT/'endpoints.json');base=set(read_json(INPUT/'base_E48.json'));main=set(read_json(INPUT/'previous_main70.json'));updates=defaultdict(list)
    if (INPUT/'annotation_updates.jsonl').exists():
        for r in (json.loads(l) for l in (INPUT/'annotation_updates.jsonl').open()):updates[r['pair_id']].append(r)
    merged={}
    for r in sources:
        key=r['pair_id']
        if key not in merged:
            a=endpoints[str(r['entry_event'])];b=endpoints[str(r['exit_event'])];merged[key]={'pair_id':key,'entry_event':r['entry_event'],'exit_event':r['exit_event'],'entry_seconds':a['time'],'exit_seconds':b['time'],'entry_interval':a['interval'],'exit_interval':b['interval'],'source_methods':[],'source_records':[],'event_chain_variants':[],'new_entry_proposal':False,'approximate_entry':False,'historical_labels':history.get(key,[]),'annotation_updates':updates.get(key,[]),'base_E48':key in base,'previous_main70':key in main}
        dst=merged[key]
        if r['method'] not in dst['source_methods']:dst['source_methods'].append(r['method'])
        dst['source_records'].append(r)
        if r['event_chain'] not in dst['event_chain_variants']:dst['event_chain_variants'].append(r['event_chain'])
        dst['new_entry_proposal']|=r['new_entry_proposal'];dst['approximate_entry']|=r['approximate_entry']
    by_entry=defaultdict(list);by_exit=defaultdict(list)
    for r in merged.values():by_entry[r['entry_event']].append(r['pair_id']);by_exit[r['exit_event']].append(r['pair_id'])
    for r in merged.values():
        labels=sorted({s['label'] for s in r['historical_labels']});recent=[s for s in r['annotation_updates'] if s['label']];r['historical_label_display']='' if not labels else labels[0] if len(labels)==1 else '分歧 '+ '/'.join(labels);r['current_label']=recent[-1]['label'] if recent else labels[0] if len(labels)==1 else '';r['label_display']=recent[-1]['label'] if recent else r['historical_label_display'];r['needs_label_resolution']=len(labels)>1 and not recent;r['duration_seconds']=r['exit_seconds']-r['entry_seconds'];r['record_use']='停留候选' if r['duration_seconds']>=CONFIG['minimum_duration_seconds'] else '短停留竞争';r['entry_alternatives']=[p for p in by_entry[r['entry_event']] if p!=r['pair_id']];r['exit_alternatives']=[p for p in by_exit[r['exit_event']] if p!=r['pair_id']];r['evidence_link']='审核入口.html#'+r['pair_id'];r['entry_evidence']=f'endpoints/{r["entry_event"]:04d}.json';r['exit_evidence']=f'endpoints/{r["exit_event"]:04d}.json'
        if r['duration_seconds']<=0:raise ValueError(f'Invalid canonical event times: {r["pair_id"]}')
    priority=lambda r:0 if not r['historical_labels'] and not r['current_label'] else 1 if r['needs_label_resolution'] else 2 if r['current_label']=='2' else 3 if r['current_label']=='1' else 4
    rows=sorted(merged.values(),key=lambda r:(priority(r),not r['new_entry_proposal'],r['entry_seconds'],r['exit_seconds']))
    def stamp(t):
        t=round(t);return f'{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}'
    def formatted(t,interval,approx=False):return ('约 ' if approx or interval[1]-interval[0]>.6 else '')+CONFIG['date']+' '+stamp(t)
    jsonl_out(DATA/'统一数据集.jsonl',rows);json_out(DATA/'workbook_input.json',{'rows':rows,'headers':HEADER,'date':CONFIG['date']})
    with (DATA/'统一标注表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(HEADER)
        for r in rows:w.writerow([r['pair_id'],formatted(r['entry_seconds'],r['entry_interval'],r['approximate_entry']),formatted(r['exit_seconds'],r['exit_interval']),stamp(r['duration_seconds']),r['label_display'],'','','','、'.join(r['source_methods']),len(r['entry_alternatives']),len(r['exit_alternatives']),r['evidence_link'],r['record_use']])
    for eid,e in endpoints.items():json_out(DATA/'endpoints'/f'{int(eid):04d}.json',e)
    with (DATA/'模型标注任务.jsonl').open('w') as f:
        for r in rows:
            task={'pair_id':r['pair_id'],'entry_seconds':r['entry_seconds'],'exit_seconds':r['exit_seconds'],'duration_seconds':r['duration_seconds'],'record_use':r['record_use'],'new_entry_proposal':r['new_entry_proposal'],'entry':endpoints[str(r['entry_event'])],'exit':endpoints[str(r['exit_event'])],'entry_alternatives':r['entry_alternatives'],'exit_alternatives':r['exit_alternatives'],'prior_labels':r['historical_labels'],'required_output':{'配对编号':r['pair_id'],'本轮标注':'0 / 1 / 2','进店动作成立':'是 / 否 / 不确定','备注':''}}
            f.write(json.dumps(task,ensure_ascii=False,separators=(',',':'))+'\n')
    compact=[{k:r[k] for k in ('pair_id','entry_event','exit_event','entry_seconds','exit_seconds','historical_label_display','current_label','needs_label_resolution','record_use','new_entry_proposal','source_methods','entry_alternatives','exit_alternatives')} for r in rows]
    template=(HERE/'review.html').read_text();template=template.replace('/*DATA_ROWS*/',json.dumps(compact,ensure_ascii=False)).replace('/*DATA_ENDPOINTS*/',json.dumps(endpoints,ensure_ascii=False));(DATA/'审核入口.html').write_text(template)
    summary={'unique_pairs':len(rows),'long_candidates':sum(r['record_use']=='停留候选' for r in rows),'short_competitors':sum(r['record_use']=='短停留竞争' for r in rows),'historically_reviewed':sum(bool(r['historical_labels']) for r in rows),'unreviewed':sum(not r['historical_labels'] and not r['current_label'] for r in rows),'historical_label_counts':dict(Counter(r['historical_label_display'] for r in rows)),'any_historical_positive':sum(any(v['label']=='1' for v in r['historical_labels']) for r in rows),'old_E48_retained':sum(r['base_E48'] for r in rows),'previous_main70_retained':sum(r['previous_main70'] for r in rows),'method_rows_before_deduplication':len(sources),'source_counts':dict(Counter(m for r in rows for m in r['source_methods'])),'unique_endpoints':len(endpoints),'annotation_updates':sum(len(r['annotation_updates']) for r in rows),'one_to_one_selection_performed':False,'historical_12_used':False,'dataset_sha256':hashlib.sha256((DATA/'统一数据集.jsonl').read_bytes()).hexdigest()}
    json_out(DATA/'数据说明.json',summary);print(json.dumps(summary,ensure_ascii=False))

parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
p=sub.add_parser('refresh',help='Collect all current method outputs and export one dataset');p.add_argument('--project-root',type=Path,default=Path(CONFIG['default_project_root']))
sub.add_parser('build',help='Rebuild the unique dataset from packaged input snapshots')
p=sub.add_parser('import-labels',help='Append returned labels and refresh the dataset');p.add_argument('file',type=Path)
args=parser.parse_args()
if args.command=='refresh':collect(args.project_root);build()
elif args.command=='build':build()
else:import_labels(args.file);build()
