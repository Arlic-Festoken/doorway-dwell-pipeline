"""Self-contained delivery of new hypotheses, accepted-label union and evidence."""
from datetime import datetime
import csv,json,os,time,zipfile
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results';PREV=ROOT/'doorway_recall_20261109';start=time.time();cv2.setNumThreads(2)
ledger={'stage':'examples_delivery','pid':os.getpid(),'cwd':str(HERE),'output':str(OUT),'started_at':datetime.now().isoformat(),'gpu':None,'stop_condition':'delivery archive complete or 600 seconds'}
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(ledger)+'\n')
new=json.loads((OUT/'new_candidates.json').read_text());positive=json.loads((OUT/'positive_workbook.json').read_text())['groups']['所有判1候选'];short=json.loads((OUT/'short_competitors.json').read_text());proposals=json.loads((OUT/'new_entry_proposals.json').read_text());proposal_ids={r['event_id'] for r in proposals};groups={'新增待审核':new,'已有判1':positive,'不足30秒竞争':short};allrows=[{**r,'group':g} for g,rr in groups.items() for r in rr];need={e for r in allrows for e in r.get('event_chain',[r['entry_event'],r['exit_event']])}|proposal_ids
endpoints={};files=set();(OUT/'endpoints').mkdir(exist_ok=True);(HERE/'pairs').mkdir(exist_ok=True)
for eid in sorted(need):
    path=OUT/'endpoints'/f'{eid:04d}.json'
    if eid not in proposal_ids:path.write_text((PREV/'results/endpoints'/f'{eid:04d}.json').read_text())
    e=json.loads(path.read_text());endpoints[eid]=e;paths=[e['panel']]+[v['path'] for v in e['videos']]+[v[k] for v in e['views'] for k in ('crop','context') if v.get(k)]
    for relative in paths:
        target=HERE/relative
        if not target.exists():target.parent.mkdir(exist_ok=True,parents=True);target.symlink_to(PREV/relative)
        files.add(relative)
def stamp(t):
    t=round(t);return f'{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}'
def full(t,interval,approx=False):return ('约 ' if approx or interval[1]-interval[0]>.6 else '')+'2026-01-26 '+stamp(t)
labels={'feedback_recall':'审核反馈重排','feedback_two_models':'审核反馈双模型','accepted_pair_template':'已判1人物扩展','new_entry_sequence':'新入口序列补回'}
for group,rows in groups.items():
    with (OUT/f'{group}_行号对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['工作簿行号','配对编号','入口事件','出口事件','来源','与主表竞争的配对','新审核_0异人_1同人_2不足','备注'])
        for n,r in enumerate(rows,2):w.writerow([n,r['pair_id'],r['entry_event'],r['exit_event'],'、'.join(labels.get(s,s) for s in r.get('candidate_sources',[])),','.join(r.get('competes_with_main',[])),'',''])
    with (OUT/f'{group}_模型任务.jsonl').open('w') as f:
        for r in rows:
            task={**r,'entry':endpoints[r['entry_event']],'exit':endpoints[r['exit_event']],'pair_panel':f'pairs/{r["pair_id"]}.jpg','required_output':{'pair_id':r['pair_id'],'label':'0 different / 1 same / 2 insufficient','entrance_supported':'yes / no / uncertain','supporting_views':[],'reason':''}}
            if len(r.get('event_chain',[]))>2:task['intermediate_crossings']=[endpoints[e] for e in r['event_chain'][1:-1]]
            for s in r.get('seed_support',[]):
                seed=next(p for p in positive if p['pair_id']==s['seed_pair_id']);s['entry_endpoint']=f'endpoints/{seed["entry_event"]:04d}.json';s['exit_endpoint']=f'endpoints/{seed["exit_event"]:04d}.json'
            f.write(json.dumps(task,ensure_ascii=False)+'\n')
with (OUT/'新增样例来源.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['配对编号','进入时间','出店时间','停留时间','来源','入口清晰模板数','出口清晰模板数','外观排序分_非概率','支持它的已判1配对','入口证据类型','新入口区间秒','与主表竞争'])
    for r in new:w.writerow([r['pair_id'],full(r['entry_seconds'],r['entry_interval'],r.get('approximate_entry',False)),full(r['exit_seconds'],r['exit_interval']),stamp(r['duration_seconds']),'、'.join(labels.get(s,s) for s in r['candidate_sources']),r['entry_clean_frames'],r['exit_clean_frames'],round(r['appearance_evidence_score'],4),','.join(s['seed_pair_id'] for s in r['seed_support']),r.get('new_entry_source','既有端点'),round(r['entry_interval'][1]-r['entry_interval'][0],3),','.join(r['competes_with_main'])])
with (OUT/'不足30秒竞争_时间表.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['配对编号','进入时间','出店时间','停留时间'])
    for r in short:w.writerow([r['pair_id'],full(r['entry_seconds'],r['entry_interval'],r.get('approximate_entry',False)),full(r['exit_seconds'],r['exit_interval']),stamp(r['duration_seconds'])])
for r in allrows:
    relative=f'pairs/{r["pair_id"]}.jpg';destination=HERE/relative
    if not destination.exists():
        images=[cv2.imread(str(HERE/endpoints[e]['panel'])) for e in [r['entry_event'],r['exit_event']]];h=max(int(im.shape[0]*800/im.shape[1]) for im in images);canvas=np.full((h+70,1620,3),25,np.uint8)
        for k,im in enumerate(images):
            im=cv2.resize(im,(800,int(im.shape[0]*800/im.shape[1])));canvas[70:70+im.shape[0],k*820:k*820+800]=im;cv2.putText(canvas,('ENTRY' if k==0 else 'EXIT')+f' {r["entry_event"] if k==0 else r["exit_event"]:04d}',(k*820+15,55),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),2)
        cv2.putText(canvas,r['pair_id']+'  candidate evidence',(15,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1);cv2.imwrite(str(destination),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
    files.add(relative)
summary=json.loads((OUT/'ingest_summary.json').read_text());mining=json.loads((OUT/'mining_summary.json').read_text())
text=f'''店铺停留时长新增样例（2026-11-09）

本包提供 {len(new)} 条此前未审核的配对，另附 {len(short)} 条不足30秒的竞争组合。
新组合来自：原端点的审核反馈/已判1模板扩展3条；新补入口序列52条。补回14个入口提议，其中11个有外观匹配出口。新入口需要同时审核进店动作和人物身份，尚未算入已判1结果。

store_dwell_判1融合更新.xlsx：主表70条，保留旧E版48条及此前56条基础；所有判1去重合集133条，另列63条有端点竞争的判1。4条跨轮标签分歧单列。133条不能直接视作133次互不重叠到访。
store_dwell_新增55条待审核.xlsx：三列时间；配对编号见新增待审核_行号对照.csv。每行的入口、出口、连续片段及截图见新增待审核_模型任务.jsonl。已判0或2的旧组合没有原样重送。

打开候选审核索引.html可逐条查看两端视频、裁剪、原画面。图片和视频路径相对于解压目录。模型任务用pair_id回传0/1/2；新入口另答entrance_supported。视频时间采用原始PTS与校准时钟。不同fragment不拼成连续轨迹。
排序模型用129条无跨轮分歧的判1及6074条判0训练，不使用12条历史记录；分数只是筛选分，不是测得的正确率。新入口匹配由多张COPE/Sapiens外观证据筛选，保留全部超过本轮证据门槛的组合，没有Top-K数量截断。已判1模板扩展不递归产生真值。
入口区间超过0.6秒或用身体框补脚点时，时间标“约”；新入口的肩髋连续性、脚点来源和原始观测保存在new_entry_proposals.json。少于30秒先作为竞争项解决身份，再从最终停留表排除，不与别的长停留自动合并。
用户自行完成模型审核；本轮没有调用视觉大模型或人工判人。
'''
(OUT/'使用说明.txt').write_text(text)
compact=[{'pair_id':r['pair_id'],'entry_event':r['entry_event'],'exit_event':r['exit_event'],'entry_seconds':r['entry_seconds'],'exit_seconds':r['exit_seconds'],'group':r['group'],'sources':'、'.join(labels.get(s,s) for s in r.get('candidate_sources',[])),'new_entry_proposal':r.get('new_entry_proposal',False)} for r in allrows]
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>店铺停留时长新增样例</title><style>body{font:15px system-ui,sans-serif;max-width:1440px;margin:28px auto;padding:0 20px;color:#253745;background:#f5f7f9}p{line-height:1.7}table{width:100%;border-collapse:collapse;background:white}th,td{padding:10px;text-align:left;border-bottom:1px solid #dbe2e8}button,select,input{padding:9px 12px;border:1px solid #afbdc7;background:white;border-radius:5px;margin:4px}button{cursor:pointer}a{color:#245e82}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.photos{display:flex;flex-wrap:wrap;gap:6px}.photos figure{width:120px;margin:0}.photos img{width:120px;height:260px;object-fit:contain;background:#eee}video{max-width:100%}#evidence{background:white;padding:18px;margin-top:18px}#evidence:empty{display:none}h1{font-size:26px}h2{font-size:20px}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><h1>店铺停留时长新增样例</h1>'''
page+=f'<p>{len(new)} 条新候选 · {len(short)} 条短停留竞争项 · 133 条已有判1合集（70条主表，63条端点竞争）。<a href="store_dwell_新增55条待审核.xlsx">新增时间表</a> · <a href="store_dwell_判1融合更新.xlsx">判1融合表</a> · <a href="新增待审核_行号对照.csv">审核编号对照</a> · <a href="使用说明.txt">说明</a></p><p>新入口候选须同时核对进店动作和人物身份。每行是待核对的进出假设；共享端点的行需择一。</p>'
page+='''<select id="group" onchange="draw()"><option>新增待审核</option><option>已有判1</option><option>不足30秒竞争</option></select><input id="search" placeholder="配对编号" oninput="draw()"><p id="count"></p><table><thead><tr><th>配对编号</th><th>进入</th><th>出店</th><th>停留</th><th>来源</th><th>证据</th></tr></thead><tbody id="list"></tbody></table><div id="evidence"></div><script>'''
page+='const rows='+json.dumps(compact,ensure_ascii=False)+';const endpoints='+json.dumps(endpoints,ensure_ascii=False)+';'
page+='''function stamp(t){t=Math.round(t);return [Math.floor(t/3600),Math.floor(t%3600/60),t%60].map(x=>String(x).padStart(2,'0')).join(':')}function draw(){const g=document.getElementById('group').value,q=document.getElementById('search').value.trim().toUpperCase(),data=rows.filter(r=>r.group===g&&(!q||r.pair_id.includes(q)));document.getElementById('count').textContent=data.length+' 条';document.getElementById('list').innerHTML=data.map(r=>`<tr><td>${r.pair_id}</td><td>${stamp(r.entry_seconds)}</td><td>${stamp(r.exit_seconds)}</td><td>${stamp(r.exit_seconds-r.entry_seconds)}</td><td>${r.sources}</td><td><button onclick="show('${r.pair_id}')">查看</button></td></tr>`).join('')}function end(e,label){return `<section><h2>${label} ${e.event_id}</h2><p>${e.crossing_confirmed_by_human===false?'新增入口提议，需审核过线动作':''}</p>${e.videos.map(v=>`<p>片段 ${v.fragment+1}</p><video controls preload="none" src="${v.path}"></video>`).join('')}<div class="photos">${e.views.map(v=>`<figure><a href="${v.crop}" target="_blank"><img loading="lazy" src="${v.crop}"></a><figcaption>${v.cell} ${stamp(v.time)} <a href="${v.context}" target="_blank">原画面</a></figcaption></figure>`).join('')}</div></section>`}function show(id){const r=rows.find(r=>r.pair_id===id),box=document.getElementById('evidence');box.innerHTML=`<h2>${id}　${stamp(r.entry_seconds)} 至 ${stamp(r.exit_seconds)}</h2><p><a href="pairs/${id}.jpg" target="_blank">两端对照面板</a></p><div class="pair">${end(endpoints[r.entry_event],'进入')}${end(endpoints[r.exit_event],'出店')}</div>`;box.scrollIntoView({behavior:'smooth'})}draw();</script></html>'''
(OUT/'候选审核索引.html').write_text(page)
data_files=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.csv','.jsonl','.txt','.html','.xlsx')]+[OUT/n for n in ('positive_union.json','new_candidates.json','short_competitors.json','ingest_summary.json','mining_summary.json','new_entry_proposals.json','new_entry_summary.json','occluded_entry_summary.json')]+list((OUT/'endpoints').glob('*.json'))
archive_path=OUT/'store_dwell_新增样例与判1融合_完整证据.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_STORED) as archive:
    for p in data_files:archive.write(p,str(p.relative_to(OUT)),compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
    for relative in sorted(files):archive.write(HERE/relative,relative)
result={'stage':'delivery_complete','new_review_candidates':len(new),'short_competitors':len(short),'positive_union':len(positive),'base_preserving_main':summary['merged_main_rows'],'retained_E48':summary['old_E48_retained'],'endpoints':len(endpoints),'evidence_files':len(files),'videos':sum(p.endswith('.mp4') for p in files),'archive_bytes':archive_path.stat().st_size,'archive':str(archive_path),'elapsed_seconds':time.time()-start}
(OUT/'delivery_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(HERE/'completion_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
with (ROOT/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps({**ledger,'stage':'examples_delivery_complete','elapsed_seconds':time.time()-start})+'\n')
