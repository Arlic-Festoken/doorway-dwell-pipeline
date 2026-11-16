import csv
from datetime import datetime
import json
import os
from pathlib import Path
import time
import zipfile

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
started=time.time();ledger={'stage':'recall_packaging','pid':os.getpid(),'cwd':str(HERE),'output':str(OUT),'started_at':datetime.now().isoformat(),'stop_condition':'archives complete or 300 seconds','gpu':None}
with (HERE.parent/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps(ledger)+'\n')
report=json.loads((OUT/'recall_summary.json').read_text());merged=json.loads((OUT/'positive_merge_summary.json').read_text());rows=[json.loads(line) for line in (OUT/'全部候选.jsonl').open()];endpoints={int(path.stem):json.loads(path.read_text()) for path in (OUT/'endpoints').glob('*.json')}
pending=[r for r in rows if r['existing_review']=='']
with (OUT/'优先新增_行号对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['工作簿行号','配对编号','入口事件','出口事件','新审核_0异人_1同人_2不足','备注'])
    for i,r in enumerate([r for r in pending if r['tier']==1],2):writer.writerow([i,r['pair_id'],r['entry_event'],r['exit_event'],'',''])
for name,data in [('优先新增_模型任务',[r for r in pending if r['tier']==1]),('扩展新增_模型任务',[r for r in pending if r['tier']<=2]),('宽召回新增_模型任务',[r for r in pending if r['tier']<=3]),('全部新增_模型任务',pending)]:
    with (OUT/f'{name}.jsonl').open('w') as f:
        for r in data:
            a=endpoints[r['entry_event']];b=endpoints[r['exit_event']]
            f.write(json.dumps({'pair_id':r['pair_id'],'entry_time_seconds':r['entry_seconds'],'exit_time_seconds':r['exit_seconds'],'duration_seconds':r['duration_seconds'],'entry':a,'exit':b,'required_output':{'pair_id':r['pair_id'],'label':'0 different / 1 same / 2 insufficient','supporting_views':[],'reason':''}},ensure_ascii=False)+'\n')
instructions='''本包用途：尽量保留正确进出配对，再由视觉模型筛选。

1. “store_dwell_48条基础融合.xlsx”以旧E版48条判1为底座，原样保留，加入8条端点不冲突的判1记录，主表56条。各轮判1去重共有72条，其余16条端点有竞争，单列保留。“所有判1候选”不是72次可直接相加的到访。4条判1记录跨轮有标签分歧，原标签和来源均在判1融合来源对照.csv。
2. 01、02、03、04为累积集合：多模型双向Top3、Top10、Top30、全部有图像可检索且时间合法的组合。分数和排名只决定审核先后，不是同人概率或真值。
3. 优先从“优先新增_模型任务.jsonl”开始。每条已带两端图片、上下文、视频路径及输出格式；路径相对于解压目录。后续可按同一配对编号跳过已完成审核的项目。
4. 两轮已有审核快照完整保留，没有改写原表。原有0、1、2均未混入“新增”队列。0=异人，1=同人，2=证据不足，空白=未审核。任一轮判1的记录全部保留，标签分歧另留原始记录。审核顺序使用无分歧0/1反馈拟合，排序分不代表实际准确率，也不删候选。模型判断不是人工真值。
5. 先检查每端证据是否指向同一个人体。框跳到别人、只见倒影、人体缺失或无法比较时给2。比较上下装、鞋、随身物及可见纹理，指出具体图片格号或视频时间。没有明显差异不自动判1。
6. source=unvalidated_crossing_anchor的端点只有过线时单张补充截图，尚未得到连续身份验证。需要结合另一端和原画面判断；不要当成已确认身份。
7. 保留帧的顺序。不同fragment分开看，不将两段拼成连续动作。视频为门口附近片段，时间字样来自原视频PTS；红点是框估计脚点。
8. 不足30秒的竞争组合独立保存，不进主停留表。审核结束后先解决同一端点的多个判同冲突，短停留若获确认也要占用对应端点，最后才排除不足30秒的到访。不能自动选最近出口或最长停留。
9. 已有1203个过线未证实事件仍在旧结果中；本包覆盖当前750个有效端点，不把未证实的过线伪装成已确认入口。

反馈请保留pair_id，返回label及所依据的图片/视频位置。只保留标签1的行还不够：共享入口或出口的多个标签1必须比较竞争候选，再得到可计数的最终停留表。小数据包不含图片和视频；完整证据包包含全部媒体，两种包内的相对路径一致。
'''
(OUT/'使用说明.txt').write_text(instructions)
compact=[[r['pair_id'],r['entry_event'],r['exit_event'],round(r['entry_seconds'],3),round(r['exit_seconds'],3),r['tier'],r['existing_review'],r['candidate_index']] for r in rows]
endpoint_ui={str(k):{'views':v['views'],'videos':v['videos'],'panel':v['panel'],'source':v['source']} for k,v in endpoints.items()}
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>店铺停留时长候选审核</title><style>body{font:15px system-ui,sans-serif;max-width:1440px;margin:28px auto;padding:0 20px;color:#253745;background:#f5f7f9}p{line-height:1.7}table{width:100%;border-collapse:collapse;background:white}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #dae2e8}button,select,input{padding:9px 12px;border:1px solid #afbdc7;background:white;border-radius:5px;margin:4px}button{cursor:pointer}a{color:#245e82}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.photos{display:flex;flex-wrap:wrap;gap:6px}.photos figure{width:120px;margin:0}.photos img{width:120px;height:260px;object-fit:contain;background:#eee}video{max-width:100%;background:#111}#evidence{background:white;padding:18px;border:1px solid #cbd6dd;margin-top:18px}h1{font-size:26px}h2{font-size:20px}.muted{color:#5c6d78}#evidence:empty{display:none}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><h1>店铺停留时长候选审核</h1><p>一个入口可保留多个出口备选。下列行数是候选假设数，不是人数或已确认停留数。已审核记录保留原标注，默认只显示新增候选。</p>'''
page+=f'<p>以旧E版48条判1为底座，主表已融合至56条；72条判1去重合集及16条端点竞争记录一并保留。<a href="store_dwell_48条基础融合.xlsx">下载基础融合表</a> · <a href="判1融合来源对照.csv">判1来源与竞争关系</a></p><p>共 {len(rows):,} 个候选组合，新增待审核 {len(pending):,} 个。<a href="使用说明.txt">使用说明</a> · <a href="仅新增待审核_审核表.csv">新增审核表</a> · <a href="优先新增_模型任务.jsonl">优先模型任务</a> · <a href="全部新增_模型任务.jsonl">全部模型任务</a></p>'
page+='''<label>范围<select id="tier" onchange="refresh()"><option value="1">优先候选</option><option value="2">扩展候选</option><option value="3">宽召回候选</option><option value="4">全部备用组合</option></select></label><label>审核状态<select id="review" onchange="refresh()"><option value="new">新增待审核</option><option value="all">全部</option><option value="1">已有判同</option><option value="0">已有判异人</option><option value="2">已有证据不足</option></select></label><input id="search" placeholder="配对编号或入口/出口编号" oninput="refresh()"><p id="count"></p><button onclick="move(-1)">上一页</button><button onclick="move(1)">下一页</button><table><thead><tr><th>配对编号</th><th>进入时间</th><th>出店时间</th><th>停留时间</th><th>已有审核</th><th>证据</th></tr></thead><tbody id="list"></tbody></table><div id="evidence"></div><script>'''
page+='const rows='+json.dumps(compact,ensure_ascii=False)+';const endpoints='+json.dumps(endpoint_ui,ensure_ascii=False)+';'
page+='''let filtered=[],page=0;const labels={'':'未审核','0':'异人','1':'同人','2':'证据不足'};function stamp(t){t=Math.round(t);return [Math.floor(t/3600),Math.floor(t%3600/60),t%60].map(x=>String(x).padStart(2,'0')).join(':');}function refresh(){const tier=+document.getElementById('tier').value,rev=document.getElementById('review').value,q=document.getElementById('search').value.trim().toUpperCase();filtered=rows.filter(r=>r[5]<=tier&&(rev==='all'||(rev==='new'?r[6]==='':r[6]===rev))&&(!q||r[0].includes(q)||String(r[1])===q||String(r[2])===q));page=0;draw();}function move(d){page=Math.max(0,Math.min(Math.ceil(filtered.length/100)-1,page+d));draw();}function draw(){document.getElementById('count').textContent=`${filtered.length} 个候选，第 ${page+1} / ${Math.max(1,Math.ceil(filtered.length/100))} 页`;document.getElementById('list').innerHTML=filtered.slice(page*100,page*100+100).map(r=>`<tr><td>${r[0]}</td><td>${stamp(r[3])}</td><td>${stamp(r[4])}</td><td>${stamp(r[4]-r[3])}</td><td>${labels[r[6]]}</td><td><button onclick="show(${r[7]-1})">查看</button></td></tr>`).join('');}function end(e,label){return `<section><h2>${label}</h2><p class="muted">${e.source==='unvalidated_crossing_anchor'?'仅过线锚点截图，未经连续身份验证':'局部人物证据'}</p>${e.videos.map(v=>`<p>片段 ${v.fragment+1}</p><video controls preload="none" src="${v.path}"></video>`).join('')}<div class="photos">${e.views.map(v=>`<figure><a href="${v.crop}" target="_blank"><img loading="lazy" src="${v.crop}"></a><figcaption>${v.cell} ${stamp(v.time)}${v.context?` <a href="${v.context}" target="_blank">原画面</a>`:''}</figcaption></figure>`).join('')}</div></section>`;}function show(i){const r=rows[i],box=document.getElementById('evidence');box.innerHTML=`<h2>${r[0]}　${stamp(r[3])} 至 ${stamp(r[4])}</h2><div class="pair">${end(endpoints[r[1]],'进入')}${end(endpoints[r[2]],'出店')}</div>`;box.scrollIntoView({behavior:'smooth'});}refresh();</script></html>'''
(OUT/'候选审核索引.html').write_text(page)
files=set()
for e in endpoints.values():
    for v in e['videos']:files.add(v['path'])
    files.add(e['panel'])
    for v in e['views']:
        files.add(v['crop'])
        if v['context']:files.add(v['context'])
data_files=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.csv','.jsonl','.txt','.html','.xlsx')]+[OUT/name for name in ('recall_summary.json','positive_merge_summary.json','positive_union.json','positive_conflicts.json','反馈排序模型.json')]
data_files+=list((OUT/'endpoints').glob('*.json'))
with zipfile.ZipFile(OUT/'store_dwell_召回候选数据.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for p in data_files:archive.write(p,str(p.relative_to(OUT)),compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
with zipfile.ZipFile(OUT/'store_dwell_召回候选与全部证据.zip','w',zipfile.ZIP_STORED) as archive:
    for p in data_files:archive.write(p,str(p.relative_to(OUT)),compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
    for relative in sorted(files):archive.write(HERE/relative,relative)
summary={'candidate_pairs':len(rows),'new_unreviewed':len(pending),'evidence_files':len(files),'video_files':sum(n.endswith('.mp4') for n in files),'data_archive_bytes':(OUT/'store_dwell_召回候选数据.zip').stat().st_size,'full_archive_bytes':(OUT/'store_dwell_召回候选与全部证据.zip').stat().st_size}
(OUT/'package_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
with (HERE.parent/'logs/process_ledger.jsonl').open('a') as f:f.write(json.dumps({**ledger,'stage':'recall_packaging_complete','elapsed_seconds':time.time()-started})+'\n')
