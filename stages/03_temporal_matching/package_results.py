import csv
import html
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent;REC=HERE.parent/'doorway_recovery_20261108'
reports=json.loads((HERE/'results/assignment_summary.json').read_text());video=json.loads((HERE/'results/video_manifest.json').read_text());prior=json.loads((REC/'results/unique_pair_manifest.json').read_text());previous={r['pair_id']:r['scheme_rows'] for r in prior}
all_pairs={};schemes=[]
for report in reports:
    config=report['variant'];records=json.loads((HERE/'variants'/config['directory']/'delivery_records.json').read_text());scheme={**config,'complete_visits':report['complete_visits'],'unmatched_endpoints':report['unmatched_endpoints'],'short_visits_excluded':report['short_visits_excluded']};schemes.append(scheme)
    for row,r in enumerate(records['confirmed'],2):
        key=f'I{r["entry_event"]:04d}_O{r["exit_event"]:04d}'
        if key not in all_pairs:all_pairs[key]={**r,'pair_id':key,'scheme_rows':{},'previous_scheme_rows':previous.get(key,{})}
        all_pairs[key]['scheme_rows'][config['id']]=row
    scheme['new_vs_previous_five']=sum(f'I{r["entry_event"]:04d}_O{r["exit_event"]:04d}' not in previous for r in records['confirmed'])
    scheme['over_20_minutes']=sum(r['duration_seconds']>=1200 for r in records['confirmed'])
pairs=sorted(all_pairs.values(),key=lambda r:(r['entry_seconds'],r['exit_seconds']));files=set();blocks=[]
def stamp(t):
    t=round(t);return f'{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}'
def time_stamp(t,interval):return ('约 ' if interval and interval[1]-interval[0]>.6 else '')+stamp(t)
with (HERE/'results/时序方案审核对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['配对编号','进入时间','出店时间','停留时间','本轮方案及Excel行号','上一轮相同配对','审核结果_0异人_1同人_2不足','备注'])
    for r in pairs:
        locations='、'.join(f'{s} 第{row}行' for s,row in r['scheme_rows'].items());old_locations='、'.join(f'{s} 第{row}行' for s,row in r['previous_scheme_rows'].items()) or '本轮新增配对';entry=time_stamp(r['entry_seconds'],r['entry_interval']);exit=time_stamp(r['exit_seconds'],r['exit_interval']);duration=stamp(r['duration_seconds']);writer.writerow([r['pair_id'],entry,exit,duration,locations,old_locations,'',''])
        ends=[]
        for end,label in [('entry','进入'),('exit','出店')]:
            e=video[str(r[end+'_event'])];clips=[]
            for v in e['videos']:
                files.add(v['path']);clips.append(f'<p>片段 {v["fragment"]+1}：{stamp(v["interval"][0])} 至 {stamp(v["interval"][1])}</p><video controls playsinline preload="none" src="{v["path"]}"></video>')
            files.add(e['panel']);clips.append(f'<a href="{e["panel"]}" target="_blank">查看该端点静态证据</a>')
            if not e['videos']:clips.insert(0,'<p>此端点沿用静态证据，没有满足连续性条件的短视频。</p>')
            ends.append(f'<section><h3>{label} · 事件 {e["event_id"]}</h3>{"".join(clips)}</section>')
        # Intermediate short-return endpoints stay individually inspectable.
        additional=[]
        for eid in r['event_chain'][1:-1]:
            for v in video[str(eid)]['videos']:files.add(v['path']);additional.append(f'<a href="{v["path"]}">事件 {eid} 的短时返回片段</a>')
        title=f'{r["pair_id"]}　{entry} → {exit}　停留 {duration}'
        blocks.append(f'<details data-schemes="{" ".join(r["scheme_rows"])}" data-new="{int(not r["previous_scheme_rows"])}"><summary>{html.escape(title)}<small>{html.escape(locations)}　{html.escape(old_locations)}</small></summary><button onclick="playPair(this)">同步播放两端片段</button><div class="pair">{"".join(ends)}</div><p>{"　".join(additional)}</p></details>')
overview=''.join(f'<tr><td>{s["id"]} {s["name"]}</td><td>{s["complete_visits"]}</td><td>{s["unmatched_endpoints"]}</td><td>{s["new_vs_previous_five"]}</td><td><a href="datasets/{s["workbook"]}">下载表格</a></td></tr>' for s in schemes)
buttons='<button onclick="filterScheme(\'all\')">全部</button><button onclick="filterScheme(\'new\')">仅本轮新增配对</button>'+''.join(f'<button onclick="filterScheme(\'{s["id"]}\')">方案 {s["id"]}</button>' for s in schemes)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>店铺停留时长时序方案审核</title><style>body{font-family:system-ui,sans-serif;max-width:1480px;margin:32px auto;padding:0 20px;color:#243541;background:#f6f8fa}h1{font-size:26px}p{line-height:1.7}table{border-collapse:collapse;background:white}td,th{padding:12px 18px;border-bottom:1px solid #d5dfe3;text-align:left}details{margin:12px 0;padding:16px;background:white;border:1px solid #d5dfe3;border-radius:8px}summary{font-weight:600;cursor:pointer}small{display:block;font-weight:400;margin-top:8px;color:#546773}button{padding:10px 16px;margin:6px 4px;border:1px solid #aab9c3;border-radius:5px;background:white;cursor:pointer}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}video{width:100%;background:#111}a{color:#245a7b}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><h1>店铺停留时长时序方案审核</h1>'''+f'<p>五版共 {len(pairs)} 个不同配对，相同配对只需审核一次。沿用上一轮事件编号。门口视频提供原画面、目标框和放大人像，画面时间为原始 PTS。黄色线为门口边界，红点为框估计脚点。两个区间片段分别显示，不把断开的动作连成连续证据。</p><table><tr><th>方案</th><th>完整记录</th><th>未配对端点</th><th>相较上一轮新增的不同配对</th><th>数据表</th></tr>{overview}</table><p>F 为同一批连续帧的静态对照，G 使用时序网络，H 增加顺序对齐，I 进一步融合原外观评分。F–I 使用相同接纳门槛；J 为 I 的较宽门槛版。所有完整记录均不少于30秒。</p><p><a href="时序方案审核对照.csv">下载审核对照CSV</a>：0=异人，1=同人，2=证据不足，留空=未审核。</p>'+buttons+''.join(blocks)+'''<script>function filterScheme(id){for(const r of document.querySelectorAll('details'))r.hidden=id==='new'?r.dataset.new!=='1':id!=='all'&&!r.dataset.schemes.split(' ').includes(id);}function playPair(button){const d=button.closest('details');for(const v of d.querySelectorAll('section video:first-of-type')){v.currentTime=0;v.play();}}</script></html>'''
(HERE/'results/时序方案审核索引.html').write_text(page);(HERE/'results/unique_pair_manifest.json').write_text(json.dumps(pairs,ensure_ascii=False));(HERE/'results/scheme_catalog.json').write_text(json.dumps(schemes,ensure_ascii=False,indent=2))
with (HERE/'results/方案对比.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['方案','完整记录','未配对端点','短停留排除','新增不同配对','20分钟以上记录'])
    for s in schemes:w.writerow([s['id']+' '+s['name'],s['complete_visits'],s['unmatched_endpoints'],s['short_visits_excluded'],s['new_vs_previous_five'],s['over_20_minutes']])
archive_path=HERE/'results/store_dwell_时序五方案与视频证据.zip'
with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_STORED) as archive:
    for name in ['时序方案审核索引.html','时序方案审核对照.csv','方案对比.csv','scheme_catalog.json','unique_pair_manifest.json','video_manifest.json']:archive.write(HERE/'results'/name,name)
    for s in schemes:archive.write(HERE/'results/datasets'/s['workbook'],'datasets/'+s['workbook'])
    for name in sorted(files):archive.write(HERE/name,name)
summary={'schemes':schemes,'unique_pairs':len(pairs),'new_pairs_vs_previous_five':sum(not r['previous_scheme_rows'] for r in pairs),'common_all_five':sum(len(r['scheme_rows'])==5 for r in pairs),'video_files':sum(name.endswith('.mp4') for name in files),'archive_bytes':archive_path.stat().st_size,'accuracy_evaluated':False,'historical_records_used':False}
(HERE/'results/multi_variant_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
