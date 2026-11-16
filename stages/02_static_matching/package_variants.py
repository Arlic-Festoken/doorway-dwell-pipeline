"""A stable pair index shares review labels across the alternative datasets."""
import csv
import html
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
catalog=[('A','A_multiview','多视角','store_dwell_A_多视角.xlsx'),('B','B_parts','服装部位','store_dwell_B_服装部位.xlsx'),('C','F_scene_adapt','场景适配','store_dwell_C_场景适配.xlsx'),('D','C_coverage','覆盖增强','store_dwell_D_覆盖增强.xlsx'),('E','E_exploration','候选扩展','store_dwell_E_候选扩展.xlsx')]
pack={r['event_id']:r for r in json.loads((HERE/'results/evidence_manifest.json').read_text())['events']};pairs={};schemes=[]
for letter,directory,name,filename in catalog:
    records=json.loads((HERE/'variants'/directory/'delivery_records.json').read_text());report=json.loads((HERE/'variants'/directory/'delivery_report.json').read_text());scheme={'id':letter,'internal_directory':directory,'name':name,'workbook':filename,'complete_visits':report['complete_visits'],'unmatched_endpoints':report['unmatched_endpoints'],'short_visits_excluded':report['short_visits_excluded']};schemes.append(scheme)
    for row,record in enumerate(records['confirmed'],2):
        key=f"I{record['entry_event']:04d}_O{record['exit_event']:04d}"
        if key not in pairs:pairs[key]={'pair_id':key,**record,'scheme_rows':{}}
        pairs[key]['scheme_rows'][letter]=row
pairs=sorted(pairs.values(),key=lambda r:(r['entry_seconds'],r['exit_seconds']));files=set();blocks=[]
def stamp(t):
    t=round(t);return f'{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}'
def interval_stamp(t,interval):return ('约 ' if interval and interval[1]-interval[0]>.6 else '')+stamp(t)
csv_path=HERE/'results/多方案审核对照.csv'
with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['配对编号','进入时间','出店时间','停留时间','方案及Excel行号','审核结果_0异人_1同人_2不足','备注'])
    for r in pairs:
        locations='、'.join(f'{scheme} 第{row}行' for scheme,row in r['scheme_rows'].items());entry=interval_stamp(r['entry_seconds'],r['entry_interval']);exit=interval_stamp(r['exit_seconds'],r['exit_interval']);duration=stamp(r['duration_seconds'])
        writer.writerow([r['pair_id'],entry,exit,duration,locations,'',''])
        panels=[]
        for end,label in [('entry','进入'),('exit','出店')]:
            e=pack[r[end+'_event']];files.add(e['panel']);r[end+'_evidence']=e
            for frame in e['frames']:files.update([frame['crop'],frame['context']])
            panels.append(f'<section><h3>{label} · 事件 {e["event_id"]}</h3><a href="{e["panel"]}" target="_blank"><img loading="lazy" src="{e["panel"]}" alt="{label}证据"></a></section>')
        title=f"{r['pair_id']}　{entry} → {exit}　停留 {duration}"
        blocks.append(f'<details data-schemes="{" ".join(r["scheme_rows"])}"><summary>{html.escape(title)}<small>{html.escape(locations)}</small></summary><div class="pair">{"".join(panels)}</div></details>')
overview=''.join(f'<tr><td>{s["id"]} {s["name"]}</td><td>{s["complete_visits"]}</td><td>{s["unmatched_endpoints"]}</td><td><a href="datasets/{s["workbook"]}">下载表格</a></td></tr>' for s in schemes)
buttons='<button onclick="showScheme(\'all\')">全部配对</button>'+''.join(f'<button onclick="showScheme(\'{s["id"]}\')">{s["id"]} {s["name"]}</button>' for s in schemes)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>店铺停留时长多方案审核索引</title><style>body{max-width:1440px;margin:32px auto;padding:0 20px;font-family:system-ui,sans-serif;color:#20323c;background:#f6f8f9}h1{font-size:26px}p{line-height:1.7}table{border-collapse:collapse;background:white;margin:18px 0}th,td{padding:12px 22px;text-align:left;border-bottom:1px solid #d5dfe3}button{padding:10px 16px;margin:4px;border:1px solid #aabfc4;background:white;border-radius:5px;cursor:pointer}details{background:white;border:1px solid #d5dfe3;border-radius:8px;margin:12px 0;padding:16px}summary{cursor:pointer;font-size:17px;font-weight:600}small{display:block;margin-top:8px;font-size:14px;font-weight:400;color:#526870}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;height:auto}h3{font-size:16px}a{color:#215e64}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><h1>店铺停留时长多方案审核索引</h1>'''+f'<p>共 {len(pairs)} 个不同配对。相同配对在各版中只列一次，编号固定；下方标明对应方案和Excel行号。各版统一使用本轮补全的局部视图，便于比较。绿色框为目标人体，红点为脚点，黄线为门口边界；估计脚点在图中标明。</p><table><tr><th>方案</th><th>完整自动配对</th><th>未配对端点</th><th>数据表</th></tr>{overview}</table><p><a href="多方案审核对照.csv">打开审核对照CSV</a>，可在审核结果列填写：0=异人，1=同人，2=证据不足。留空表示尚未审核。</p>'+buttons+''.join(blocks)+'''<script>function showScheme(id){for(const row of document.querySelectorAll('details'))row.hidden=id!=='all'&&!row.dataset.schemes.split(' ').includes(id);}</script></html>'''
(HERE/'results/多方案审核索引.html').write_text(page);(HERE/'results/unique_pair_manifest.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2));(HERE/'results/scheme_catalog.json').write_text(json.dumps(schemes,ensure_ascii=False,indent=2))
archive_path=HERE/'results/store_dwell_五方案数据与证据.zip'
with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
    for name in ['多方案审核索引.html','多方案审核对照.csv','unique_pair_manifest.json','scheme_catalog.json']:archive.write(HERE/'results'/name,name)
    for s in schemes:archive.write(HERE/'results/datasets'/s['workbook'],'datasets/'+s['workbook'])
    for name in sorted(files):archive.write(HERE/name,name)
summary={'schemes':schemes,'unique_pairs':len(pairs),'common_to_all_schemes':sum(len(r['scheme_rows'])==5 for r in pairs),'evidence_files':len(files),'archive_bytes':archive_path.stat().st_size,'accuracy_evaluation_performed':False,'historical_records_used':False}
(HERE/'results/multi_variant_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
