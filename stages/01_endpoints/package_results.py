"""Package final row evidence for the user's review, without model judging."""
import html
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
records=json.loads((HERE/'results/delivery_records.json').read_text())['confirmed']
manifest=json.loads((HERE/'results/evidence_manifest.json').read_text())
by_event={e['event_id']:e for e in manifest['events']}
old=json.loads((HERE.parent/'doorway_balance_20261108/results/delivery_records.json').read_text())['confirmed']
old_pairs={(r['entry_event'],r['exit_event']) for r in old}

def stamp(t):
    t=round(t);return f'{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}'

blocks=[];row_manifest=[];files=set()
for index,record in enumerate(records,2):
    duration=stamp(record['duration_seconds'])
    title=f"第 {index} 行 · 进入 {stamp(record['entry_seconds'])} · 出店 {stamp(record['exit_seconds'])} · 停留 {duration}"
    panels=[]
    row={'excel_row':index,'entry_seconds':record['entry_seconds'],'exit_seconds':record['exit_seconds'],'duration_seconds':record['duration_seconds'],'entry_event':record['entry_event'],'exit_event':record['exit_event'],'unchanged_previous_pair':(record['entry_event'],record['exit_event']) in old_pairs}
    for end,label in [('entry','进入'),('exit','出店')]:
        e=by_event[record[end+'_event']];files.add(e['panel'])
        panels.append(f'<section><h3>{label} · 事件 {e["event_id"]}</h3><a href="{e["panel"]}" target="_blank"><img loading="lazy" src="{e["panel"]}" alt="{label}证据"></a></section>')
        row[end+'_evidence']=e
        for frame in e['frames']:files.update([frame['crop'],frame['context']])
    blocks.append(f'<details><summary>{html.escape(title)}</summary><div class="pair">{"".join(panels)}</div></details>');row_manifest.append(row)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>店铺停留时长新版停留证据</title><style>body{max-width:1440px;margin:30px auto;padding:0 20px;font-family:system-ui,sans-serif;color:#20323c;background:#f6f8f9}h1{font-size:26px}p{line-height:1.65}details{background:white;border:1px solid #d5dfe3;border-radius:8px;margin:12px 0;padding:16px}summary{cursor:pointer;font-size:17px;font-weight:600}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;height:auto}h3{font-size:16px}a{color:#215e64}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><h1>店铺停留时长新版停留证据</h1><p>行号对应新版 Excel 的“自动关联停留”工作表。展开一行查看两端证据，点击图片可查看大图。绿色框为目标人体，红点为脚点，黄线为门口边界。每格同时保留人物裁剪和全画面位置。</p>'''+''.join(blocks)+'</html>'
(HERE/'results/证据索引.html').write_text(page)
(HERE/'results/review_row_manifest.json').write_text(json.dumps(row_manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(HERE/'results/store_dwell_新版证据包.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
    archive.writestr('证据索引.html',page)
    archive.write(HERE/'results/review_row_manifest.json','review_row_manifest.json')
    for name in sorted(files):archive.write(HERE/name,name)
summary={'complete_rows':len(records),'unchanged_previous_pairs':sum(r['unchanged_previous_pair'] for r in row_manifest),'new_or_changed_pairs':sum(not r['unchanged_previous_pair'] for r in row_manifest),'evidence_files':len(files),'zip_bytes':(HERE/'results/store_dwell_新版证据包.zip').stat().st_size}
(HERE/'results/package_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
