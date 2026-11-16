"""Ship one self-contained workflow and annotation dataset, with shared media once."""
from datetime import datetime,timezone
import argparse,json,os,time,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent;BUNDLE=HERE.parent;parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();started=time.time()
summary=json.loads((BUNDLE/'dataset/数据说明.json').read_text());files=sorted(p for p in BUNDLE.rglob('*') if p.is_file() and not p.name.startswith('._') and not p.name.startswith('.') and not p.name.endswith('.inspect.ndjson') and '__pycache__' not in p.parts and 'node_modules' not in p.parts and p.suffix!='.zip')
if not (BUNDLE/'dataset/统一标注表.xlsx').is_file():raise FileNotFoundError('Export the unified XLSX before packaging')
args.output.parent.mkdir(parents=True,exist_ok=True);ledger=HERE.parent.parent/'logs/process_ledger.jsonl';ledger.parent.mkdir(exist_ok=True);record={'stage':'unified_packaging','pid':os.getpid(),'cwd':str(BUNDLE),'output':str(args.output),'gpu':None,'started_at':datetime.now(timezone.utc).isoformat(),'stop_condition':'archive complete or 600 seconds'}
with ledger.open('a') as f:f.write(json.dumps(record)+'\n')
with zipfile.ZipFile(args.output,'w',zipfile.ZIP_STORED) as archive:
    for p in files:
        compressed=p.suffix in ('.json','.jsonl','.csv','.html','.txt','.md','.py','.mjs','.ndjson')
        archive.write(p,str(p.relative_to(BUNDLE)),compress_type=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED,compresslevel=3 if compressed else None)
result={**summary,'archive':str(args.output),'archive_bytes':args.output.stat().st_size,'files':len(files),'video_files':sum(p.suffix=='.mp4' for p in files),'elapsed_seconds':time.time()-started,'status':'complete'}
(HERE.parent.parent/'completion_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
with ledger.open('a') as f:f.write(json.dumps({**record,'status':'complete','elapsed_seconds':time.time()-started})+'\n')
