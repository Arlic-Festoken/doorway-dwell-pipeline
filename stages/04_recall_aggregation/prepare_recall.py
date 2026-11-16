"""Separate broad retrieval artifacts from all reviewed and assigned results."""
import json
from pathlib import Path
import shutil
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;TEMP=ROOT/'doorway_temporal_20261109';REC=ROOT/'doorway_recovery_20261108'
for name in ('results','logs','features','crops','videos','panels','contexts','anchors'):(HERE/name).mkdir(exist_ok=True)
for name in ('events.json','sequences.json'):
    if not (HERE/name).exists():(HERE/name).symlink_to(TEMP/name)
shutil.copy2(TEMP/'results/时序方案审核对照.csv',HERE/'results/已有审核快照.csv')
shutil.copy2(TEMP/'results/unique_pair_manifest.json',HERE/'results/previous_pair_manifest.json')
for folder in ('videos','panels'):
    for source in (TEMP/folder).iterdir():
        if source.is_file() and not (HERE/folder/source.name).exists():(HERE/folder/source.name).symlink_to(source)
packs=json.loads((REC/'results/evidence_manifest.json').read_text())['events'];events=json.loads((TEMP/'events.json').read_text());samples=json.loads((REC/'results/clean_feature_samples.json').read_text());with_image={r['event_id'] for r in samples};frames=[];anchor_events=[]
clocks=[json.loads((ROOT/'work/clocks'/f'VID{i:05d}.json').read_text())['offset_seconds'] for i in range(106)]
pts=[np.asarray([float(r['best_effort_timestamp_time']) for r in json.loads((ROOT/'work/native_crossings'/f'{i:03d}_pts.json').read_text())['frames']]) for i in range(106)]
for p in packs:
    eid=p['event_id']
    if eid in with_image:continue
    e=events[eid];clip=int(e['clip']);fi=int(np.argmin(abs(pts[clip]+clocks[clip]-e['time'])));box=np.array(e['crossing_box']).clip([0,0,0,0],[1919,1079,1920,1080])
    if box[2]-box[0]<12 or box[3]-box[1]<40:continue
    row={'key':f'a{eid:04d}','event_id':eid,'clip':clip,'native_frame_index':fi,'time':float(pts[clip][fi]+clocks[clip]),'box':box.tolist(),'visible':True,'overlap':None,'path':f'anchors/e{eid:04d}.jpg','identity_sequence_confirmed':False}
    frames.append(row);anchor_events.append(eid)
(HERE/'frames.json').write_text(json.dumps(frames));(HERE/'results/anchor_sources.json').write_text(json.dumps(frames,indent=2));print(json.dumps({'existing_clean_image_endpoints':len(with_image),'supplemental_anchor_endpoints':len(anchor_events),'anchors_are_retrieval_only':True}),flush=True)
