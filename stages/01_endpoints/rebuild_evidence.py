import os
"""Anchor doorway evidence to a continuous local body, preserving source PTS."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get('DWELL_PROJECT_ROOT', '.'))
BASE = ROOT / 'doorway_pairer_20261108/query_round'
PREVIOUS = ROOT / 'doorway_balance_20261108'
HERE = Path(__file__).resolve().parent
for name in ('results', 'logs', 'crops', 'contexts', 'panels'):
    (HERE / name).mkdir(exist_ok=True)
cv2.setNumThreads(2)
started = time.time()
events = json.loads((PREVIOUS / 'events_final.json').read_text())
dispositions = json.loads((PREVIOUS / 'results/event_dispositions.json').read_text())
old_reviews = json.loads((HERE / 'old_reviews.json').read_text())
canonical = dispositions['canonical']
active = [e['event_id'] for e in events if e['eligible_crossing'] and canonical[e['event_id']] == e['event_id']]
priority = {r[k] for r in old_reviews['rows'] if r['disagreement'] or r['row'] in (2, 22, 25) for k in ('entry_event', 'exit_event')}
measurements = defaultdict(list)
cached_crops = defaultdict(list)
features = {}
for path in sorted((BASE / 'sequences').glob('clip_*.json')):
    data = json.loads(path.read_text())
    for eid, values in data['measurements'].items():
        measurements[int(eid)].extend(values)
    z = np.load(BASE / 'features' / (path.stem + '.npz'))
    for i, key in enumerate(z['keys']):
        features[str(key)] = (z['cope'][i].astype(np.float32), z['sapiens'][i].astype(np.float32))
    for r in data['crops']:
        cached_crops[r['event_id']].append(r)
bank = dict(np.load(ROOT / 'rebuild_20261107/work/bank/observations.npz'))
order = np.argsort(bank['time'])
bank_times = bank['time'][order]
nodes = json.loads((ROOT / 'rebuild_20261107/work/bank/nodes.json').read_text())
boundary = np.asarray(json.loads((ROOT / 'scene.json').read_text())['boundary_pixels'])
inventory = json.loads((ROOT / 'source_inventory.json').read_text())
clocks = [json.loads((ROOT / 'work/clocks' / f'VID{i:05d}.json').read_text())['offset_seconds'] for i in range(106)]
pts = [np.asarray([float(r['best_effort_timestamp_time']) for r in json.loads((ROOT / 'work/native_crossings' / f'{i:03d}_pts.json').read_text())['frames']]) for i in range(106)]

def ious(box, boxes):
    boxes = np.asarray(boxes)
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2])).prod(1)
    return intersection / ((boxes[:, 2:] - boxes[:, :2]).prod(1) + np.prod(np.asarray(box[2:]) - box[:2]) - intersection).clip(1)

frame_bodies = {}
def collision(r):
    stamp = r['time']
    index = min(int(np.searchsorted(bank_times, stamp)), len(bank_times) - 1)
    if index and abs(bank_times[index-1]-stamp) < abs(bank_times[index]-stamp):
        index -= 1
    if abs(bank_times[index]-stamp) > .075:
        return 'other_bodies_unobserved'
    key = float(bank_times[index])
    if key not in frame_bodies:
        lo, hi = np.searchsorted(bank_times, [key-.001, key+.001])
        boxes = bank['boxes'][order[lo:hi]]
        boxes = boxes[(boxes[:, 4] >= .3) & ((boxes[:, 3]-boxes[:, 1]) >= 70)]
        keep = []
        for j in np.argsort(-boxes[:, 4]):
            if not keep or ious(boxes[j, :4], boxes[keep, :4]).max() < .65:
                keep.append(int(j))
        frame_bodies[key] = boxes[keep, :4]
    boxes = frame_bodies[key]
    if not len(boxes):
        return 'other_bodies_unobserved'
    overlap = ious(np.asarray(r['box']), boxes)
    target = int(np.argmax(overlap))
    if overlap[target] < .35:
        return 'anchor_not_supported_by_body_detection'
    other = np.arange(len(boxes)) != target
    if np.any(overlap[other] > .3):
        return 'overlapping_other_body'
    return None

def valid(r, pose_required):
    b = np.asarray(r['box']); h = b[3]-b[1]
    if h < 100 or h/max(b[2]-b[0], 1) < 1.25:
        return 'small_or_non_upright_body'
    if pose_required:
        p = np.asarray(r['pose'])
        if np.sum(p[[5, 6, 11, 12], 2] >= .4) < 3 or r['foot'] is None:
            return 'missing_torso_or_foot'
        if min(p[5, 1], p[6, 1]) >= max(p[11, 1], p[12, 1]):
            return 'invalid_body_vertical_order'
    return collision(r)

def follow(pool, at, target_box, pose_required=True):
    # A contaminated observation breaks a branch; never jump past an occlusion.
    unique = {}
    for r in pool:
        key = round(r['time'], 4)
        if key not in unique or r.get('quality', 0) > unique[key].get('quality', 0):
            unique[key] = r
    pool = sorted(unique.values(), key=lambda r: r['time'])
    rejected = Counter()
    anchors = []
    for k, r in enumerate(pool):
        if abs(r['time']-at) > .3:
            continue
        overlap = float(ious(np.asarray(target_box), [r['box']])[0])
        reason = valid(r, pose_required)
        if reason:
            rejected[reason] += 1
        elif overlap > .5:
            anchors.append((overlap - abs(r['time']-at), k))
    if not anchors:
        return [], dict(rejected), 'no_clean_crossing_anchor'
    anchor = max(anchors)[1]
    selected = [pool[anchor]]
    for step in (-1, 1):
        previous = pool[anchor]
        for k in range(anchor+step, len(pool) if step == 1 else -1, step):
            r = pool[k]
            if not at-.6 <= r['time'] <= at+1.1:
                break
            reason = valid(r, pose_required)
            height = min(r['box'][3]-r['box'][1], previous['box'][3]-previous['box'][1])
            overlap = float(ious(np.asarray(previous['box']), [r['box']])[0])
            motion = np.linalg.norm(np.asarray(r['foot'])-previous['foot'])/max(height, 1) if r['foot'] is not None and previous['foot'] is not None else float('inf')
            if reason or abs(r['time']-previous['time']) > .25 or overlap <= .5 or motion > .35:
                rejected[reason or 'discontinuous_box_or_foot'] += 1
                break
            selected.append(r); previous = r
    return sorted(selected, key=lambda r: r['time']), dict(rejected), None

def choose(chain, at, limit=8):
    selected = []
    for target in np.linspace(at-.5, at+1, limit):
        available = [r for r in chain if r not in selected]
        if not available:
            break
        selected.append(min(available, key=lambda r: abs(r['time']-target)))
    return sorted(selected, key=lambda r: r['time'])

endpoints = []
requests = defaultdict(lambda: defaultdict(list))
clean_features = {}; feature_keys = []
for eid in active:
    e = events[eid]; source = e.get('source_event', eid)
    item = {'event_id': eid, 'direction': e['direction'], 'time': e['time'], 'interval': e['interval'], 'source_category': e['source_category'], 'priority': eid in priority}
    if e.get('crossgap_nodes'):
        selected = []; reasons = Counter()
        for part, node in enumerate(e['crossgap_nodes']):
            rows = np.asarray(nodes[node]['rows']); times = bank['time'][rows]
            at = e['interval'][part]; rows = rows[abs(times-at) <= 1.1]
            pool = []
            for row in rows:
                box = bank['boxes'][row, :4].tolist(); foot = [(box[0]+box[2])/2, box[3]]
                pool.append({'time': float(bank['time'][row]), 'clip': int(bank['clip'][row]), 'box': box, 'foot': foot, 'quality': float(bank['boxes'][row, 4]), 'foot_method': 'box_estimate', 'fragment': part})
            nearest = min(pool, key=lambda r: abs(r['time']-at)) if pool else None
            chain, rejected, error = follow(pool, at, nearest['box'], False) if nearest else ([], {}, 'no_fragment')
            selected.extend(choose(chain, at, 4)); reasons.update(rejected)
        item.update(kind='two_fragment_interval', continuous_crossing=False, ready_for_judge=len(selected)>=6, rejection_reasons=dict(reasons), status='interval_requires_cross_fragment_verification')
        chain = []
    else:
        pool = [r for r in measurements[source] if abs(r['time']-e['time'])<=1.25]
        chain, rejected, error = follow(pool, e['time'], e['crossing_box'])
        selected = choose(chain, e['time'])
        before = any(r['time'] <= e['time']-.2 for r in selected)
        after = any(r['time'] >= e['time']+.4 for r in selected)
        ready = len(selected)>=6 and before and after
        item.update(kind='continuous_pose_sequence', continuous_crossing=bool(chain), ready_for_judge=ready, rejection_reasons=rejected, status='ready' if ready else error or 'insufficient_clean_temporal_coverage')
        for crop in cached_crops[source]:
            matching = [r for r in chain if abs(r['time']-crop['time'])<.045 and ious(np.asarray(r['box']), [crop['box']])[0]>.9]
            if matching and crop['path'] in features:
                key = f'{eid}:{crop["path"]}'
                clean_features[key] = features[crop['path']]
                feature_keys.append({'event_id': eid, 'key': key, 'time': crop['time'], 'box': crop['box'], 'original_crop': crop['path']})
    item['frames'] = []
    for k, r in enumerate(selected):
        clip = r['clip']; frame_index = int(np.argmin(abs(pts[clip]+clocks[clip]-r['time'])))
        sample = {'cell': f'V{k+1}', 'time': float(pts[clip][frame_index]+clocks[clip]), 'clip': clip, 'native_frame_index': frame_index, 'box': r['box'], 'foot': r['foot'], 'foot_method': r.get('foot_method', 'pose_ankles'), 'crop': f'crops/e{eid:04d}_v{k+1}.jpg', 'context': f'contexts/e{eid:04d}_v{k+1}.jpg'}
        if 'fragment' in r: sample['fragment'] = r['fragment']
        item['frames'].append(sample); requests[clip][frame_index].append(sample)
    item['panel'] = f'panels/e{eid:04d}.jpg'
    endpoints.append(item)
manifest = {'events': endpoints, 'old_review_priority_rows': [r['row'] for r in old_reviews['rows'] if r['disagreement'] or r['row'] in (2, 22, 25)], 'historical_records_used': False, 'rules': {'adjacent_iou_min': .5, 'foot_motion_max_heights': .35, 'other_body_overlap_max': .3, 'maximum_adjacent_gap_seconds': .25, 'minimum_views_for_judging': 6}}
(HERE/'results/evidence_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False))
keys = sorted(clean_features)
np.savez_compressed(HERE/'clean_features.npz', keys=np.asarray(keys), cope=np.stack([clean_features[k][0] for k in keys]), sapiens=np.stack([clean_features[k][1] for k in keys]))
(HERE/'results/clean_feature_samples.json').write_text(json.dumps(feature_keys))
print(json.dumps({'stage':'selection_complete','endpoints':len(endpoints),'ready':sum(e['ready_for_judge'] for e in endpoints),'feature_samples':len(keys),'selected_frames':sum(len(e['frames']) for e in endpoints),'seconds':round(time.time()-started,1)}), flush=True)

for clip, by_frame in sorted(requests.items()):
    indices = sorted(by_frame)
    needed = [i for i in indices if any(not (HERE/s['context']).exists() or not (HERE/s['crop']).exists() for s in by_frame[i])]
    if not needed: continue
    expression = '+'.join(f'eq(n,{i})' for i in needed)
    command = ['ffmpeg','-nostdin','-v','error','-threads','2','-filter_threads','2','-i',str(ROOT/'raw/DCIMA'/inventory[clip]['file']),'-vf',f"select='{expression}'",'-frames:v',str(len(needed)),'-vsync','0','-an','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
    with (HERE/'logs/decode.log').open('a') as log, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log) as decoder:
        for index in needed:
            raw = decoder.stdout.read(1920*1080*3)
            if len(raw)!=1920*1080*3: raise RuntimeError(f'Incomplete native frame {clip}:{index}')
            frame = np.frombuffer(raw, np.uint8).reshape(1080,1920,3)
            for sample in by_frame[index]:
                x1,y1,x2,y2 = np.rint(sample['box']).astype(int)
                x1,x2 = np.clip([x1,x2],0,1920); y1,y2 = np.clip([y1,y2],0,1080)
                if x2<=x1 or y2<=y1: raise ValueError('Invalid crop bounds')
                cv2.imwrite(str(HERE/sample['crop']), frame[y1:y2,x1:x2], [cv2.IMWRITE_JPEG_QUALITY,95])
                thumb = cv2.resize(frame, (640,360)); scale = 1/3
                cv2.polylines(thumb, [np.rint(boundary*scale).astype(np.int32)], False, (0,210,255), 1)
                cv2.rectangle(thumb,(round(x1*scale),round(y1*scale)),(round(x2*scale),round(y2*scale)),(0,255,0),2)
                cv2.circle(thumb,tuple(np.rint(np.asarray(sample['foot'])*scale).astype(int)),4,(0,0,255),-1)
                cv2.imwrite(str(HERE/sample['context']),thumb,[cv2.IMWRITE_JPEG_QUALITY,92])
        if decoder.wait(): raise RuntimeError(f'Native decode failed: {clip}')
    print(json.dumps({'stage':'decoded_clip','clip':clip,'frames':len(needed),'elapsed_seconds':round(time.time()-started,1)}),flush=True)

font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
for endpoint in endpoints:
    frames = endpoint['frames']; panel = Image.new('RGB',(1280,80+510*max(1,int(np.ceil(len(frames)/4)))),'#f5f5f5'); draw=ImageDraw.Draw(panel)
    title = f"Event {endpoint['event_id']} | {endpoint['kind']} | {endpoint['status']}"
    draw.text((12,8),title,fill='black',font=font)
    draw.text((12,36),'Green: target box. Red: observed foot or labelled box estimate. Yellow: doorway.',fill='black',font=font)
    for k, sample in enumerate(frames):
        x=(k%4)*320;y=80+(k//4)*510;stamp=sample['time'];label=f"{sample['cell']}  {int(stamp)//3600:02}:{int(stamp)%3600//60:02}:{stamp%60:05.2f}"
        draw.text((x+4,y),label,fill='black',font=font)
        crop=Image.open(HERE/sample['crop']);crop.thumbnail((312,290));panel.paste(crop,(x+(320-crop.width)//2,y+28+(290-crop.height)//2))
        context=Image.open(HERE/sample['context']);context.thumbnail((320,180));panel.paste(context,(x,y+322))
        if sample['foot_method']=='box_estimate':draw.text((x+4,y+300),'FOOT ESTIMATED / SEPARATE FRAGMENT',fill='#9b2929',font=font)
    panel.save(HERE/endpoint['panel'],quality=92)
summary={'endpoints':len(endpoints),'ready_for_judge':sum(e['ready_for_judge'] for e in endpoints),'continuous_ready':sum(e['ready_for_judge'] and e['continuous_crossing'] for e in endpoints),'statuses':dict(Counter(e['status'] for e in endpoints)),'frames':sum(len(e['frames']) for e in endpoints),'clean_feature_samples':len(keys),'priority_rows':manifest['old_review_priority_rows'],'historical_records_used':False,'elapsed_seconds':round(time.time()-started,1)}
(HERE/'results/evidence_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
