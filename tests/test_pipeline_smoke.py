"""Synthetic smoke test for workflow/pipeline.py (deduplication, <30s split, label import)."""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class PipelineSmokeTest(unittest.TestCase):
    def test_build_and_import_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            wf = tmp_dir / "workflow"
            shutil.copytree(REPO / "workflow", wf)
            inputs = wf / "inputs"
            inputs.mkdir(parents=True, exist_ok=True)

            # Synthetic endpoints (2 entries, 2 exits)
            endpoints = {
                "1": {"event_id": 1, "time": 36000.0, "interval": [35999.9, 36000.1], "panel": "panels/0001.jpg", "videos": [], "views": []},
                "2": {"event_id": 2, "time": 36100.0, "interval": [36099.9, 36100.1], "panel": "panels/0002.jpg", "videos": [], "views": []},
                "3": {"event_id": 3, "time": 36020.0, "interval": [36019.9, 36020.1], "panel": "panels/0003.jpg", "videos": [], "views": []},
                "4": {"event_id": 4, "time": 36500.0, "interval": [36499.9, 36500.1], "panel": "panels/0004.jpg", "videos": [], "views": []},
            }
            (inputs / "endpoints.json").write_text(json.dumps(endpoints), encoding="utf-8")
            (inputs / "base_E48.json").write_text("[]", encoding="utf-8")
            (inputs / "previous_main70.json").write_text("[]", encoding="utf-8")

            # Historical label conflict on I0001_O0004
            history = {
                "I0001_O0004": [
                    {"source": "static", "label": "0", "note": ""},
                    {"source": "temporal", "label": "1", "note": ""},
                ]
            }
            (inputs / "label_history.json").write_text(json.dumps(history), encoding="utf-8")

            # Raw candidates before dedup: 3 rows -> 2 unique pairs (1 long, 1 short <30s)
            candidates = [
                {
                    "pair_id": "I0001_O0004", "entry_event": 1, "exit_event": 4,
                    "entry_seconds": 36000.0, "exit_seconds": 36500.0,
                    "entry_interval": [35999.9, 36000.1], "exit_interval": [36499.9, 36500.1],
                    "event_chain": [1, 4], "method": "A", "new_entry_proposal": False,
                    "approximate_entry": False, "details": {},
                },
                {
                    "pair_id": "I0001_O0004", "entry_event": 1, "exit_event": 4,
                    "entry_seconds": 36000.0, "exit_seconds": 36500.0,
                    "entry_interval": [35999.9, 36000.1], "exit_interval": [36499.9, 36500.1],
                    "event_chain": [1, 4], "method": "H", "new_entry_proposal": False,
                    "approximate_entry": False, "details": {},
                },
                {
                    "pair_id": "I0001_O0003", "entry_event": 1, "exit_event": 3,
                    "entry_seconds": 36000.0, "exit_seconds": 36020.0,
                    "entry_interval": [35999.9, 36000.1], "exit_interval": [36019.9, 36020.1],
                    "event_chain": [1, 3], "method": "N_短停留竞争", "new_entry_proposal": False,
                    "approximate_entry": False, "details": {},
                },
            ]
            with (inputs / "method_candidates.jsonl").open("w", encoding="utf-8") as f:
                for row in candidates:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Run build
            subprocess.run([sys.executable, str(wf / "pipeline.py"), "build"], check=True)
            summary = json.loads((tmp_dir / "dataset/数据说明.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["unique_pairs"], 2)
            self.assertEqual(summary["long_candidates"], 1)
            self.assertEqual(summary["short_competitors"], 1)
            self.assertEqual(summary["method_rows_before_deduplication"], 3)

            # Import annotation via CSV
            returned = tmp_dir / "returned.csv"
            with returned.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["配对编号", "本轮标注", "进店动作成立", "备注"])
                w.writerow(["I0001_O0004", "1", "是", "verified"])
            subprocess.run([sys.executable, str(wf / "pipeline.py"), "import-labels", str(returned)], check=True)

            updated_summary = json.loads((tmp_dir / "dataset/数据说明.json").read_text(encoding="utf-8"))
            self.assertEqual(updated_summary["annotation_updates"], 1)


if __name__ == "__main__":
    unittest.main()
