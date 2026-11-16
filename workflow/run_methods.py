"""Regenerate method outputs from prepared doorway clips/features in an isolated run directory."""
from datetime import datetime, timezone
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))

PHASE_TO_STAGE = {
    "doorway_recovery_20261108": "02_static_matching",
    "doorway_temporal_20261109": "03_temporal_matching",
    "doorway_recall_20261109": "04_recall_aggregation",
    "doorway_examples_20261109": "05_feedback_mining",
}


def metadata_copy(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name.startswith("._") or p.suffix in (".zip", ".xlsx", ".html"):
            continue
        q = dst / p.name
        if p.is_dir():
            metadata_copy(p, q)
        else:
            shutil.copy2(p, q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("DWELL_PROJECT_ROOT", CONFIG.get("default_project_root", "."))),
        help="Root directory containing prepared doorway clips, features, and caches.",
    )
    parser.add_argument("--run-root", type=Path, required=True, help="New isolated output directory.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["static", "temporal", "recall", "examples"],
        default=["static", "temporal", "recall", "examples"],
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES index.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-step wall timeout in seconds.")
    args = parser.parse_args()

    root = args.project_root.resolve()
    run = args.run_root.resolve()
    if run.exists():
        raise FileExistsError("Use a new --run-root; existing runs and annotations are never overwritten.")
    if root == run or root.is_relative_to(run):
        raise ValueError("--run-root must not contain --project-root.")

    run.mkdir(parents=True)
    phases = list(PHASE_TO_STAGE.keys())
    for src in root.iterdir():
        if src.resolve() == run or run.is_relative_to(src.resolve()):
            continue
        if src.name not in phases + ["logs"]:
            (run / src.name).symlink_to(src, target_is_directory=src.is_dir())
    (run / "logs").mkdir()

    for phase, stage_dir in PHASE_TO_STAGE.items():
        src = root / phase
        dst = run / phase
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            for p in src.iterdir():
                if p.name.startswith("._") or p.suffix in (".zip", ".xlsx"):
                    continue
                q = dst / p.name
                if p.name == "logs":
                    q.mkdir(exist_ok=True)
                    continue
                if p.name in ("results", "variants"):
                    metadata_copy(p, q)
                    if phase == "doorway_examples_20261109" and p.name == "results":
                        target = q / "endpoints"
                        if target.exists():
                            shutil.rmtree(target)
                    continue
                if p.name in ("crops", "contexts", "panels", "videos"):
                    q.mkdir(exist_ok=True)
                    if phase != "doorway_examples_20261109":
                        for media in p.rglob("*"):
                            if media.is_file():
                                target = q / media.relative_to(p)
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.symlink_to(media.resolve())
                    continue
                if p.is_file() and p.suffix in (".py", ".json", ".csv", ".md"):
                    shutil.copy2(p, q)
                else:
                    q.symlink_to(p.resolve(), target_is_directory=p.is_dir())

        stage_src = REPO_ROOT / "stages" / stage_dir
        if stage_src.exists():
            for script in stage_src.glob("*.py"):
                shutil.copy2(script, dst / script.name)

    stage_plan = {
        "static": [
            (phases[0], "probability_pairer.py", []),
            (phases[0], "probability_pairer.py", ["--scene-adapt"]),
            (phases[0], "assign_variants.py", []),
        ],
        "temporal": [(phases[1], "pair_sequences.py", [])],
        "recall": [(phases[2], "build_recall.py", []), (phases[2], "merge_positives.py", [])],
        "examples": [
            (phases[3], name, [])
            for name in [
                "ingest_reviews.py",
                "find_missing_entries.py",
                "recover_occluded_entries.py",
                "mine_examples.py",
                "complete_new_candidates.py",
                "render_new_entries.py",
            ]
        ],
    }

    venv_py = root / ".venv-kpr/bin/python"
    python_bin = str(venv_py) if venv_py.exists() else sys.executable
    env = {
        **os.environ,
        "DWELL_PROJECT_ROOT": str(run),
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }
    vendor = root / "rebuild_20261107/vendor"
    if vendor.exists():
        env["PYTHONPATH"] = str(vendor)

    ledger = run / "logs/process_ledger.jsonl"
    for stage in ["static", "temporal", "recall", "examples"]:
        if stage not in args.stages:
            continue
        for phase, name, extra in stage_plan[stage]:
            log = run / "logs" / f'{stage}_{Path(name).stem}{"_scene" if extra else ""}.log'
            command = [python_bin, str(run / phase / name), *extra]
            with log.open("w", encoding="utf-8") as f:
                process = subprocess.Popen(command, cwd=run / phase, env=env, stdout=f, stderr=subprocess.STDOUT)
                record = {
                    "stage": stage,
                    "script": name,
                    "pid": process.pid,
                    "cwd": str(run / phase),
                    "gpu": args.gpu,
                    "log": str(log),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "stop_condition": f"complete or {args.timeout}s",
                }
                with ledger.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(record) + "\n")
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    raise TimeoutError(f"{name} exceeded {args.timeout}s; see {log}")
            if code:
                raise RuntimeError(f"{name} exited {code}; see {log}")
            with ledger.open("a", encoding="utf-8") as out:
                out.write(json.dumps({**record, "status": "complete"}) + "\n")

    subprocess.run([sys.executable, str(HERE / "pipeline.py"), "refresh", "--project-root", str(run)], check=True)
    print(json.dumps({"run_root": str(run), "dataset": str(HERE.parent / "dataset/统一数据集.jsonl")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
