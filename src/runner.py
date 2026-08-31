"""
runner.py -- resumable driver for the factorial experiment.

Colab disconnects. The queue is therefore rebuilt from disk on every start: any
(dataset, arch, arm, seed) already present in results.csv is skipped, and each
completed run is appended and flushed immediately. Killing the session mid-queue
costs one run, not the batch.

    python src/runner.py --config configs/nlst.yaml --results results/results.csv
    python src/runner.py --config configs/nlst.yaml --results results/results.csv --dry-run

Grid order is deliberate: it sweeps ARMS INNERMOST, so an interrupted queue
still yields complete arm comparisons for the architectures it got through.
A half-finished queue that has all six architectures on the slice arm and none
on the patient arm is worth nothing; a queue with two architectures across all
three arms is already a result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

KEY_FIELDS = ["dataset", "arch", "arm", "seed"]

RESULT_FIELDS = KEY_FIELDS + [
    "n_train", "n_val", "n_test",
    "test_accuracy", "test_macro_f1", "test_macro_auc", "test_loss",
    "val_accuracy", "val_loss",
    "per_class_f1_json", "confusion_json",
    "epochs_run", "best_epoch", "collapsed_to_single_class",
    "train_seconds", "manifest_sha1", "split_sha1",
    "git_commit", "started_utc", "finished_utc", "status", "note",
]


def sha1_of(obj) -> str:
    return hashlib.sha1(str(obj).encode()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def load_done(results_path: Path) -> set[tuple]:
    if not results_path.exists():
        return set()
    df = pd.read_csv(results_path)
    if df.empty:
        return set()
    ok = df[df.get("status", "ok") == "ok"]
    return set(map(tuple, ok[KEY_FIELDS].astype(str).values))


def append_row(results_path: Path, row: dict) -> None:
    """Append one run and flush. Never hold results in memory across runs."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new = not results_path.exists()
    with results_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()


def build_queue(cfg: dict, done: set[tuple]) -> list[dict]:
    queue = []
    for arch in cfg["architectures"]:
        for seed in cfg["seeds"]:
            for arm in cfg["arms"]:            # innermost: arms complete first
                key = (str(cfg["dataset"]), str(arch), str(arm), str(seed))
                if key in done:
                    continue
                queue.append(dict(dataset=cfg["dataset"], arch=arch,
                                  arm=arm, seed=seed))
    return queue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the queue and exit without training")
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most N runs this session")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    done = load_done(args.results)
    queue = build_queue(cfg, done)

    total = len(cfg["architectures"]) * len(cfg["seeds"]) * len(cfg["arms"])
    print(f"{cfg['dataset']}: {total} runs in grid, {len(done)} done, "
          f"{len(queue)} remaining")
    if args.dry_run:
        for r in queue:
            print("  ", r)
        return

    # imported late so --dry-run works without TensorFlow present
    from data import load_split_manifest
    from train import train_one

    manifest = load_split_manifest(cfg)
    manifest_sha = sha1_of(pd.util.hash_pandas_object(manifest).sum())

    for i, run in enumerate(queue, 1):
        if args.limit and i > args.limit:
            print(f"reached --limit {args.limit}; stopping cleanly")
            break

        # the seed draws the split AND initialises the model, so mean +- SD
        # across seeds carries split variability as well as training variability
        split_col = f"split_{run['arm']}_seed{run['seed']}"
        if split_col not in manifest.columns:
            raise KeyError(f"{split_col} missing -- rebuild splits for this grid")
        sub = manifest
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.time()
        print(f"\n[{i}/{len(queue)}] {run['dataset']} | {run['arch']} | "
              f"{run['arm']} | seed {run['seed']}", flush=True)

        row = dict(run, started_utc=started, git_commit=git_commit(),
                   manifest_sha1=manifest_sha,
                   split_sha1=sha1_of(tuple(sub[split_col])))
        try:
            metrics = train_one(cfg=cfg, manifest=sub, split_col=split_col,
                                arch=run["arch"], seed=run["seed"])
            row.update(metrics)
            row["status"] = "ok"
        except Exception as exc:                       # keep the queue alive
            row["status"] = "failed"
            row["note"] = f"{type(exc).__name__}: {exc}"[:400]
            print("  FAILED:", row["note"], flush=True)

        row["train_seconds"] = round(time.time() - t0, 1)
        row["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        append_row(args.results, row)
        print(f"  -> {row['status']} in {row['train_seconds']}s "
              f"acc={row.get('test_accuracy')}", flush=True)

    print("\nqueue drained for this session")


if __name__ == "__main__":
    main()
