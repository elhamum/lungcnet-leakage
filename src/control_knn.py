"""
control_knn.py -- the control that separates memorisation from learning.

The trained models reach 57-77% test accuracy under a slice-level split and
fall to 41-44% under a participant-level split, which is within a few points of
the majority-class baseline. Two readings fit that pattern:

    (a) leakage inflated a real signal, or
    (b) there was never a usable signal, and the slice arm scored highly only
        because adjacent slices of the same scan sat on both sides of the split

A k-nearest-neighbour classifier on heavily downsampled raw pixels cannot learn
a diagnostic feature. It can only retrieve the most similar images it has
already seen. Run it through the same three arms, on the same frozen splits:

    if it recovers most of the slice-arm accuracy and collapses to baseline
    under the participant arm, the inflation is retrieval of near-duplicate
    slices -- which is exactly what reading (a) claims the CNNs are also doing

Alongside the accuracy it reports where each test image's nearest neighbour
came from. Under a slice split, most test images should find their nearest
neighbour inside their own scan. That number is the mechanism stated directly.

Several resolutions and several values of k can be swept in one run, which is
what answers the two objections a reviewer raises first: that the control is
weak because the input is tiny, and that one neighbour is the weakest possible
rule. Images are read once per resolution and cached, so extra k values are
free.

    python control_knn.py --config configs/nlst.yaml
    python control_knn.py --config configs/nlst.yaml --size 32 64 --k 1 5

Needs only CPU. Reads the frozen splits if make_splits.py has been run;
otherwise redraws them from the manifest with the same seeds, which is
deterministic and gives identical partitions.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.neighbors import NearestNeighbors

CLASSES = ["benign", "normal", "suspicious"]


def load_images(
    df: pd.DataFrame, images_root: str, size: int, workers: int = 32
) -> np.ndarray:
    """One row per manifest row, greyscale, downsampled, flattened to [0, 1].

    Threaded on purpose. On Colab the images sit behind the Google Drive FUSE
    mount, where each file open costs a network round trip, so a serial loop
    over ~5,800 files takes tens of minutes while the actual decoding is
    trivial. The work is I/O bound and releases the GIL, so threads help.
    """
    n = len(df)
    vecs = np.empty((n, size * size), dtype=np.float32)
    paths = [Path(images_root) / p for p in df["path"].to_numpy()]
    done = 0
    step = max(250, n // 20)

    def one(i: int) -> int:
        with Image.open(paths[i]) as im:
            arr = np.asarray(
                im.convert("L").resize((size, size), Image.BILINEAR),
                dtype=np.float32,
            )
        vecs[i] = arr.ravel() / 255.0
        return i

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in as_completed([pool.submit(one, i) for i in range(n)]):
            done += 1
            if done % step == 0 or done == n:
                rate = done / max(time.time() - t0, 1e-6)
                print(
                    f"    {done:,}/{n:,} images  ({rate:,.0f}/s, "
                    f"{(n - done) / rate:,.0f}s left)",
                    flush=True,
                )
    return vecs


def split_columns(cfg: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    """Frozen splits if they exist on disk, otherwise redraw them identically."""
    frozen = Path(cfg["splits"])
    if frozen.exists():
        df = pd.read_csv(frozen)
        if len(df) != len(manifest):
            raise SystemExit(
                f"{frozen} has {len(df)} rows but the manifest has {len(manifest)}; "
                "they are out of step, regenerate the splits"
            )
        print(f"using frozen splits from {frozen}")
        return df

    print(f"{frozen} not found -- redrawing splits from the manifest (deterministic)")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from splits import make_split

    out = manifest.copy()
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            out[f"split_{arm}_seed{seed}"] = make_split(
                manifest,
                arm,
                seed=seed,
                fractions=tuple(cfg.get("fractions", (0.70, 0.15, 0.15))),
                jitter=cfg.get("jitter", 0.35),
            )
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1(cm: np.ndarray) -> tuple[float, list[float]]:
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)
    f1 = []
    for i in range(len(CLASSES)):
        prec = correct[i] / predicted[i] if predicted[i] else 0.0
        rec = correct[i] / support[i] if support[i] else 0.0
        f1.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1)) * 100, [round(v * 100, 2) for v in f1]


def vote(labels: np.ndarray) -> np.ndarray:
    """Majority label over each row of neighbour labels.

    Ties go to the nearest neighbour, because `labels` arrives ordered by
    distance and the first occurrence of a tied count is therefore the closest.
    """
    out = np.empty(len(labels), dtype=int)
    for i, row in enumerate(labels):
        counts = np.bincount(row, minlength=len(CLASSES))
        best = counts.max()
        tied = {c for c in range(len(CLASSES)) if counts[c] == best}
        out[i] = next(c for c in row if c in tied)
    return out


def run_cell(
    df: pd.DataFrame,
    vecs: np.ndarray,
    split_col: str,
    arm: str,
    seed: int,
    size: int,
    k: int,
) -> dict:
    part = df[split_col].to_numpy()
    tr = np.flatnonzero(part == "train")
    te = np.flatnonzero(part == "test")
    if len(tr) == 0 or len(te) == 0:
        raise SystemExit(f"{split_col} is missing a train or test partition")

    y = np.array([CLASSES.index(l) for l in df["label"]])
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn.fit(vecs[tr])
    dist, idx = nn.kneighbors(vecs[te], return_distance=True)
    neigh = tr[idx]                      # (n_test, k), ordered by distance
    top1 = neigh[:, 0]

    y_true = y[te]
    y_pred = vote(y[neigh]) if k > 1 else y[top1]
    cm = confusion(y_true, y_pred)
    acc = float((y_true == y_pred).mean()) * 100
    f1, per_class = macro_f1(cm)

    # Provenance is always measured on the single nearest neighbour, whatever k
    # is: it is a statement about what the test image most resembles, not about
    # the voting rule.
    pid = df["patient_id"].to_numpy()
    sid = df["study_uid"].to_numpy()
    same_patient = pid[te] == pid[top1]
    same_exam = sid[te] == sid[top1]

    acc_same = (
        float((y_true[same_patient] == y_pred[same_patient]).mean()) * 100
        if same_patient.any()
        else float("nan")
    )
    acc_diff = (
        float((y_true[~same_patient] == y_pred[~same_patient]).mean()) * 100
        if (~same_patient).any()
        else float("nan")
    )

    support = cm.sum(axis=1)
    majority = float(support.max() / support.sum()) * 100

    return {
        "dataset": "nlst",
        "method": f"knn{k}_pixels{size}",
        "size": size,
        "k": k,
        "arm": arm,
        "seed": seed,
        "n_train": len(tr),
        "n_test": len(te),
        "test_accuracy": round(acc, 2),
        "test_macro_f1": round(f1, 2),
        "majority_baseline": round(majority, 2),
        "above_baseline": round(acc - majority, 2),
        "pct_nn_same_patient": round(float(same_patient.mean()) * 100, 2),
        "pct_nn_same_exam": round(float(same_exam.mean()) * 100, 2),
        "acc_when_nn_same_patient": None if np.isnan(acc_same) else round(acc_same, 2),
        "acc_when_nn_other_patient": None if np.isnan(acc_diff) else round(acc_diff, 2),
        "mean_nn_distance": round(float(dist[:, 0].mean()), 4),
        "per_class_f1_json": json.dumps(dict(zip(CLASSES, per_class))),
        "confusion_json": json.dumps(cm.tolist()),
    }


def pixels_for(df, root, size, args):
    """Load this resolution, using the cache directory when one is given."""
    cache = Path(args.cache_dir) / f"pixels{size}.npy" if args.cache_dir else None
    if cache and cache.exists():
        vecs = np.load(cache)
        if vecs.shape == (len(df), size * size):
            print(f"  {size}x{size}: loaded from cache {cache}")
            return vecs
        print(f"  {size}x{size}: cache {cache} has wrong shape, re-reading")
    print(
        f"  {size}x{size}: reading {len(df):,} images from {root} "
        f"({args.workers} parallel readers)",
        flush=True,
    )
    vecs = load_images(df, root, size, workers=args.workers)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vecs)
        print(f"  {size}x{size}: cached to {cache}")
    return vecs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--images-root", default=None,
                    help="override images_root from the config")
    ap.add_argument("--size", type=int, nargs="+", default=[32],
                    help="downsample edge(s) in pixels, e.g. --size 32 64")
    ap.add_argument("--k", type=int, nargs="+", default=[1],
                    help="neighbour count(s), e.g. --k 1 5")
    ap.add_argument("--out", type=Path, default=Path("data/results_control_knn.csv"))
    ap.add_argument("--workers", type=int, default=32,
                    help="parallel readers; raise it if Drive is slow")
    ap.add_argument("--cache-dir", default=None,
                    help="directory for pixels<size>.npy, so reruns skip the read")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    root = args.images_root or cfg["images_root"]

    manifest = pd.read_csv(cfg["manifest"])
    df = split_columns(cfg, manifest)

    rows = []
    for size in args.size:
        vecs = pixels_for(df, root, size, args)
        for k in args.k:
            for arm in cfg["arms"]:
                for seed in cfg["seeds"]:
                    col = f"split_{arm}_seed{seed}"
                    if col not in df.columns:
                        raise SystemExit(f"missing split column {col}")
                    r = run_cell(df, vecs, col, arm, seed, size, k)
                    rows.append(r)
                    print(
                        f"  {size}px k={k} {arm:8} seed {seed}  "
                        f"acc {r['test_accuracy']:6.2f}  "
                        f"(baseline {r['majority_baseline']:.2f})  "
                        f"NN same-patient {r['pct_nn_same_patient']:6.2f}%",
                        flush=True,
                    )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    summary = (
        out.groupby(["size", "k", "arm"], sort=False)
        .agg(
            acc_mean=("test_accuracy", "mean"),
            acc_sd=("test_accuracy", "std"),
            macro_f1=("test_macro_f1", "mean"),
            baseline=("majority_baseline", "mean"),
            nn_same_patient=("pct_nn_same_patient", "mean"),
        )
        .round(2)
    )
    print("\nretrieval control, mean across seeds:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
