"""
control_knn.py -- the control that separates memorisation from learning.

The trained models reach 57-77% test accuracy under a slice-level split and
fall to 41-44% under a participant-level split, which is within a few points of
the majority-class baseline. Two readings fit that pattern:

    (a) leakage inflated a real signal, or
    (b) there was never a usable signal, and the slice arm scored highly only
        because adjacent slices of the same scan sat on both sides of the split

A reviewer cannot choose between those from the training results alone, and
reading (b) would undercut the paper. This script settles it without training
anything.

A 1-nearest-neighbour classifier on heavily downsampled raw pixels cannot learn
a diagnostic feature. It can only retrieve the most similar image it has
already seen. Run it through the same three arms, on the same frozen splits:

    if 1-NN recovers most of the slice-arm accuracy and collapses to baseline
    under the participant arm, the inflation is retrieval of near-duplicate
    slices -- which is exactly what reading (a) claims the CNNs are also doing

Alongside the accuracy it reports where each test image's nearest neighbour
came from. Under a slice split, most test images should find their nearest
neighbour inside their own scan. That number is the mechanism stated directly,
and it is the figure most likely to convince a sceptical reviewer.

    python src/control_knn.py --config configs/nlst.yaml

Needs only CPU. Reads the frozen splits if make_splits.py has been run;
otherwise redraws them from the manifest with the same seeds, which is
deterministic and gives identical partitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.neighbors import NearestNeighbors

CLASSES = ["benign", "normal", "suspicious"]


def load_images(df: pd.DataFrame, images_root: str, size: int) -> np.ndarray:
    """One row per manifest row, greyscale, downsampled, flattened to [0, 1]."""
    vecs = np.empty((len(df), size * size), dtype=np.float32)
    for i, rel in enumerate(df["path"].to_numpy()):
        path = Path(images_root) / rel
        with Image.open(path) as im:
            arr = np.asarray(
                im.convert("L").resize((size, size), Image.BILINEAR),
                dtype=np.float32,
            )
        vecs[i] = arr.ravel() / 255.0
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


def run_cell(
    df: pd.DataFrame,
    vecs: np.ndarray,
    split_col: str,
    arm: str,
    seed: int,
) -> dict:
    part = df[split_col].to_numpy()
    tr = np.flatnonzero(part == "train")
    te = np.flatnonzero(part == "test")
    if len(tr) == 0 or len(te) == 0:
        raise SystemExit(f"{split_col} is missing a train or test partition")

    y = np.array([CLASSES.index(l) for l in df["label"]])
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean")
    nn.fit(vecs[tr])
    dist, idx = nn.kneighbors(vecs[te], return_distance=True)
    neigh = tr[idx[:, 0]]

    y_true, y_pred = y[te], y[neigh]
    cm = confusion(y_true, y_pred)
    acc = float((y_true == y_pred).mean()) * 100
    f1, per_class = macro_f1(cm)

    # where did the retrieved neighbour come from?
    pid = df["patient_id"].to_numpy()
    sid = df["study_uid"].to_numpy()
    same_patient = pid[te] == pid[neigh]
    same_exam = sid[te] == sid[neigh]

    # accuracy split by whether the neighbour was the test image's own patient
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
        "method": "knn1_pixels",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--images-root", default=None,
                    help="override images_root from the config")
    ap.add_argument("--size", type=int, default=32,
                    help="downsample edge in pixels (default 32)")
    ap.add_argument("--out", type=Path, default=Path("data/results_control_knn.csv"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    root = args.images_root or cfg["images_root"]

    manifest = pd.read_csv(cfg["manifest"])
    df = split_columns(cfg, manifest)

    print(f"loading {len(df):,} images at {args.size}x{args.size} from {root}")
    vecs = load_images(df, root, args.size)

    rows = []
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            col = f"split_{arm}_seed{seed}"
            if col not in df.columns:
                raise SystemExit(f"missing split column {col}")
            rows.append(run_cell(df, vecs, col, arm, seed))
            r = rows[-1]
            print(
                f"  {arm:8} seed {seed}  acc {r['test_accuracy']:6.2f}  "
                f"(baseline {r['majority_baseline']:.2f})  "
                f"NN same-patient {r['pct_nn_same_patient']:6.2f}%  "
                f"same-exam {r['pct_nn_same_exam']:6.2f}%"
            )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    summary = (
        out.groupby("arm", sort=False)
        .agg(
            acc_mean=("test_accuracy", "mean"),
            acc_sd=("test_accuracy", "std"),
            baseline=("majority_baseline", "mean"),
            nn_same_patient=("pct_nn_same_patient", "mean"),
            nn_same_exam=("pct_nn_same_exam", "mean"),
        )
        .round(2)
    )
    print("\n1-NN on raw pixels, mean across seeds:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
