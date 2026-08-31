"""
analyse.py -- turn results_nlst.csv into the paper's tables.

Everything here is recomputed from the stored confusion matrices rather than
read from the flags written at training time. That matters: the degeneracy check
in `train.py` was wrong for the first runs of this study, and deriving it here
means every run is judged by the same rule regardless of when it was executed.

    python src/analyse.py --results results_nlst.csv --out tables/

Produces:
    table_arm_by_arch.csv    macro F1, mean +- SD across seeds
    table_inflation.csv      the headline: F1 points added by each leaky split
    table_per_class.csv      per-class F1 by arm -- where the failure actually is
    table_degenerate.csv     runs that never predicted some class
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["benign", "normal", "suspicious"]
ARM_ORDER = ["slice", "exam", "patient"]


def from_confusion(row) -> dict:
    """Recompute per-class F1, accuracy and the majority baseline from the matrix.

    cm[i][j] = true class i predicted as j.
    """
    cm = np.array(json.loads(row["confusion_json"]), dtype=float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)

    prec = np.divide(correct, predicted, out=np.zeros_like(correct),
                     where=predicted > 0)
    rec = np.divide(correct, support, out=np.zeros_like(correct),
                    where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(correct),
                   where=(prec + rec) > 0)

    never = [CLASSES[i] for i in range(len(CLASSES)) if predicted[i] == 0]
    return dict(
        accuracy=100 * correct.sum() / cm.sum(),
        majority_baseline=100 * support.max() / support.sum(),
        macro_f1=100 * f1.mean(),
        **{f"f1_{c}": 100 * f1[i] for i, c in enumerate(CLASSES)},
        n_never_predicted=len(never),
        never_predicted=";".join(never),
        degenerate=bool(never),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("tables"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    r = pd.read_csv(args.results)
    r = r[r.status == "ok"].copy()
    if r.empty:
        raise SystemExit("no successful runs in the results file")

    derived = pd.DataFrame([from_confusion(x) for _, x in r.iterrows()])
    d = pd.concat([r[["arch", "arm", "seed", "epochs_run"]].reset_index(drop=True),
                   derived], axis=1)
    d["arm"] = pd.Categorical(d.arm, ARM_ORDER, ordered=True)
    d["above_baseline"] = d.accuracy - d.majority_baseline

    # --- macro F1 by architecture and arm -----------------------------------
    t1 = (d.groupby(["arch", "arm"], observed=True)
          .agg(macro_f1_mean=("macro_f1", "mean"),
               macro_f1_sd=("macro_f1", "std"),
               accuracy_mean=("accuracy", "mean"),
               above_baseline_mean=("above_baseline", "mean"),
               epochs_mean=("epochs_run", "mean"),
               n_seeds=("macro_f1", "size"))
          .round(2).reset_index())
    t1.to_csv(args.out / "table_arm_by_arch.csv", index=False)

    # --- the headline: inflation relative to the correct split ---------------
    piv = d.pivot_table(index="arch", columns="arm", values="macro_f1",
                        aggfunc="mean", observed=True)
    t2 = pd.DataFrame(index=piv.index)
    if "patient" in piv:
        for leaky in ("slice", "exam"):
            if leaky in piv:
                t2[f"{leaky}_minus_patient"] = (piv[leaky] - piv["patient"]).round(2)
    t2 = t2.reset_index()
    t2.to_csv(args.out / "table_inflation.csv", index=False)

    # --- per-class F1: where the failure actually is -------------------------
    t3 = (d.groupby(["arch", "arm"], observed=True)[[f"f1_{c}" for c in CLASSES]]
          .mean().round(2).reset_index())
    t3.to_csv(args.out / "table_per_class.csv", index=False)

    # --- degenerate runs ------------------------------------------------------
    t4 = d[d.degenerate][["arch", "arm", "seed", "macro_f1", "accuracy",
                          "majority_baseline", "never_predicted"]].round(2)
    t4.to_csv(args.out / "table_degenerate.csv", index=False)

    print("macro F1 by architecture and split arm\n")
    print(t1.to_string(index=False))
    print("\n\ninflation from leakage (macro F1 points vs patient-level split)\n")
    print(t2.to_string(index=False))
    print("\n\nper-class F1 (mean across seeds)\n")
    print(t3.to_string(index=False))
    print(f"\n\ndegenerate runs: {len(t4)} of {len(d)}")
    if len(t4):
        print(t4.to_string(index=False))
        print("\nThese runs never predicted the listed class. Their macro F1 is not "
              "comparable to a run that predicts all three, and they must be "
              "reported as such rather than averaged in silently.")

    below = d[d.above_baseline <= 0]
    if len(below):
        print(f"\n\nruns at or below the majority-class baseline: {len(below)}")
        print(below[["arch", "arm", "seed", "accuracy",
                     "majority_baseline"]].round(2).to_string(index=False))

    print(f"\ntables written to {args.out}/")


if __name__ == "__main__":
    main()
