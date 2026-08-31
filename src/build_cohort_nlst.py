"""
build_cohort_nlst.py -- select the NLST participants for the leakage benchmark.

The output CSV *is* the cohort definition. Anyone with the public clinical
package can rerun this and get the identical participant list, which is what
makes the study reproducible without redistributing any imaging.

Selection rules, in order:

 1. CT arm only -- participants appearing in the screen table.
 2. Every exam must be diagnostically adequate (ctdxqual == 1). One bad scan
    excludes the participant, not just that exam, so the three split arms all
    see the same people.
 3. At least two usable screening results (scr_res in 1..6). The exam-level arm
    exists to show that a participant recurs across timepoints; a participant
    with a single exam cannot demonstrate that either way.
 4. Stratify on the participant's most severe screening result across exams,
    then take equal numbers per stratum with a fixed seed.

`--oversample` picks extra participants per stratum. TCIA hosts 26,254 of the
26,453 CT-arm participants, so a few will have no images. The download step
takes the first N available per stratum and records which reserves it used.

    python build_cohort_nlst.py --n 200 --oversample 40 --out cohort_nlst_200.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABEL_MAP = {1: "normal", 2: "benign", 3: "benign",
             4: "suspicious", 5: "suspicious", 6: "suspicious"}
SEVERITY = {"normal": 0, "benign": 1, "suspicious": 2}


def exam_table(prsn: pd.DataFrame) -> pd.DataFrame:
    """One row per (participant, screening year) with a three-class label."""
    long = prsn.melt(
        id_vars="pid",
        value_vars=["scr_res0", "scr_res1", "scr_res2"],
        var_name="col", value_name="scr_res",
    )
    long["study_yr"] = long.col.str[-1].astype(int)
    long["label"] = long.scr_res.map(LABEL_MAP)
    return long.dropna(subset=["label"])[["pid", "study_yr", "scr_res", "label"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prsn", default="nlst_780_prsn_idc_20210527.csv")
    ap.add_argument("--screen", default="nlst_780_screen_idc_20210527.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--oversample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out", type=Path, default=Path("cohort_nlst_200.csv"))
    args = ap.parse_args()

    prsn = pd.read_csv(args.prsn, low_memory=False)
    screen = pd.read_csv(args.screen, low_memory=False)

    # 1. CT arm
    prsn = prsn[prsn.pid.isin(set(screen.pid))]
    print(f"CT-arm participants                      {prsn.pid.nunique():>7,}")

    # 2. all exams diagnostically adequate
    bad = set(screen.loc[screen.ctdxqual != 1, "pid"])
    prsn = prsn[~prsn.pid.isin(bad)]
    print(f"after excluding non-diagnostic scans      {prsn.pid.nunique():>7,}")

    exams = exam_table(prsn)

    # 3. two or more usable exams
    keep = exams.groupby("pid").size()
    keep = set(keep[keep >= 2].index)
    exams = exams[exams.pid.isin(keep)]
    print(f"after requiring >=2 usable exams          {len(keep):>7,}")

    # 4. stratify on most severe result, sample equally
    stratum = (exams.assign(sev=exams.label.map(SEVERITY))
               .groupby("pid").sev.max()
               .map({v: k for k, v in SEVERITY.items()})
               .rename("stratum"))
    print("\navailable per stratum:")
    print(stratum.value_counts().to_string())

    per = args.n // 3
    extra = args.n - per * 3          # remainder goes to the largest stratum
    take = {"normal": per, "benign": per + extra, "suspicious": per}

    picked = []
    for name, want in take.items():
        pool = stratum[stratum == name].index.to_series()
        want_total = want + args.oversample
        if len(pool) < want_total:
            raise SystemExit(f"stratum {name} has {len(pool)}, need {want_total}")
        sel = pool.sample(want_total, random_state=args.seed).tolist()
        for rank, pid in enumerate(sel):
            picked.append(dict(pid=pid, stratum=name, rank=rank,
                               role="primary" if rank < want else "reserve"))

    cohort = pd.DataFrame(picked)
    out = (exams.merge(cohort, on="pid")
           .sort_values(["stratum", "rank", "study_yr"])
           [["pid", "stratum", "role", "rank", "study_yr", "scr_res", "label"]])

    out.to_csv(args.out, index=False)

    prim = out[out.role == "primary"]
    print(f"\nwrote {args.out}")
    print(f"  primary  {prim.pid.nunique():>4} participants, {len(prim):>5} exams")
    print(f"  reserve  {out[out.role=='reserve'].pid.nunique():>4} participants")
    print("\nexam labels in the primary cohort:")
    print(prim.label.value_counts().to_string())
    print("\nexams per primary participant:")
    print(prim.groupby("pid").size().value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
