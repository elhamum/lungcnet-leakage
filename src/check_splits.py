"""
check_splits.py -- run the splitter against synthetic cohorts shaped like NLST
and LIDC-IDRI, before any real data exists.

This is not a unit test for its own sake. It answers the question the reviewer
will ask: does the exam-level arm actually leak, and by how much, given a
realistic amount of longitudinal recurrence? If the synthetic answer is "barely
at all", the three-arm design needs rethinking before compute is spent.

    python src/check_splits.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from splits import (
    ARM_GROUP_COL,
    assert_pool_identical,
    audit_leakage,
    build_all_arms,
    split_size_report,
)


def synth_nlst(n_patients=300, seed=0) -> pd.DataFrame:
    """Participants recur across T0/T1/T2 screens; label is a screening outcome."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"1{p:05d}"
        label = rng.choice(["benign", "malignant", "normal"], p=[0.35, 0.15, 0.50])
        n_visits = rng.choice([1, 2, 3], p=[0.15, 0.25, 0.60])
        for t in range(n_visits):
            study = f"{pid}.T{t}"
            n_series = rng.choice([1, 2], p=[0.7, 0.3])
            for s in range(n_series):
                series = f"{study}.S{s}"
                for k in range(int(rng.integers(6, 30))):
                    rows.append(
                        dict(
                            image_id=f"{series}.I{k:03d}",
                            patient_id=pid,
                            study_uid=study,
                            series_uid=series,
                            timepoint=f"T{t}",
                            label=label,
                        )
                    )
    return pd.DataFrame(rows)


def synth_lidc(n_patients=400, seed=0) -> pd.DataFrame:
    """One study per patient, several nodules per patient, label per nodule."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"LIDC-IDRI-{p:04d}"
        study = f"{pid}.ST"
        series = f"{study}.SE"
        for n in range(int(rng.integers(1, 5))):
            nodule = f"{pid}.N{n}"
            label = rng.choice(["benign", "malignant", "indeterminate"], p=[0.45, 0.30, 0.25])
            for k in range(int(rng.integers(3, 15))):
                rows.append(
                    dict(
                        image_id=f"{nodule}.I{k:03d}",
                        patient_id=pid,
                        study_uid=study,
                        series_uid=series,
                        nodule_key=nodule,
                        label=label,
                    )
                )
    return pd.DataFrame(rows)


def report(name: str, manifest: pd.DataFrame, arms, seed=0) -> None:
    print(f"\n{'=' * 72}\n{name}  --  {len(manifest):,} images, "
          f"{manifest.patient_id.nunique()} patients, "
          f"{manifest.study_uid.nunique()} exams\n{'=' * 72}")
    split = build_all_arms(manifest, arms, seed=seed)
    assert_pool_identical(split, arms)

    print("\nPartition sizes (must be comparable across arms):")
    print(split_size_report(split, arms).to_string(index=False,
          float_format=lambda v: f"{v:6.2f}"))

    print("\nLeakage audit:")
    audit = audit_leakage(split, arms)
    print(audit.to_string(index=False))

    # the assertion the design depends on
    for arm in arms:
        row = audit[(audit.arm == arm) & (audit.key == "patient_id")
                    & (audit.partition == "test")]
        if not row.empty:
            print(f"  {arm:8s} -> {row.pct_leaked.iloc[0]:6.2f}% of test images "
                  f"share a patient with train")


def main() -> None:
    report("NLST-shaped (longitudinal)", synth_nlst(),
           arms=["slice", "exam", "patient"])
    report("LIDC-shaped (multi-nodule)", synth_lidc(),
           arms=["slice", "nodule", "patient"])

    print("\nSeed stability of the patient arm (NLST-shaped):")
    m = synth_nlst()
    for s in (0, 1, 2):
        sp = build_all_arms(m, ["patient"], seed=s)
        n_test = (sp.split_patient == "test").sum()
        print(f"  seed {s}: test = {n_test:,} images "
              f"({100 * n_test / len(sp):.2f}% of pool)")


if __name__ == "__main__":
    main()
