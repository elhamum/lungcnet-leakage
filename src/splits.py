"""
splits.py -- split arms for the leakage benchmarking study.

Every arm draws from the SAME image pool. The only thing that changes between
arms is the unit that is kept intact when the pool is partitioned:

    slice    -- each image is its own group (the common, leaky practice)
    series   -- one DICOM series kept intact
    exam     -- one study/visit kept intact  (NLST: one screening timepoint)
    patient  -- all images of a participant kept intact (correct)
    nodule   -- LIDC optional arm: one nodule kept intact

Two properties matter for the paper and are enforced here:

1. Matched partition sizes. If the patient-level arm ends up with a smaller
   test partition than the slice-level arm, a reviewer will say the delta is a
   sample-size effect, not leakage. The greedy allocator below targets the same
   per-class image counts in every arm and `split_size_report` shows how close
   it got.

2. Quantified leakage. `audit_leakage` reports, for each arm, what fraction of
   test images come from a patient (or exam) that also appears in train. That
   turns "arm 2 still leaks" from an assertion into a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# arm name -> manifest column holding the group key
ARM_GROUP_COL = {
    "slice": "image_id",
    "series": "series_uid",
    "exam": "study_uid",
    "patient": "patient_id",
    "nodule": "nodule_key",
}

PARTITIONS = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)


def make_split(
    manifest: pd.DataFrame,
    arm: str,
    seed: int,
    fractions=DEFAULT_FRACTIONS,
    partitions=PARTITIONS,
    label_col: str = "label",
    jitter: float = 0.35,
) -> pd.Series:
    """Assign every row of `manifest` to a partition, keeping `arm` groups intact.

    Returns a Series aligned to manifest.index with values from `partitions`.

    Greedy allocation: groups are visited roughly largest-first and each is
    placed in whichever partition is currently furthest below its per-class
    target. Largest-first matters -- placing big groups last is what makes
    group splits drift away from the requested sizes.

    `jitter` perturbs the size ordering (lognormal, sigma=jitter) so that
    different seeds produce genuinely different partitions rather than the same
    split with the small groups shuffled. jitter=0 gives strict largest-first,
    which matches sizes marginally better but makes seeds nearly redundant.
    """
    if arm not in ARM_GROUP_COL:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_GROUP_COL)}")
    group_col = ARM_GROUP_COL[arm]
    if group_col not in manifest.columns:
        raise KeyError(
            f"arm {arm!r} needs column {group_col!r}, which is not in the manifest"
        )
    if manifest[group_col].isna().any():
        raise ValueError(f"column {group_col!r} has missing values; cannot group safely")
    if len(fractions) != len(partitions):
        raise ValueError("fractions and partitions must be the same length")

    labels = sorted(manifest[label_col].unique())
    counts = (
        manifest.groupby([group_col, label_col], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
    )
    group_keys = counts.index.to_numpy()
    mat = counts.to_numpy(dtype=float)          # (n_groups, n_classes)
    class_totals = mat.sum(axis=0)              # (n_classes,)
    targets = np.outer(np.asarray(fractions, dtype=float), class_totals)  # (P, C)
    denom = np.maximum(targets, 1.0)            # avoid div-by-zero on empty cells

    rng = np.random.default_rng(seed)
    sizes = mat.sum(axis=1)
    if jitter > 0:
        key = sizes * rng.lognormal(0.0, jitter, size=len(sizes))
    else:
        key = sizes + rng.random(len(sizes)) * 1e-6
    order = np.argsort(-key, kind="stable")

    alloc = np.zeros_like(targets)
    assignment = {}
    for gi in order:
        vec = mat[gi]
        costs = []
        for p in range(len(partitions)):
            trial = alloc.copy()
            trial[p] += vec
            costs.append(np.sum(((trial - targets) / denom) ** 2))
        best = int(np.argmin(costs))
        alloc[best] += vec
        assignment[group_keys[gi]] = partitions[best]

    return manifest[group_col].map(assignment)


def build_all_arms(
    manifest: pd.DataFrame,
    arms,
    seed: int,
    fractions=DEFAULT_FRACTIONS,
    label_col: str = "label",
) -> pd.DataFrame:
    """Return a copy of `manifest` with one `split_<arm>` column per arm."""
    out = manifest.copy()
    for arm in arms:
        out[f"split_{arm}"] = make_split(
            manifest, arm, seed=seed, fractions=fractions, label_col=label_col
        )
    return out


def split_size_report(
    split_manifest: pd.DataFrame, arms, label_col: str = "label"
) -> pd.DataFrame:
    """Image counts per arm x partition, plus per-class counts.

    Read this before running anything. If the test partitions differ by more
    than a couple of percent between arms, fix that first -- it contaminates
    the delta the whole paper rests on.
    """
    rows = []
    for arm in arms:
        col = f"split_{arm}"
        for part, sub in split_manifest.groupby(col, observed=True):
            row = {
                "arm": arm,
                "partition": part,
                "n_images": len(sub),
                "pct_of_pool": 100 * len(sub) / len(split_manifest),
            }
            for lab, n in sub[label_col].value_counts().items():
                row[f"n_{lab}"] = int(n)
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["arm", "partition"])
        .reset_index(drop=True)
        .fillna(0)
    )


def audit_leakage(
    split_manifest: pd.DataFrame,
    arms,
    keys=("patient_id", "study_uid"),
    train="train",
    held_out=("val", "test"),
) -> pd.DataFrame:
    """How much of each held-out partition is contaminated, per arm.

    For each key (patient, exam), reports:
      shared_groups   -- groups present in both train and the held-out partition
      pct_leaked      -- % of held-out images whose key also occurs in train

    Expected shape of the result, if the design is sound:
      slice arm    -> pct_leaked near 100 for patient and exam
      exam arm     -> pct_leaked 0 for exam, well above 0 for patient (NLST
                      longitudinal recurrence -- the paper's main point)
      patient arm  -> pct_leaked exactly 0 for both
    """
    rows = []
    for arm in arms:
        col = f"split_{arm}"
        for key in keys:
            if key not in split_manifest.columns:
                continue
            train_keys = set(split_manifest.loc[split_manifest[col] == train, key])
            for part in held_out:
                sub = split_manifest[split_manifest[col] == part]
                if sub.empty:
                    continue
                leaked = sub[key].isin(train_keys)
                rows.append(
                    {
                        "arm": arm,
                        "key": key,
                        "partition": part,
                        "n_images": len(sub),
                        "n_leaked_images": int(leaked.sum()),
                        "pct_leaked": round(100 * leaked.mean(), 2),
                        "shared_groups": int(sub.loc[leaked, key].nunique()),
                        "n_groups": int(sub[key].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def assert_pool_identical(split_manifest: pd.DataFrame, arms) -> None:
    """Every arm must partition the same rows -- no arm may silently drop images."""
    for arm in arms:
        col = f"split_{arm}"
        missing = split_manifest[col].isna().sum()
        if missing:
            raise AssertionError(f"arm {arm!r} left {missing} images unassigned")


if __name__ == "__main__":
    import sys

    print(__doc__)
    sys.exit(0)
