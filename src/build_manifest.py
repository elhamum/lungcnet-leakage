"""
build_manifest.py -- turn a DICOM tree into one CSV row per image, carrying the
identifiers every split arm needs.

Run this before anything else. It also verifies the assumptions the study
depends on, which is the whole reason IQ-OTH/NCCD could not support this
design: identifiers must actually be present, populated, and non-constant.

    python src/build_manifest.py --root /data/LIDC-IDRI --dataset lidc \
        --out manifests/lidc_raw.csv

    python src/build_manifest.py --root /data/NLST --dataset nlst \
        --out manifests/nlst_raw.csv

Output columns:
    image_id, path, patient_id, study_uid, series_uid, sop_uid,
    study_date, series_desc, modality, slice_thickness, rows, cols,
    timepoint (nlst), legacy_pid (nlst)

`label` is NOT produced here -- labels come from a separate join (LIDC XML
annotations, or the NLST outcome table). Keeping them apart means a labelling
mistake never silently reshapes the splits.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:  # pragma: no cover
    sys.exit("pydicom is required:  pip install pydicom")

DICOM_SUFFIXES = {".dcm", ".dicom", ""}

# NLST carries prior ACRIN/LSS identifiers in these private/legacy slots
LEGACY_PID_TAGS = [(0x0010, 0x1000), (0x0013, 0x0040)]


def _read_header(path: str) -> dict | None:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
    except (InvalidDicomError, OSError, AttributeError):
        return None

    def g(name, default=""):
        v = getattr(ds, name, default)
        return "" if v is None else str(v).strip()

    legacy = ""
    for tag in LEGACY_PID_TAGS:
        if tag in ds:
            legacy = str(ds[tag].value).strip()
            break

    return dict(
        path=path,
        patient_id=g("PatientID"),
        study_uid=g("StudyInstanceUID"),
        series_uid=g("SeriesInstanceUID"),
        sop_uid=g("SOPInstanceUID"),
        study_date=g("StudyDate"),
        series_desc=g("SeriesDescription"),
        modality=g("Modality"),
        slice_thickness=g("SliceThickness"),
        instance_number=g("InstanceNumber"),
        rows=g("Rows"),
        cols=g("Columns"),
        legacy_pid=legacy,
    )


def candidate_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in DICOM_SUFFIXES:
                yield os.path.join(dirpath, fn)


def scan(root: Path, workers: int = 8, modality: str = "CT") -> pd.DataFrame:
    files = list(candidate_files(root))
    if not files:
        sys.exit(f"no candidate DICOM files under {root}")
    print(f"scanning {len(files):,} candidate files under {root} ...", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rec in enumerate(ex.map(_read_header, files, chunksize=64), 1):
            if rec is not None:
                rows.append(rec)
            if i % 20000 == 0:
                print(f"  {i:,} / {len(files):,}", flush=True)

    df = pd.DataFrame(rows)
    unreadable = len(files) - len(df)
    if unreadable:
        print(f"  {unreadable:,} files were not readable DICOM (skipped)")
    if modality:
        before = len(df)
        df = df[df.modality == modality]
        print(f"  kept {len(df):,} / {before:,} instances with Modality == {modality}")
    return df.reset_index(drop=True)


def add_nlst_timepoints(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each participant's studies in date order -> T0 / T1 / T2.

    NLST study dates are deliberately fictitious. They are internally coded so
    that the ORDER of the three annual screens is preserved, but any interval
    computed from them is meaningless. Order only; never subtract these dates.
    """
    df = df.copy()
    per_study = (
        df.groupby(["patient_id", "study_uid"], as_index=False)["study_date"]
        .min()
        .sort_values(["patient_id", "study_date", "study_uid"])
    )
    per_study["timepoint"] = (
        "T" + per_study.groupby("patient_id").cumcount().astype(str)
    )

    ties = (
        per_study.groupby(["patient_id", "study_date"]).size().gt(1).sum()
    )
    if ties:
        print(f"  WARNING: {ties} patient/date pairs have >1 study on the same "
              f"coded date -- timepoint order is ambiguous for those; "
              f"tie-broken by study_uid")

    return df.merge(
        per_study[["patient_id", "study_uid", "timepoint"]],
        on=["patient_id", "study_uid"],
        how="left",
    )


def validate(df: pd.DataFrame, dataset: str) -> None:
    """Check the identifiers the split arms rely on. Fails loudly, not silently."""
    print("\n--- identifier check " + "-" * 52)
    problems = []

    n_missing = (df.patient_id == "").sum()
    if n_missing:
        problems.append(f"{n_missing:,} instances have an empty PatientID")
    if df.patient_id.nunique() <= 1:
        problems.append("PatientID is constant across the cohort -- "
                        "patient-level splitting is impossible")

    dup = df.sop_uid.duplicated().sum()
    if dup:
        problems.append(f"{dup:,} duplicated SOPInstanceUIDs "
                        f"(same instance stored twice?)")

    # a series must not straddle two patients or two studies
    for child, parent in (("series_uid", "patient_id"), ("series_uid", "study_uid"),
                          ("study_uid", "patient_id")):
        bad = df.groupby(child)[parent].nunique().gt(1).sum()
        if bad:
            problems.append(f"{bad} values of {child} map to more than one {parent}")

    n_pat = df.patient_id.nunique()
    n_stu = df.study_uid.nunique()
    n_ser = df.series_uid.nunique()
    print(f"patients {n_pat:,} | studies {n_stu:,} | series {n_ser:,} | "
          f"instances {len(df):,}")
    print(f"studies per patient: "
          f"{df.groupby('patient_id').study_uid.nunique().describe()[['mean','min','max']].to_dict()}")

    if dataset == "nlst":
        multi = df.groupby("patient_id").study_uid.nunique().gt(1).mean()
        print(f"participants with more than one screening exam: {100 * multi:.1f}%")
        if multi < 0.10:
            problems.append(
                "fewer than 10% of participants recur across timepoints -- the "
                "exam-level arm will barely differ from the patient-level arm, "
                "which removes the paper's main contribution. Re-check the "
                "subset selection before running anything."
            )
        if "legacy_pid" in df and (df.legacy_pid != "").any():
            n = df.loc[df.legacy_pid != ""].groupby("patient_id").legacy_pid.nunique()
            if n.gt(1).any():
                problems.append(f"{n.gt(1).sum()} participants carry more than one "
                                f"legacy PID -- confirm which identifier is stable")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  ! " + p)
    else:
        print("no identifier problems found")
    print("-" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=["lidc", "nlst"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--modality", default="CT")
    args = ap.parse_args()

    df = scan(args.root, workers=args.workers, modality=args.modality)
    if args.dataset == "nlst":
        df = add_nlst_timepoints(df)

    df.insert(0, "image_id", df.sop_uid.where(df.sop_uid != "",
                                              other=df.path.map(lambda p: Path(p).stem)))
    df.insert(1, "dataset", args.dataset)
    validate(df, args.dataset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
