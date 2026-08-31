"""
check_slice_mapping.py -- does sct_slice_num point at the slice the radiologist meant?

NLST records a slice number for each >=4mm nodule, but never states which series
it indexes or which direction it counts. Every slice-level label in the study
depends on that mapping being right, and a wrong mapping fails silently -- the
labels look fine and are simply attached to the wrong images.

So: take a handful of very large nodules, render the slice each candidate rule
picks out, and look. A 60mm mass is unmistakable. Whichever column shows it is
the correct rule.

    python src/check_slice_mapping.py \
        --dicom-root /data/nlst_pilot \
        --targets pilot_targets.csv \
        --out pilot_sheets

Produces one PNG per nodule: rows are the series in that exam, columns are the
indexing rules, and each panel is captioned. No DICOM expertise needed to read
the output -- find the tumour, note the column heading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

# window used for display only, never for the saved training images
LUNG_WINDOW = (-600, 1500)   # level, width


def load_series(series_dir: Path):
    """Return instances sorted head->foot, plus their InstanceNumbers."""
    slices = []
    for p in series_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(p))
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "CT":
            continue
        z = None
        if hasattr(ds, "ImagePositionPatient"):
            z = float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            z = float(ds.SliceLocation)
        slices.append((z, int(getattr(ds, "InstanceNumber", 0)), ds))
    if not slices:
        return [], []
    # superior (higher z) first
    slices.sort(key=lambda t: (-t[0] if t[0] is not None else t[1]))
    return [s[2] for s in slices], [s[1] for s in slices]


def to_hu(ds):
    arr = ds.pixel_array.astype(np.float32)
    return arr * float(getattr(ds, "RescaleSlope", 1)) + float(
        getattr(ds, "RescaleIntercept", 0))


def windowed(ds):
    level, width = LUNG_WINDOW
    hu = to_hu(ds)
    lo, hi = level - width / 2, level + width / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def candidates(slice_num: int, ordered, instance_numbers):
    """The competing rules for what `slice_num` means."""
    n = len(ordered)
    i = int(slice_num)
    rules = {
        "1-based, top-down": i - 1,
        "0-based, top-down": i,
        "1-based, bottom-up": n - i,
        "0-based, bottom-up": n - i - 1,
    }
    out = {k: (v if 0 <= v < n else None) for k, v in rules.items()}
    # InstanceNumber is a separate hypothesis: match on the tag, not position
    match = [j for j, num in enumerate(instance_numbers) if num == i]
    out["InstanceNumber tag"] = match[0] if match else None
    return out


def sheet_for_nodule(row, dicom_root: Path, out_dir: Path) -> None:
    pid, yr = int(row.pid), int(row.study_yr)
    pat_dir = dicom_root / str(pid)
    if not pat_dir.is_dir():
        print(f"  no directory for pid {pid} -- skipped")
        return

    series_dirs = [d for d in pat_dir.rglob("*") if d.is_dir()
                   and any(d.glob("*.dcm"))]
    if not series_dirs:
        print(f"  no series with .dcm under {pat_dir}")
        return

    rows = []
    for sd in series_dirs:
        ordered, inums = load_series(sd)
        if len(ordered) < 20:            # scouts and dose reports
            continue
        rows.append((sd, ordered, inums))
    if not rows:
        return

    rule_names = list(candidates(row.slice, rows[0][1], rows[0][2]).keys())
    fig, axes = plt.subplots(len(rows), len(rule_names),
                             figsize=(3 * len(rule_names), 3 * len(rows)),
                             squeeze=False)

    for r, (sd, ordered, inums) in enumerate(rows):
        cands = candidates(row.slice, ordered, inums)
        thick = getattr(ordered[0], "SliceThickness", "?")
        for c, name in enumerate(rule_names):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            idx = cands[name]
            if idx is None:
                ax.text(0.5, 0.5, "out of range", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
            else:
                ax.imshow(windowed(ordered[idx]), cmap="gray")
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{sd.name[:18]}\n{len(ordered)} sl, {thick}mm",
                              fontsize=7)

    fig.suptitle(
        f"pid {pid}  T{yr}  recorded slice {int(row.slice)}   "
        f"{row.long_mm:.0f}x{row.perp_mm:.0f} mm in {row.lobe}",
        fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"pid{pid}_T{yr}.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"  wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", required=True, type=Path,
                    help="folder holding one subfolder per participant")
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("pilot_sheets"))
    args = ap.parse_args()

    targets = pd.read_csv(args.targets)
    for _, row in targets.iterrows():
        print(f"pid {int(row.pid)} T{int(row.study_yr)} "
              f"({row.long_mm:.0f}mm, {row.lobe}):")
        sheet_for_nodule(row, args.dicom_root, args.out)

    print("\nOpen each PNG. Find the tumour, note which column heading it sits "
          "under. If the same rule wins every time, that is the mapping. If "
          "different rules win on different scans, the slice numbers are not "
          "usable and we label per exam instead.")


if __name__ == "__main__":
    main()
