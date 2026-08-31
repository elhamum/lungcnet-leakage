"""
make_splits.py -- freeze the splits for a whole config, once, before training.

Writes one `split_<arm>_seed<n>` column per cell of the grid, so every run
reads its partition from disk rather than redrawing it. The size report and the
leakage audit are written alongside; both are paper tables, not debug output.

    python src/make_splits.py --config configs/nlst.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from splits import audit_leakage, make_split, split_size_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    manifest = pd.read_csv(cfg["manifest"])
    if "label" not in manifest.columns:
        raise SystemExit("manifest has no `label` column -- run the label join first")

    out = manifest.copy()
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            out[f"split_{arm}_seed{seed}"] = make_split(
                manifest, arm, seed=seed,
                fractions=tuple(cfg.get("fractions", (0.70, 0.15, 0.15))),
                jitter=cfg.get("jitter", 0.35),
            )

    dest = Path(cfg["splits"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(out):,} images x "
          f"{len(cfg['arms']) * len(cfg['seeds'])} split columns)")

    # per-seed reports, renamed so the shared report functions can be reused
    sizes, audits = [], []
    for seed in cfg["seeds"]:
        view = out.rename(
            columns={f"split_{a}_seed{seed}": f"split_{a}" for a in cfg["arms"]}
        )
        s = split_size_report(view, cfg["arms"]); s["seed"] = seed; sizes.append(s)
        a = audit_leakage(view, cfg["arms"]); a["seed"] = seed; audits.append(a)

    size_df, audit_df = pd.concat(sizes), pd.concat(audits)
    size_df.to_csv(dest.with_name(dest.stem + "_sizes.csv"), index=False)
    audit_df.to_csv(dest.with_name(dest.stem + "_leakage.csv"), index=False)

    print("\npartition sizes:")
    print(size_df.to_string(index=False))
    print("\nleakage audit (test partition, patient key):")
    print(audit_df[(audit_df.partition == "test") &
                   (audit_df.key == "patient_id")].to_string(index=False))

    # the check that protects the headline comparison
    spread = (size_df[size_df.partition == "test"]
              .groupby("seed").pct_of_pool.agg(lambda s: s.max() - s.min()))
    if (spread > 1.0).any():
        print("\n! test partition sizes differ by more than 1 percentage point "
              "between arms; reviewers will read the delta as a sample-size "
              "effect. Reduce `jitter` or rebalance before running.")


if __name__ == "__main__":
    main()
