# Data leakage in CT-based lung nodule classification

Code and cohort definitions for a benchmarking study measuring how much
slice-level data leakage inflates reported performance in CT lung cancer
classification.

The experiment holds architecture, data and hyperparameters fixed and varies
only the unit kept intact when the data is partitioned. The result is the
**difference between split strategies**, not the absolute score of any model.

Follow-up to *LungCNET: A High-Performance Deep CNN Model for Lung Cancer
Detection Evaluated Against Widely Used CNN Benchmarks*, Bioengineering (MDPI),
2026 — where a patient-level split could not be performed at all, because the
IQ-OTH/NCCD public release provides no slice-to-patient mapping.

## Split arms

| arm | unit kept intact | leaks? |
|---|---|---|
| slice | none — images split individually | yes, heavily |
| exam | one screening visit | yes, at patient level |
| patient | all images of a participant | no |

The exam arm is the point of the study. It looks patient-aware, but in a
longitudinal screening trial the same participant returns for annual scans, so
their T0 exam can sit in training while their T2 exam sits in test.

## Data

**NLST** (National Lung Screening Trial), via the TCIA public release.
No images are redistributed here. The cohort is defined by
`cohort_nlst_200.csv` and rebuilt from the public sources by the code below.

200 participants, 581 exams, 5,810 images. Stratified by the participant's most
severe screening result. Every participant has two or more screening exams.

Labels come from `scr_res` in the NLST clinical package: code 1 → normal,
codes 2–3 → benign, codes 4–6 → suspicious. These are the radiologist's
screening interpretation, not a pathological diagnosis.

## Reproducing the dataset

1. Download the NLST clinical package from the
   [TCIA NLST collection page](https://www.cancerimagingarchive.net/collection/nlst/)
   — "Clinical data including data dictionaries", `package-nlst-780.2021-05-28.zip`,
   2.18 MB, no application required. Unzip twice; you want the CSVs.

2. Rebuild the participant list (reproduces `cohort_nlst_200.csv` exactly):

   ```
   python src/build_cohort_nlst.py --n 200 --oversample 40 \
       --prsn nlst_780_prsn_idc_20210527.csv \
       --screen nlst_780_screen_idc_20210527.csv \
       --out cohort_nlst_200.csv
   ```

3. Download and convert the imaging with `notebooks/NLST_build_cohort_Colab.ipynb`.
   It pulls one CT series per exam from the TCIA API, keeps ten slices, writes
   256×256 PNGs, and deletes the DICOM as it goes. Roughly 48 GB passes through;
   about 194 MB is kept. Resumable.

   Preprocessing, fixed for the whole study: lung window (level −600, width 1500),
   ten slices evenly spaced across the middle 70% of the series, 256×256, matching
   paper 1.

   Series selection: the smallest `OPA` reconstruction with at least 100 slices.
   `OPL` localiser scouts are discarded.

## Running the experiment

```
pip install -r requirements.txt

python src/make_splits.py --config configs/nlst.yaml     # freeze all arm x seed splits
PYTHONPATH=src python src/runner.py --config configs/nlst.yaml \
       --results results/results_nlst.csv                # resumable factorial queue
```

`make_splits.py` writes one `split_<arm>_seed<n>` column per grid cell, plus the
partition-size table and the leakage audit. Check the size table before training:
if test partitions differ between arms by more than a percentage point, the delta
is contaminated by sample size.

Set `images_root` in the config to wherever the PNGs live.

The default grid is four architectures (LungCNET, VGG16, InceptionV3,
MobileNetV2) x 3 arms x 3 seeds = 36 runs. ResNet50 and YOLOv11 from paper 1 are
omitted to keep the queue feasible on free Colab; the four retained cover both
stable and unstable convergence behaviour, which the seed-variance analysis needs.

`runner.py` rebuilds its queue from `results.csv` on every start, so an
interrupted session costs one run. Arms sweep innermost, so a partial queue still
yields complete arm comparisons.

## Layout

```
configs/nlst.yaml               experiment grid + paper 1 hyperparameters
cohort_nlst_200.csv             the cohort definition
data/manifest_nlst.csv          one row per image (produced by the notebook)
data/participant_status.csv     substitutions made during the download
src/build_cohort_nlst.py        participant selection
src/build_manifest.py           DICOM tree -> manifest, with identifier checks
src/splits.py                   group-aware splitter + leakage audit
src/check_splits.py             splitter self-test on synthetic cohorts, no data needed
src/make_splits.py              freezes all arm x seed splits
src/check_slice_mapping.py      validates NLST sct_slice_num against the images
src/data.py                     tf.data pipeline over the frozen splits
src/models.py                   the architectures, carried over from paper 1
src/train.py                    one run -> metrics
src/runner.py                   resumable experiment driver
notebooks/                      Colab notebooks
```

## Notes on NLST that cost us time

Recorded here because none of it is documented and all of it was established
against the data.

- **Separate API server.** The full trial is at
  `nlst.cancerimagingarchive.net/nbia-api/services/v1`. The `NLST` collection on
  the main TCIA server is a 501-patient annotation subset, not the trial.
- **Timepoints.** `ClinicalTrialTimePointID` gives T0/T1/T2 directly. The
  StudyDate year (1999/2000/2001) and the leading field of `SeriesDescription`
  independently agree. Dates are fictitious; order them, never subtract them.
- **Slice ordering.** Sort by `ImagePositionPatient` z. `InstanceNumber` runs in
  opposite directions on different scanners and must not be used.
- **`sct_slice_num` counts superior to inferior**, indexing the largest
  reconstruction of that exam. Verified on 801 nodules: mean fractional position
  from the top of the scan is 0.367 for upper-lobe nodules, 0.550 for middle and
  0.595 for lower. The inverse rule reverses that ordering.
- **Slice coordinates exist only for abnormality code 51** (non-calcified nodule
  >= 4 mm). All fourteen other abnormality codes have none, which is why labels
  here are exam-level rather than slice-level.
- **`screen` is not the radiologist's read.** It holds acquisition parameters and
  a diagnostic-quality flag. The screening result is in `prsn`, as `scr_res0/1/2`.

## Citation

Eskandarnia, E.; Adepoju, P.; Kiani, K.; Mansouri, T.; BinRajab, A.
*LungCNET: A High-Performance Deep CNN Model for Lung Cancer Detection Evaluated
Against Widely Used CNN Benchmarks.* Bioengineering (MDPI), 2026.
