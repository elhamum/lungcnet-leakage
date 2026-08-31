

    manifest_nlst.csv          one row per image, from Drive
    participant_status.csv     download outcome per participant, from Drive

Both are small and are committed - together with `../cohort_nlst_200.csv` they
pin the exact cohort and the exact slices, so the dataset can be rebuilt without
redistributing any TCIA imaging.

`splits_nlst.csv` and its two report files are written here by `make_splits.py`.
