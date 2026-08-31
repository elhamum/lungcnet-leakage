"""
data.py -- tf.data pipeline over the frozen splits.

Ported from paper 1's notebook with two changes only: PNG instead of JPEG, and
partitions come from a `split_<arm>_seed<n>` column rather than three separate
CSVs. The augmentation, resize and batching are byte-for-byte the same, so any
difference between the two papers cannot come from the input pipeline.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import tensorflow as tf

CLASSES = np.array(["benign", "normal", "suspicious"])


def load_split_manifest(cfg: dict) -> pd.DataFrame:
    """Read the frozen splits and resolve every image path, failing loudly."""
    df = pd.read_csv(cfg["splits"])
    root = cfg["images_root"]
    df["abspath"] = [os.path.join(root, p) for p in df["path"]]

    missing = [p for p in df["abspath"].head(200) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the first 200 images not found under {root!r}; "
            f"e.g. {missing[0]}. Check images_root in the config."
        )
    unknown = set(df["label"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unexpected labels in manifest: {sorted(unknown)}")
    return df


def _proc(path, aug: bool, size: int):
    im = tf.io.read_file(path)
    im = tf.image.decode_png(im, channels=3)
    if aug:
        im = tf.image.random_flip_left_right(im)
    im = tf.image.convert_image_dtype(im, tf.float32)
    return tf.image.resize(im, [size, size])


def batches(df: pd.DataFrame, aug: bool, shuf: bool, seed: int,
            size: int = 256, batch: int = 32) -> tf.data.Dataset:
    lab = np.array([l == CLASSES for l in df["label"]], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((df["abspath"].to_numpy(), lab))
    if shuf:
        ds = ds.shuffle(len(df), seed=seed)
    ds = ds.map(lambda p, l: (_proc(p, aug, size), l),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def partitions(manifest: pd.DataFrame, split_col: str):
    """Split the manifest into (train, val, test) frames."""
    g = {k: v for k, v in manifest.groupby(split_col)}
    for name in ("train", "val", "test"):
        if name not in g:
            raise KeyError(f"{split_col} has no {name!r} partition")
    return g["train"], g["val"], g["test"]


def y_of(df: pd.DataFrame) -> np.ndarray:
    return np.array([np.where(CLASSES == l)[0][0] for l in df["label"]])
