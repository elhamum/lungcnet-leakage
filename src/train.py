"""
train.py -- one cell of the factorial: (architecture, arm, seed) -> metrics.

Evaluation is paper 1's: macro F1 and macro one-vs-rest ROC-AUC on a held-out
test partition used once, after training, with no role in model selection.

The collapse check exists because paper 1 saw MobileNetV2 pin its validation
accuracy at the majority-class share while validation loss diverged. That is a
degenerate run, not a low score, and averaging it into a mean without flagging it
would misrepresent the architecture.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize

from data import CLASSES, batches, partitions, y_of
from models import build, make_callbacks


def train_one(cfg: dict, manifest: pd.DataFrame, split_col: str,
              arch: str, seed: int) -> dict:
    tf.keras.utils.set_random_seed(seed)      # weight init + shuffle + augmentation

    img   = int(cfg["input"]["size"][0])
    batch = int(cfg["input"]["batch_size"])
    tr, va, te = partitions(manifest, split_col)

    ds_tr = batches(tr, aug=True,  shuf=True,  seed=seed, size=img, batch=batch)
    ds_va = batches(va, aug=False, shuf=False, seed=seed, size=img, batch=batch)
    ds_te = batches(te, aug=False, shuf=False, seed=seed, size=img, batch=batch)

    model = build(arch, img)
    t0 = time.time()
    hist = model.fit(ds_tr, validation_data=ds_va,
                     epochs=int(cfg["training"]["max_epochs"]),
                     callbacks=make_callbacks(arch), verbose=2)
    secs = time.time() - t0

    probs = model.predict(ds_te, verbose=0)
    pred  = np.argmax(probs, axis=1)
    y_te  = y_of(te)

    rep = classification_report(y_te, pred, target_names=list(CLASSES),
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=range(len(CLASSES)))
    try:
        auc = roc_auc_score(label_binarize(y_te, classes=list(range(len(CLASSES)))),
                            probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")

    h = pd.DataFrame(hist.history)
    best = int(h["val_loss"].idxmin())

    # degenerate-run detection
    n_pred_classes = int(len(np.unique(pred)))
    majority_share = float(pd.Series(y_of(va)).value_counts(normalize=True).max())
    collapsed = bool(
        n_pred_classes < int(cfg["collapse_check"]["min_predicted_classes"])
        or (cfg["collapse_check"]["flag_if_val_acc_equals_majority_share"]
            and abs(float(h.loc[best, "val_accuracy"]) - majority_share) < 1e-3)
    )

    return dict(
        n_train=len(tr), n_val=len(va), n_test=len(te),
        test_accuracy=round(float(rep["accuracy"]) * 100, 2),
        test_macro_f1=round(float(rep["macro avg"]["f1-score"]) * 100, 2),
        test_macro_auc=None if np.isnan(auc) else round(float(auc), 4),
        test_loss=None,
        val_accuracy=round(float(h.loc[best, "val_accuracy"]) * 100, 2),
        val_loss=round(float(h.loc[best, "val_loss"]), 4),
        per_class_f1_json=json.dumps(
            {c: round(rep[c]["f1-score"] * 100, 2) for c in CLASSES}),
        confusion_json=json.dumps(cm.tolist()),
        epochs_run=len(h), best_epoch=best + 1,
        collapsed_to_single_class=collapsed,
        train_seconds=round(secs, 1),
        note="collapsed onto a single class" if collapsed else "",
    )
