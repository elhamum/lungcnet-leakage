"""
models.py -- the six architectures from paper 1, unchanged.

Copied from the paper 1 notebooks rather than reimplemented. If these definitions
drift, the two papers stop being comparable and the leakage delta gets tangled up
with architectural differences. Do not tune them here.

LungCNET: custom, trained from scratch, ~59.21M parameters (the Flatten after
three conv blocks at 256x256 is where nearly all of them live).
"""

from __future__ import annotations

from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import (InceptionV3, MobileNetV2, ResNet50,
                                           VGG16)

N_CLASSES = 3

# unfreeze / lr / schedule exactly as published in paper 1
BENCH = {
    "vgg16":       dict(fn=VGG16,       unfreeze=4,  lr=1e-4, sched="exp"),
    "resnet50":    dict(fn=ResNet50,    unfreeze=10, lr=1e-4, sched="exp"),
    "inceptionv3": dict(fn=InceptionV3, unfreeze=20, lr=1e-3, sched="plateau"),
    "mobilenetv2": dict(fn=MobileNetV2, unfreeze=10, lr=1e-3, sched="plateau"),
}


def build_bench(name: str, img: int = 256):
    cfg = BENCH[name]
    base = cfg["fn"](weights="imagenet", include_top=False,
                     input_shape=(img, img, 3))
    base.trainable = True
    for l in base.layers[:-cfg["unfreeze"]]:
        l.trainable = False
    m = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
    lr = cfg["lr"]
    if cfg["sched"] == "exp":
        lr = optimizers.schedules.ExponentialDecay(cfg["lr"], 10000, 0.96,
                                                   staircase=True)
    m.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def build_lungcnet(img: int = 256):
    m = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img, img, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation="relu"), layers.Dropout(0.5),
        layers.Dense(256, activation="relu"), layers.Dropout(0.5),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
    m.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def build(arch: str, img: int = 256):
    arch = arch.lower()
    if arch == "lungcnet":
        return build_lungcnet(img)
    if arch in BENCH:
        return build_bench(arch, img)
    raise ValueError(f"unknown architecture {arch!r}; "
                     f"expected lungcnet or one of {sorted(BENCH)}")


def make_callbacks(arch: str):
    """Early stopping on val_loss with restore-best-weights, as in paper 1."""
    out = [callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                   restore_best_weights=True)]
    if BENCH.get(arch.lower(), {}).get("sched") == "plateau":
        out.append(callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                               patience=2, min_lr=1e-6))
    return out
