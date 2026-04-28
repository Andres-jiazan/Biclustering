"""
Dataset loading utilities for face recognition experiments.

Supports:
  - ORL (AT&T) Database of Faces
  - Extended Yale B (optional)

ORL dataset structure expected:
  data/ORL/
    s1/  (subject 1)
      1.pgm ... 10.pgm
    s2/
      ...
    s40/

Usage:
  from data.load_dataset import load_orl, train_test_split_orl
"""

import os
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
#  ORL / AT&T face dataset
# --------------------------------------------------------------------------- #

def load_orl(data_dir: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ORL (AT&T) face database.

    Returns
    -------
    X : ndarray, shape (400, 10304)
        Each row is a flattened 92×112 grayscale image, pixel values in [0, 1].
    y : ndarray, shape (400,)
        Class labels 0–39 (subject index).
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "ORL")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"ORL dataset directory not found: {data_dir}\n"
            "Download from: https://cam-orl.co.uk/facedatabase.html\n"
            "Expected structure: ORL/s1/1.pgm … ORL/s40/10.pgm"
        )

    images, labels = [], []
    subjects = sorted(
        [d for d in os.listdir(data_dir) if d.startswith("s")],
        key=lambda x: int(x[1:])
    )

    for label, subject in enumerate(subjects):
        subject_dir = os.path.join(data_dir, subject)
        pgm_files = sorted(
            [f for f in os.listdir(subject_dir) if f.endswith(".pgm")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        for pgm in pgm_files:
            img = Image.open(os.path.join(subject_dir, pgm)).convert("L")
            images.append(np.array(img, dtype=np.float32).ravel() / 255.0)
            labels.append(label)

    X = np.array(images)   # (400, 10304)
    y = np.array(labels)   # (400,)
    return X, y


def train_test_split_orl(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split ORL dataset per class.

    Parameters
    ----------
    n_train : int
        Number of images per subject used for training (default 5, max 10).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(random_state)
    n_per_class = 10  # ORL always has 10 images per subject

    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        perm = rng.permutation(len(idx))
        train_idx.extend(idx[perm[:n_train]].tolist())
        test_idx.extend(idx[perm[n_train:]].tolist())

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# --------------------------------------------------------------------------- #
#  Extended Yale B (optional)
# --------------------------------------------------------------------------- #

def load_yaleb(data_dir: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Extended Yale B face database (CroppedYale format).

    Expected structure:
      data/YaleB/
        yaleB01/  (subject 1)
          yaleB01_P00A+000E+00.pgm ...
        yaleB02/
          ...

    Returns
    -------
    X : ndarray, shape (N, H*W)
        Flattened grayscale images, pixel values in [0, 1].
    y : ndarray, shape (N,)
        Class labels (0-indexed subject index).
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "YaleB")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Extended Yale B dataset directory not found: {data_dir}"
        )

    images, labels = [], []
    subjects = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    img_size = None
    for label, subject in enumerate(subjects):
        subject_dir = os.path.join(data_dir, subject)
        pgm_files = [f for f in os.listdir(subject_dir) if f.endswith(".pgm")]
        for pgm in sorted(pgm_files):
            img = Image.open(os.path.join(subject_dir, pgm)).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            if img_size is None:
                img_size = arr.shape
            images.append(arr.ravel())
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)
    return X, y


# --------------------------------------------------------------------------- #
#  Quick dataset info helper
# --------------------------------------------------------------------------- #

def dataset_info(X: np.ndarray, y: np.ndarray, name: str = "Dataset") -> None:
    """Print basic statistics about a loaded dataset."""
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(f"[{name}]")
    print(f"  Samples   : {n_samples}")
    print(f"  Features  : {n_features}")
    print(f"  Classes   : {n_classes}")
    print(f"  Per class : {n_samples // n_classes}")
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")
