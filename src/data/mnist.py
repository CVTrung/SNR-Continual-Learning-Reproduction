"""Load MNIST into NumPy arrays from a local .npz or Parquet file.

Status: first executable version (loading only; no download, no tasks).
Inputs: path to an .npz file (keys x_train, y_train, x_test, y_test; images uint8 (N, 28, 28))
        or to a Parquet file with an `image` column (PNG bytes) and a `label` column.
Outputs: x float32 (N, 784) scaled to [0, 1]; y int64 (N,).
Responsibilities: only read and convert data. Do not create tasks or models here.
Decision to record in docs/reproduction_notes.md: pixels are divided by 255 and not
standardized further, because the paper does not state its preprocessing.
"""
import io

import numpy as np


def load_mnist(path, split="train"):
    """Read MNIST from `path`.

    Parameters
    ----------
    path : str or Path
        e.g. data/mnist/mnist.npz (preferred) or data/mnist/train-00000-of-00001.parquet.
    split : "train" or "test"
        Only used for .npz files (a Parquet file already holds a single split).

    Returns
    -------
    x : np.ndarray, float32, shape (N, 784), values in [0, 1]
    y : np.ndarray, int64, shape (N,)
    """
    path = str(path)
    if path.endswith(".npz"):
        with np.load(path) as f:
            images, labels = f[f"x_{split}"], f[f"y_{split}"]
    else:
        import pandas as pd  # imported here so the .npz route needs only NumPy
        from PIL import Image

        df = pd.read_parquet(path)
        images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"]))) for d in df["image"]])
        labels = df["label"].to_numpy()
    x = images.reshape(len(images), -1).astype(np.float32) / 255.0
    y = labels.astype(np.int64)
    return x, y