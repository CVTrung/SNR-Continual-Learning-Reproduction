"""Load and prepare MNIST images and labels.

Status: design scaffold; no executable implementation.
Inputs: Hugging Face/Kaggle reference, revision, cache path, and preparation settings.
Intended outputs: Prepared base data with source/preprocessing/split metadata.
Responsibilities: Verify hosted revision, normalization, subsets, and train/test splits. Preserve original labels. Download programmatically; do not define tasks or models.
"""
