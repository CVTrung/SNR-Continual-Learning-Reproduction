"""Load and prepare CIFAR-10 images and labels.

Status: design scaffold; no executable implementation.
Inputs: Hugging Face/Kaggle reference, revision, cache path, and preparation settings.
Intended outputs: Prepared base data with source/preprocessing/split metadata.
Responsibilities: Verify channel order, preprocessing, label encoding, and splits. Random task labels belong in tasks/. Download programmatically; do not define tasks or models.
"""
