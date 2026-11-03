"""Load and prepare ImageNet-32 images and class identifiers.

Status: design scaffold; no executable implementation.
Inputs: Hugging Face/Kaggle reference, revision, cache path, and preparation settings.
Intended outputs: Prepared base data with source/preprocessing/split metadata.
Responsibilities: Verify the downsampled variant, access conditions, counts, and class mapping. Do not silently replace the dataset. Download programmatically; do not define tasks or models.
"""
