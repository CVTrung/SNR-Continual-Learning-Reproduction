# Dataset sources and local inventory

Purpose: record where data is stored, which source is proposed, and what remains unverified. Inputs: a read-only file inventory and the source links below. Intended output: a versioned provenance record after validation. Status: file presence/size only was inspected; datasets were not loaded, downloaded, transformed, moved, or overwritten.

## Storage boundary

Raw files are in data/ at the repository root (or a future explicitly configured external cache). src/data/ contains loader modules only. data/ is ignored by Git. File presence does not establish a successful loader, correct schema, sample count, source provenance, or equivalence to the paper.

## Proposed sources and observations

| Dataset | Proposed/reference source | Observed local location and apparent splits | Revision/version and sample count | Preprocessing status |
| --- | --- | --- | --- | --- |
| MNIST | [HF ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist) | data/mnist/; train/test Parquet files | Actual download origin, revision, and row counts not verified | Normalization, flattening, subsets, and evaluation splits not verified |
| CIFAR-10 | [HF uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) | data/cifar10/; train/test Parquet files | Actual download origin, revision, and row counts not verified | Channel order, normalization, subsets, and labels not verified |
| ImageNet-32 | [Candidate HF mirror](https://huggingface.co/datasets/ChocolateDave/imagenet-32); [original project](https://patrykchrabaszcz.github.io/Imagenet32/) | data/imagenet32/; seven train shards and one val shard | Mirror equivalence, source revision, row counts, class mapping, and per-class counts not verified | Downsampling provenance, preprocessing, and class-pair selection not verified |
| Shakespeare | [Authors' reference file](https://github.com/ajozefiak/SelfNormalizedResets/blob/main/src/data/all_shakespeare.txt) | data/shakespeare/all_shakespeare.txt; no split inferred | Actual origin/commit, text identity, and token counts not verified | Corpus normalization, Tempest selection, tokenizer/vocabulary, and context construction not verified |

These links identify proposed sources, not proof that the local files came from them. A Hugging Face mirror is not automatically equivalent to the paper dataset. No source-to-local checksum comparison has been performed.

## Course-compatible acquisition

The final pipeline must load programmatically from Hugging Face or Kaggle and reconstruct required data structures automatically. Existing local files can serve as a cache once provenance is established, but manual placement alone does not fulfill the acquisition requirement. A matching hosted Shakespeare source has not been verified. Decide how to satisfy that requirement after checking the reference corpus and permitted hosting; no dataset upload is authorized by this document.

## Verification record to complete later

For each source record URL/dataset identifier, immutable revision or reference commit, access conditions, download procedure, local/cache path, schema, splits, measured row counts, checksums where appropriate, preprocessing, and paper-equivalence checks. Record any differences explicitly. Check the Shakespeare corpus against both base and scaled language protocols; do not substitute an arbitrary text sample.

Task permutations, random labels, class pairing, and vocabulary remapping belong in src/tasks/. Exact experiment configurations and result conventions remain deferred.

## Observed files

Sizes below are filesystem observations only, not integrity or content validation.

| Local file | Bytes |
| --- | --- |
| data/cifar10/test-00000-of-00001.parquet | 23,940,850 |
| data/cifar10/train-00000-of-00001.parquet | 119,705,255 |
| data/imagenet32/train-00000-of-00007.parquet | 431,585,783 |
| data/imagenet32/train-00001-of-00007.parquet | 431,659,830 |
| data/imagenet32/train-00002-of-00007.parquet | 431,511,929 |
| data/imagenet32/train-00003-of-00007.parquet | 431,422,502 |
| data/imagenet32/train-00004-of-00007.parquet | 431,337,862 |
| data/imagenet32/train-00005-of-00007.parquet | 431,671,616 |
| data/imagenet32/train-00006-of-00007.parquet | 431,338,158 |
| data/imagenet32/val-00000-of-00001.parquet | 118,132,481 |
| data/mnist/test-00000-of-00001.parquet | 2,595,890 |
| data/mnist/train-00000-of-00001.parquet | 15,561,616 |
| data/shakespeare/all_shakespeare.txt | 5,342,509 |
