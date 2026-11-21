"""Download the core-scope datasets programmatically and verify them by SHA-256.

Run from the repo root:  python -m scripts.prepare_data            (both datasets)
                         python -m scripts.prepare_data --only mnist
Sources, each pinned to a revision:
- MNIST: Hugging Face dataset ylecun/mnist, revision 77f3279092a1c1579b2250db8eafed0ad422088c,
  files mnist/{train,test}-00000-of-00001.parquet -> src/data/mnist/
- Shakespeare: the complete-works text published with the paper's code
  (github.com/ajozefiak/SelfNormalizedResets, commit 8b84f0c, src/data/all_shakespeare.txt)
  -> src/data/shakespeare/all_shakespeare.txt, stored with LF line endings. No Hugging Face
  copy was found to match it byte for byte, so the pinned GitHub raw URL is the programmatic
  source. Only the text file is downloaded; no author code enters this repository.
An existing file with the right hash is kept, so the script never re-downloads it. A file with
a different hash is never overwritten: the script stops and reports it. The folders are
ignored by Git. Provenance is printed and written next to the data (PROVENANCE.json).
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path("src/data")
MNIST_REPO, MNIST_REV = "ylecun/mnist", "77f3279092a1c1579b2250db8eafed0ad422088c"
MNIST_FILES = {
    "train-00000-of-00001.parquet": "f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf",
    "test-00000-of-00001.parquet": "d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b",
}
SHAKESPEARE_URL = ("https://raw.githubusercontent.com/ajozefiak/SelfNormalizedResets/"
                   "8b84f0c295cf23269fdbfe20e5d03186638c318f/src/data/all_shakespeare.txt")
SHAKESPEARE_SHA = "4e81a0f25becafebd498f8833d7b8841de36dbe90b8034b2ca9f1a47d9c69a23"  # LF endings


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_existing(path, expected):
    """True if `path` exists with the expected hash; raise if it exists with another hash."""
    if not path.exists():
        return False
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{path} exists with sha256 {actual}, expected {expected}; "
                         "not overwriting it. Move it away and re-run.")
    print(f"ok (cached)  {path}")
    return True


def prepare_mnist():
    from huggingface_hub import hf_hub_download

    out = ROOT / "mnist"
    out.mkdir(parents=True, exist_ok=True)
    for name, digest in MNIST_FILES.items():
        target = out / name
        if check_existing(target, digest):
            continue
        cached = hf_hub_download(MNIST_REPO, f"mnist/{name}", repo_type="dataset",
                                 revision=MNIST_REV)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(Path(cached).read_bytes())
        if sha256(tmp) != digest:
            tmp.unlink()
            raise SystemExit(f"hash mismatch after download: {target}")
        tmp.replace(target)
        print(f"downloaded   {target}")
    meta = {"source": f"hf://datasets/{MNIST_REPO}", "revision": MNIST_REV, "sha256": MNIST_FILES}
    (out / "PROVENANCE.json").write_text(json.dumps(meta, indent=1))


def prepare_shakespeare():
    out = ROOT / "shakespeare"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "all_shakespeare.txt"
    if not check_existing(target, SHAKESPEARE_SHA):
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as response:
            data = response.read().replace(b"\r\n", b"\n")
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        if sha256(tmp) != SHAKESPEARE_SHA:
            tmp.unlink()
            raise SystemExit("hash mismatch for the Shakespeare corpus")
        tmp.replace(target)
        print(f"downloaded   {target}")
    meta = {"source": SHAKESPEARE_URL, "sha256_lf": SHAKESPEARE_SHA,
            "note": "The Tempest is extracted at load time by src/data/shakespeare.py"}
    (out / "PROVENANCE.json").write_text(json.dumps(meta, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["mnist", "shakespeare"], default=None)
    args = p.parse_args()
    if args.only in (None, "mnist"):
        prepare_mnist()
    if args.only in (None, "shakespeare"):
        prepare_shakespeare()


if __name__ == "__main__":
    main()
