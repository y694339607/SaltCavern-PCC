#!/usr/bin/env python3
"""Verify SHA-256 fingerprints of released assets against the manifest.

Supplemental utility authored for the public repository release.
Usage:
    python code/supplemental/verify_assets.py --manifest docs/checkpoint_sha256.tsv --dir checkpoints/
"""
import argparse
import hashlib
import sys
from pathlib import Path

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="TSV with columns: file, role, size_bytes, sha256")
    ap.add_argument("--dir", required=True, help="directory holding the assets")
    args = ap.parse_args()

    ok, fail, missing = 0, 0, 0
    for line in Path(args.manifest).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        name, _role, size, digest = line.split("\t")
        p = Path(args.dir) / name
        if not p.exists():
            print(f"MISSING  {name}")
            missing += 1
            continue
        actual = sha256(p)
        actual_size = p.stat().st_size
        if actual == digest and actual_size == int(size):
            print(f"OK       {name}")
            ok += 1
        else:
            print(f"FAIL     {name}  sha256={actual}  size={actual_size}")
            fail += 1
    print(f"\n{ok} OK, {fail} fingerprint mismatch, {missing} missing")
    return 0 if (fail == 0 and missing == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
