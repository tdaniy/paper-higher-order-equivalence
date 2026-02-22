#!/usr/bin/env python3
"""Compare two output directories for byte-for-byte equality."""
from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            out[rel] = file_hash(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Determinism check: compare two directories")
    parser.add_argument("--dir-a", required=True, help="first directory")
    parser.add_argument("--dir-b", required=True, help="second directory")
    args = parser.parse_args()

    a = hash_tree(args.dir_a)
    b = hash_tree(args.dir_b)

    ok = True
    for rel in sorted(set(a) | set(b)):
        ha = a.get(rel)
        hb = b.get(rel)
        if ha != hb:
            ok = False
            print(f"DIFF {rel}: {ha} != {hb}")

    if ok:
        print("OK: directories match")
        raise SystemExit(0)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
