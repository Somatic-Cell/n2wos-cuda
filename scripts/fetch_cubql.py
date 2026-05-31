#!/usr/bin/env python3
"""Fetch NVIDIA/cuBQL into external/cuBQL.

This script keeps third-party code out of n2wos-cuda patches while making the
cuBQL backend reproducible enough for local experiments. It performs a normal
Git clone; no files from cuBQL are vendored by this repository.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

DEFAULT_REPO = "https://github.com/NVIDIA/cuBQL.git"
DEFAULT_REF = "main"


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="external/cuBQL", help="destination checkout directory")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="git repository URL")
    parser.add_argument("--ref", default=DEFAULT_REF, help="branch, tag, or commit to checkout")
    parser.add_argument("--force", action="store_true", help="remove an existing destination before cloning")
    args = parser.parse_args()

    dest = pathlib.Path(args.dest)
    if dest.exists():
        if not args.force:
            print(f"destination already exists: {dest}", file=sys.stderr)
            print("use --force to replace it", file=sys.stderr)
            return 2
        run(["rm", "-rf", str(dest)])

    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "--branch", args.ref, args.repo, str(dest)])

    try:
        run(["git", "rev-parse", "HEAD"], cwd=dest)
    except subprocess.CalledProcessError:
        return 1
    print("cuBQL checkout ready at", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
