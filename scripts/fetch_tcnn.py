#!/usr/bin/env python3
"""Fetch NVlabs/tiny-cuda-nn with recursive submodules.

The repository is intentionally not vendored by n2wos-cuda patches. tiny-cuda-nn
requires several submodules, so this script always initializes them.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="external/tiny-cuda-nn", help="destination checkout path")
    parser.add_argument("--repo", default="https://github.com/NVlabs/tiny-cuda-nn.git", help="repository URL")
    parser.add_argument("--ref", default="master", help="branch, tag, or commit to checkout")
    parser.add_argument("--update", action="store_true", help="update an existing checkout before initializing submodules")
    args = parser.parse_args()

    dest = pathlib.Path(args.dest)
    if dest.exists():
        if not (dest / ".git").exists():
            raise SystemExit(f"destination exists but is not a git checkout: {dest}")
        if args.update:
            run(["git", "fetch", "--all", "--tags"], cwd=dest)
        run(["git", "checkout", args.ref], cwd=dest)
        if args.update:
            run(["git", "pull", "--ff-only"], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--recursive", "--branch", args.ref, args.repo, str(dest)])

    run(["git", "submodule", "update", "--init", "--recursive"], cwd=dest)
    print(f"tiny-cuda-nn checkout ready at {dest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
