#!/usr/bin/env python3
"""Collect n2wos-cuda experiment outputs into a shareable tarball.

The script uses only the Python standard library. Dense CSV files are excluded by
default because field outputs can be large; pass --include-dense-csv to include
*.csv files as well.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable


DEFAULT_INCLUDE_SUFFIXES = {
    ".json",
    ".md",
    ".txt",
    ".log",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
}


def run_git(args: list[str], repo: Path) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as exc:  # pragma: no cover - diagnostic helper
        return f"<unavailable: {exc}>"


def iter_result_files(results_dir: Path, include_dense_csv: bool) -> Iterable[Path]:
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "share_latest.tar.gz":
            continue
        if path.suffix.lower() == ".csv" and include_dense_csv:
            yield path
        elif path.suffix.lower() in DEFAULT_INCLUDE_SUFFIXES:
            yield path


def write_manifest(repo: Path, results_dir: Path, bundle_files: list[Path], args: argparse.Namespace) -> Path:
    manifest = {
        "schema": "n2wos_cuda_experiment_bundle_v1",
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "repo": str(repo),
        "results_dir": str(results_dir),
        "include_dense_csv": bool(args.include_dense_csv),
        "git": {
            "commit": run_git(["rev-parse", "HEAD"], repo),
            "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
            "status_short": run_git(["status", "--short"], repo),
            "diff_stat": run_git(["diff", "--stat"], repo),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "files": [str(p.relative_to(results_dir)) for p in bundle_files],
    }
    manifest_path = results_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--include-dense-csv", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd().resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    output = args.output.resolve() if args.output else (results_dir / "share_latest.tar.gz")
    files = list(iter_result_files(results_dir, args.include_dense_csv))
    manifest = write_manifest(repo, results_dir, files, args)
    if manifest not in files:
        files.append(manifest)

    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            arcname = Path("results") / path.relative_to(results_dir)
            tar.add(path, arcname=str(arcname))

    print(f"wrote {output}")
    print(f"files: {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
