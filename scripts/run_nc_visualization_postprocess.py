#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Run NC visualization postprocess for a results tree.")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--reference-estimates-csv", default=None)
    ap.add_argument("--require-reference", action="store_true")
    ap.add_argument("--print-summary", action="store_true")
    args = ap.parse_args()

    gen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_nc_visualizations.py")
    cmd = [sys.executable, gen, "--results-root", args.results_root]
    if args.reference_estimates_csv:
        cmd += ["--reference-estimates-csv", args.reference_estimates_csv]
    if args.require_reference:
        cmd += ["--require-reference"]
    if args.print_summary:
        cmd += ["--print-summary"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
