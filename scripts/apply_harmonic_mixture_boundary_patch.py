#!/usr/bin/env python3
"""Patch the local n2wos-cuda tree with harmonic-mixture boundary modes.

This script is intentionally source-aware rather than a fixed textual diff,
because the NC/WoS evaluator has been changing quickly in this branch.  It
performs conservative edits and writes .bak files before modifying sources.

It adds two analytic harmonic boundary modes:

  - HarmonicMixtureSmooth      -> harmonic_mixture_smooth
  - HarmonicMixtureFiglike     -> harmonic_mixture_figlike

Usage from repository root:

  python scripts/apply_harmonic_mixture_boundary_patch.py --check
  python scripts/apply_harmonic_mixture_boundary_patch.py --apply

If the script cannot find the boundary-value switch, run:

  grep -R "nc_boundary_value_host\|BoundaryTextureCheckerK16\|ConstantOne" -n include src

and patch the indicated file manually using include/n2wos/nc_harmonic_mixture.hpp.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path.cwd()
HEADER = ROOT / "include" / "n2wos" / "tcnn_nc_wos.hpp"

INCLUDE_LINE = '#include "n2wos/nc_harmonic_mixture.hpp"\n'

ENUM_BLOCK = """\n  HarmonicMixtureSmooth = 13,\n  HarmonicMixtureFiglike = 14,"""

CLI_IFS_QUALIFIED = """
  if (s == "harmonic_mixture_smooth" || s == "mixture_smooth" ||
      s == "hm_smooth" || s == "fig_smooth") {
    return n2wos::NcBoundaryMode::HarmonicMixtureSmooth;
  }
  if (s == "harmonic_mixture_figlike" || s == "mixture_figlike" ||
      s == "hm_figlike" || s == "figlike" || s == "fig1" || s == "fig4") {
    return n2wos::NcBoundaryMode::HarmonicMixtureFiglike;
  }
"""

PARSE_IFS_UNQUALIFIED = """
  if (s == "harmonic_mixture_smooth" || s == "mixture_smooth" ||
      s == "hm_smooth" || s == "fig_smooth") {
    return NcBoundaryMode::HarmonicMixtureSmooth;
  }
  if (s == "harmonic_mixture_figlike" || s == "mixture_figlike" ||
      s == "hm_figlike" || s == "figlike" || s == "fig1" || s == "fig4") {
    return NcBoundaryMode::HarmonicMixtureFiglike;
  }
"""

NAME_CASES_UNQUALIFIED = """
    case NcBoundaryMode::HarmonicMixtureSmooth: return "harmonic_mixture_smooth";
    case NcBoundaryMode::HarmonicMixtureFiglike: return "harmonic_mixture_figlike";
"""

NAME_CASES_QUALIFIED = """
    case n2wos::NcBoundaryMode::HarmonicMixtureSmooth: return "harmonic_mixture_smooth";
    case n2wos::NcBoundaryMode::HarmonicMixtureFiglike: return "harmonic_mixture_figlike";
"""

VALUE_CASES_UNQUALIFIED = """
    case NcBoundaryMode::HarmonicMixtureSmooth: return n2wos::nc_harmonic_mixture_smooth_value(p);
    case NcBoundaryMode::HarmonicMixtureFiglike: return n2wos::nc_harmonic_mixture_figlike_value(p);
"""

VALUE_CASES_QUALIFIED = """
    case n2wos::NcBoundaryMode::HarmonicMixtureSmooth: return n2wos::nc_harmonic_mixture_smooth_value(p);
    case n2wos::NcBoundaryMode::HarmonicMixtureFiglike: return n2wos::nc_harmonic_mixture_figlike_value(p);
"""


@dataclass
class Edit:
    path: Path
    old: str
    new: str
    notes: list[str]

    @property
    def changed(self) -> bool:
        return self.old != self.new


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def source_files() -> list[Path]:
    out: list[Path] = []
    for root_name in ("include", "src"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for suffix in ("*.hpp", "*.cuh", "*.cpp", "*.cu"):
            out.extend(root.rglob(suffix))
    return sorted(set(out))


def add_include(text: str) -> tuple[str, bool]:
    if "nc_harmonic_mixture.hpp" in text:
        return text, False
    anchors = [
        '#include "n2wos/tcnn_nc_wos.hpp"\n',
        '#include "n2wos/wos_wavefront.hpp"\n',
    ]
    for anchor in anchors:
        if anchor in text:
            return text.replace(anchor, anchor + INCLUDE_LINE, 1), True
    return text, False


def patch_enum(path: Path) -> Edit:
    old = read(path)
    text = old
    notes: list[str] = []
    if "HarmonicMixtureSmooth" not in text:
        # Preserve existing enum numeric values.  Append after ConstantOne=12.
        patterns = [
            (r"(\bConstantOne\s*=\s*12\s*,)", r"\1" + ENUM_BLOCK),
            (r"(\bConstantOne\s*,)", r"\1" + ENUM_BLOCK),
        ]
        for pat, repl in patterns:
            text2, n = re.subn(pat, repl, text, count=1)
            if n:
                text = text2
                notes.append("added NcBoundaryMode harmonic mixture enum values")
                break
        else:
            notes.append("WARNING: could not find ConstantOne enum anchor")
    return Edit(path, old, text, notes)


def patch_eval_cli(path: Path, text: str, notes: list[str]) -> str:
    if "harmonic_mixture_smooth" in text and "HarmonicMixtureSmooth" in text:
        return text
    anchors = [
        "return n2wos::parse_nc_boundary_mode(s.c_str());",
        "return parse_nc_boundary_mode(s.c_str());",
        'if (s == "shell8"',
        'if (s == "external_charges_shell8"',
    ]
    for anchor in anchors:
        idx = text.find(anchor)
        if idx >= 0:
            notes.append("added evaluator CLI aliases")
            return text[:idx] + CLI_IFS_QUALIFIED + text[idx:]
    return text


def patch_parse_function(text: str, notes: list[str]) -> str:
    if "mixture_figlike" in text and "HarmonicMixtureFiglike" in text:
        return text
    anchors = [
        'if (s == "constant_one"',
        'if (s == "constant-one"',
        'if (s == "harmonic_x2_minus_y2"',
    ]
    for anchor in anchors:
        idx = text.find(anchor)
        if idx >= 0:
            notes.append("added parse_nc_boundary_mode aliases")
            return text[:idx] + PARSE_IFS_UNQUALIFIED + text[idx:]
    return text


def patch_name_cases(text: str, notes: list[str]) -> str:
    if 'HarmonicMixtureSmooth: return "harmonic_mixture_smooth"' in text:
        return text
    patterns = [
        (r'(case\s+NcBoundaryMode::ConstantOne\s*:\s*return\s+"constant_one"\s*;)', NAME_CASES_UNQUALIFIED),
        (r'(case\s+n2wos::NcBoundaryMode::ConstantOne\s*:\s*return\s+"constant_one"\s*;)', NAME_CASES_QUALIFIED),
    ]
    for pat, insert in patterns:
        text2, n = re.subn(pat, insert + r"\1", text, count=1)
        if n:
            notes.append("added nc_boundary_mode_name cases")
            return text2
    return text


def patch_value_cases(text: str, notes: list[str]) -> str:
    if "nc_harmonic_mixture_smooth_value" in text:
        return text
    patterns = [
        (r'(case\s+NcBoundaryMode::ConstantOne\s*:\s*return\s+1(?:\.0)?f?\s*;)', VALUE_CASES_UNQUALIFIED),
        (r'(case\s+n2wos::NcBoundaryMode::ConstantOne\s*:\s*return\s+1(?:\.0)?f?\s*;)', VALUE_CASES_QUALIFIED),
    ]
    for pat, insert in patterns:
        text2, n = re.subn(pat, insert + r"\1", text, count=1)
        if n:
            notes.append("added boundary-value cases")
            return text2
    return text


def patch_general_file(path: Path) -> Edit:
    old = read(path)
    text = old
    notes: list[str] = []

    # Only files that already mention NC boundary modes are candidates.
    if "NcBoundaryMode" not in text and "parse_nc_boundary_mode" not in text:
        return Edit(path, old, text, notes)

    text, inc = add_include(text)
    if inc:
        notes.append("added nc_harmonic_mixture include")

    text = patch_eval_cli(path, text, notes) if path.name == "eval_tcnn_nc_wos.cu" else text
    text = patch_parse_function(text, notes)
    text = patch_name_cases(text, notes)
    text = patch_value_cases(text, notes)
    return Edit(path, old, text, notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print planned edits without modifying files")
    ap.add_argument("--apply", action="store_true", help="apply edits in-place, writing .bak files first")
    args = ap.parse_args()
    if args.check == args.apply:
        ap.error("pass exactly one of --check or --apply")

    edits: list[Edit] = []
    if HEADER.exists():
        edits.append(patch_enum(HEADER))
    else:
        print(f"WARNING: missing header {HEADER}")

    for path in source_files():
        if path == HEADER:
            continue
        edit = patch_general_file(path)
        if edit.changed:
            edits.append(edit)

    changed = [e for e in edits if e.changed]
    if not changed:
        print("No changes needed; harmonic mixture support may already be applied.")
        return 0

    print("Planned edits:" if args.check else "Applying edits:")
    for edit in changed:
        print(f"  {edit.path.relative_to(ROOT)}")
        for note in edit.notes:
            print(f"    - {note}")

    if args.apply:
        for edit in changed:
            bak = edit.path.with_suffix(edit.path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(edit.path, bak)
            edit.path.write_text(edit.new, encoding="utf-8")

    print("\nAfter applying, build with:")
    print("  cmake --build ./build/cuda-release-cubql-tcnn -j")
    print("\nIf the build fails because the boundary-value switch uses a point variable")
    print("other than `p`, inspect the reported file and replace `p` in")
    print("nc_harmonic_mixture_*_value(p) with the local point variable name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
