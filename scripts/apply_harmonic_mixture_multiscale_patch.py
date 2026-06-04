#!/usr/bin/env python3
"""Add multiscale harmonic-mixture boundary modes to the local n2wos-cuda tree.

Run from repository root after 0007/0007a:

  python scripts/apply_harmonic_mixture_multiscale_patch.py --check
  python scripts/apply_harmonic_mixture_multiscale_patch.py --apply

This script is intentionally conservative and writes .bak files before modifying
source files.  It adds two additional analytic harmonic boundary modes:

  - harmonic_mixture_multiscale
  - harmonic_mixture_figlike_hf

Each term is of the form exp(k a·x) cos/sin(k b·x + phi), with a·b=0.
Therefore each term is harmonic in the solver-normalized coordinates.
"""
from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path.cwd()
HEADER = ROOT / "include" / "n2wos" / "nc_harmonic_mixture.hpp"
MODE_HEADER = ROOT / "include" / "n2wos" / "tcnn_nc_wos.hpp"
CUBQL = ROOT / "src" / "cuda" / "cubql_bvh.cu"
EVAL = ROOT / "src" / "tools" / "eval_tcnn_nc_wos.cu"

EXTRA_FUNCTIONS = r'''

template <typename P>
N2WOS_HD_INLINE float nc_harmonic_mixture_multiscale_value(P p) {
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;
    const float inv2 = 0.70710678118f;
    const float inv3 = 0.57735026919f;

    // Low / middle frequency content that remains visible in the interior.
    const float t0 = 0.20f * expf(1.35f * x) * cosf(1.35f * y + 0.20f);
    const float t1 = -0.17f * expf(-1.55f * y) * sinf(1.55f * z - 0.45f);
    const float t2 = 0.15f * expf(1.70f * z) * cosf(1.70f * x + 0.80f);

    // Diagonal middle-frequency components.  Each p/q pair is orthonormal.
    const float p3 = inv2 * (x + y);
    const float q3 = inv2 * (-x + y);
    const float t3 = 0.095f * expf(2.55f * p3) * cosf(2.55f * q3 + 0.15f);

    const float p4 = inv2 * (y + z);
    const float q4 = inv2 * (-y + z);
    const float t4 = -0.080f * expf(-2.85f * p4) * sinf(2.85f * q4 - 0.65f);

    const float p5 = inv2 * (x + z);
    const float q5 = inv2 * (-x + z);
    const float t5 = 0.070f * expf(3.15f * p5) * cosf(3.15f * q5 + 1.10f);

    // A 3D oblique pair: a=(1,1,1)/sqrt(3), b=(1,-1,0)/sqrt(2).
    const float p6 = inv3 * (x + y + z);
    const float q6 = inv2 * (x - y);
    const float t6 = 0.060f * expf(-3.35f * p6) * sinf(3.35f * q6 + 0.40f);

    // A small high-ish component.  The coefficient is deliberately small so it
    // creates visible interior texture without dominating the global field.
    const float p7 = 0.81649658093f * x - 0.40824829046f * y - 0.40824829046f * z;
    const float q7 = inv2 * (y - z);
    const float t7 = 0.035f * expf(4.00f * p7) * cosf(4.00f * q7 - 0.35f);

    const float lin = 0.035f * x - 0.025f * y + 0.020f * z;
    return 0.62f * (t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + lin);
}

template <typename P>
N2WOS_HD_INLINE float nc_harmonic_mixture_figlike_hf_value(P p) {
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;
    const float inv2 = 0.70710678118f;

    // More aggressive than multiscale.  This is intended for figure making and
    // stress testing; it may be harder for the neural cache than the smooth and
    // multiscale variants.
    const float t0 = 0.18f * expf(1.65f * x) * cosf(1.65f * y + 0.35f);
    const float t1 = -0.15f * expf(-1.75f * y) * cosf(1.75f * z - 0.20f);
    const float t2 = 0.13f * expf(1.85f * z) * sinf(1.85f * x + 0.70f);

    const float p3 = inv2 * (x + z);
    const float q3 = inv2 * (-x + z);
    const float t3 = 0.095f * expf(2.90f * p3) * sinf(2.90f * q3 + 0.55f);

    const float p4 = inv2 * (x + y);
    const float q4 = inv2 * (-x + y);
    const float t4 = -0.085f * expf(-3.25f * p4) * cosf(3.25f * q4 - 0.75f);

    const float p5 = inv2 * (y + z);
    const float q5 = inv2 * (-y + z);
    const float t5 = 0.070f * expf(3.55f * p5) * sinf(3.55f * q5 + 1.20f);

    // Two small high-frequency interior-visible terms.
    const float p6 = 0.86602540378f * x + 0.5f * z;
    const float q6 = -0.5f * x + 0.86602540378f * z;
    const float t6 = 0.040f * expf(4.20f * p6) * cosf(4.20f * q6 + 0.25f);

    const float p7 = 0.8f * y + 0.6f * z;
    const float q7 = -0.6f * y + 0.8f * z;
    const float t7 = -0.032f * expf(-4.50f * p7) * sinf(4.50f * q7 - 0.30f);

    const float lin = 0.025f * x + 0.020f * y - 0.015f * z;
    return 0.55f * (t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + lin);
}
'''

@dataclass
class Change:
    path: Path
    before: str
    after: str

    @property
    def changed(self) -> bool:
        return self.before != self.after


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    return path.read_text(encoding="utf-8")


def add_extra_functions(text: str) -> str:
    if "nc_harmonic_mixture_multiscale_value" in text:
        return text
    marker = "\n}  // namespace n2wos"
    if marker not in text:
        raise SystemExit("could not find namespace close in nc_harmonic_mixture.hpp")
    return text.replace(marker, EXTRA_FUNCTIONS + marker, 1)


def patch_enum(text: str) -> str:
    if "HarmonicMixtureMultiscale" in text and "HarmonicMixtureFiglikeHF" in text:
        return text
    # Prefer inserting after the existing Figlike enum entry from 0007.
    pat = r"(HarmonicMixtureFiglike\s*=\s*14\s*,?)"
    if re.search(pat, text):
        return re.sub(pat, r"\1\n  HarmonicMixtureMultiscale = 15,\n  HarmonicMixtureFiglikeHF = 16,", text, count=1)
    # Fallback: insert before closing of enum class NcBoundaryMode.
    pat2 = r"(enum\s+class\s+NcBoundaryMode\s*\{[^}]*)(\n\};)"
    m = re.search(pat2, text, re.S)
    if not m:
        raise SystemExit("could not find NcBoundaryMode enum")
    body = m.group(1).rstrip()
    if not body.endswith(","):
        body += ","
    ins = "\n  HarmonicMixtureMultiscale = 15,\n  HarmonicMixtureFiglikeHF = 16,"
    return text[:m.start()] + body + ins + m.group(2) + text[m.end():]


def patch_eval_cli(text: str) -> str:
    if "harmonic_mixture_multiscale" in text and "harmonic_mixture_figlike_hf" in text:
        return text
    target = "if (s == \"harmonic_mixture_figlike\""
    idx = text.find(target)
    if idx >= 0:
        # Insert after the figlike branch block.
        ret = "return n2wos::NcBoundaryMode::HarmonicMixtureFiglike;"
        ridx = text.find(ret, idx)
        if ridx < 0:
            raise SystemExit("found figlike CLI branch but not return")
        end = text.find("}\n", ridx)
        if end < 0:
            raise SystemExit("could not find end of figlike CLI branch")
        end += 2
        insert = '''
  if (s == "harmonic_mixture_multiscale" || s == "mixture_multiscale" ||
      s == "hmix_multiscale" || s == "multiscale") {
    return n2wos::NcBoundaryMode::HarmonicMixtureMultiscale;
  }
  if (s == "harmonic_mixture_figlike_hf" || s == "mixture_figlike_hf" ||
      s == "hmix_figlike_hf" || s == "figlike_hf" || s == "fig_hf") {
    return n2wos::NcBoundaryMode::HarmonicMixtureFiglikeHF;
  }
'''
        return text[:end] + insert + text[end:]
    # Fallback: insert before call to n2wos::parse_nc_boundary_mode.
    fallback = "return n2wos::parse_nc_boundary_mode(s.c_str());"
    if fallback not in text:
        raise SystemExit("could not find CLI parser insertion point")
    insert = '''
  if (s == "harmonic_mixture_multiscale" || s == "mixture_multiscale" ||
      s == "hmix_multiscale" || s == "multiscale") {
    return n2wos::NcBoundaryMode::HarmonicMixtureMultiscale;
  }
  if (s == "harmonic_mixture_figlike_hf" || s == "mixture_figlike_hf" ||
      s == "hmix_figlike_hf" || s == "figlike_hf" || s == "fig_hf") {
    return n2wos::NcBoundaryMode::HarmonicMixtureFiglikeHF;
  }
'''
    return text.replace(fallback, insert + "\n  " + fallback, 1)


def ensure_include(text: str) -> str:
    if 'nc_harmonic_mixture.hpp' in text:
        return text
    m = re.search(r"^#include .*$", text, re.M)
    if not m:
        return '#include "n2wos/nc_harmonic_mixture.hpp"\n' + text
    return text[:m.end()] + '\n#include "n2wos/nc_harmonic_mixture.hpp"' + text[m.end():]


def patch_cubql(text: str) -> str:
    text = ensure_include(text)
    # Device evaluator early returns.
    if "nc_harmonic_mixture_multiscale_value(p)" not in text:
        pat = r"(__host__\s+__device__\s+inline\s+float\s+nc_boundary_device\s*\([^)]*\)\s*\{)"
        m = re.search(pat, text)
        if not m:
            raise SystemExit("could not find nc_boundary_device")
        insert = '''
  if (mode == static_cast<int>(NcBoundaryMode::HarmonicMixtureMultiscale) || mode == 15) {
    return nc_harmonic_mixture_multiscale_value(p);
  }
  if (mode == static_cast<int>(NcBoundaryMode::HarmonicMixtureFiglikeHF) || mode == 16) {
    return nc_harmonic_mixture_figlike_hf_value(p);
  }
'''
        text = text[:m.end()] + insert + text[m.end():]
    # Enum to int mapping.
    if "NcBoundaryMode::HarmonicMixtureMultiscale" not in re.search(r"int\s+nc_boundary_to_device\s*\([^)]*\)\s*\{.*?\n\}", text, re.S).group(0):
        pat = r"(int\s+nc_boundary_to_device\s*\(\s*NcBoundaryMode\s+mode\s*\)\s*\{)"
        m = re.search(pat, text)
        if not m:
            raise SystemExit("could not find nc_boundary_to_device")
        insert = '''
  if (mode == NcBoundaryMode::HarmonicMixtureMultiscale) {
    return static_cast<int>(NcBoundaryMode::HarmonicMixtureMultiscale);
  }
  if (mode == NcBoundaryMode::HarmonicMixtureFiglikeHF) {
    return static_cast<int>(NcBoundaryMode::HarmonicMixtureFiglikeHF);
  }
'''
        text = text[:m.end()] + insert + text[m.end():]
    # Name switch cases.
    if 'return "harmonic_mixture_multiscale"' not in text:
        # Insert before default/unknown in nc_boundary_mode_name or before first existing hmix case.
        name_func = re.search(r"const\s+char\*\s+nc_boundary_mode_name\s*\([^)]*\)\s*\{.*?\n\}", text, re.S)
        if name_func:
            func = name_func.group(0)
            insert = '''    case NcBoundaryMode::HarmonicMixtureMultiscale: return "harmonic_mixture_multiscale";
    case NcBoundaryMode::HarmonicMixtureFiglikeHF: return "harmonic_mixture_figlike_hf";
'''
            pos = func.find("case NcBoundaryMode::HarmonicMixtureSmooth")
            if pos < 0:
                pos = func.rfind("default:")
            if pos < 0:
                pos = func.rfind("}")
            func2 = func[:pos] + insert + func[pos:]
            text = text[:name_func.start()] + func2 + text[name_func.end():]
    # Parser fallback.
    parse_func = re.search(r"NcBoundaryMode\s+parse_nc_boundary_mode\s*\([^)]*\)\s*\{.*?\n\}", text, re.S)
    if parse_func and "harmonic_mixture_multiscale" not in parse_func.group(0):
        pat = r"(NcBoundaryMode\s+parse_nc_boundary_mode\s*\(\s*const\s+char\*\s+text\s*\)\s*\{)"
        m = re.search(pat, text)
        if not m:
            raise SystemExit("could not find parse_nc_boundary_mode")
        insert = '''
  std::string __hmix_extra_s = text ? text : "";
  for (char& c : __hmix_extra_s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (__hmix_extra_s == "harmonic_mixture_multiscale" ||
      __hmix_extra_s == "mixture_multiscale" ||
      __hmix_extra_s == "hmix_multiscale" ||
      __hmix_extra_s == "multiscale") {
    return NcBoundaryMode::HarmonicMixtureMultiscale;
  }
  if (__hmix_extra_s == "harmonic_mixture_figlike_hf" ||
      __hmix_extra_s == "mixture_figlike_hf" ||
      __hmix_extra_s == "hmix_figlike_hf" ||
      __hmix_extra_s == "figlike_hf" ||
      __hmix_extra_s == "fig_hf") {
    return NcBoundaryMode::HarmonicMixtureFiglikeHF;
  }
'''
        text = text[:m.end()] + insert + text[m.end():]
    return text


def plan() -> list[Change]:
    changes: list[Change] = []
    h = read(HEADER)
    changes.append(Change(HEADER, h, add_extra_functions(h)))
    mh = read(MODE_HEADER)
    changes.append(Change(MODE_HEADER, mh, patch_enum(mh)))
    ev = read(EVAL)
    changes.append(Change(EVAL, ev, patch_eval_cli(ev)))
    cu = read(CUBQL)
    changes.append(Change(CUBQL, cu, patch_cubql(cu)))
    return changes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not args.check and not args.apply:
        ap.error("pass --check or --apply")
    changes = plan()
    changed = [c for c in changes if c.changed]
    if args.check:
        if changed:
            print("Would modify:")
            for c in changed:
                print(f"  {c.path}")
        else:
            print("No changes needed; multiscale harmonic mixture modes appear registered.")
    if args.apply:
        for c in changed:
            backup = c.path.with_suffix(c.path.suffix + ".pre_hmix_multiscale")
            if not backup.exists():
                shutil.copy2(c.path, backup)
            c.path.write_text(c.after, encoding="utf-8")
            print(f"patched {c.path}  backup={backup}")
        if not changed:
            print("No changes applied; already up to date.")


if __name__ == "__main__":
    main()
