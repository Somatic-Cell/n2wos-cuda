#!/usr/bin/env python3
"""Patch the local NC/WoS evaluator with manufactured unscreened Poisson modes.

This is an experimental source-aware patcher for the evolving n2wos-cuda branch.
It adds two problem modes:

  poisson_multiscale
  poisson_figlike_hf

The intended equation is -Delta u = f with Dirichlet boundary u|dOmega.
There is no absorption/screening term, hence beta=1.

The patcher updates enum/parser/name/device boundary mapping and tries to insert
source accumulation into the core WoS path.  Prefix-source accumulation for
NC-WoS/NC+2LMC is source-tree dependent; if the automatic checks at the end
report a missing prefix patch, use depth_m=0 only until the indicated functions
are patched manually.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
HEADER = ROOT / 'include' / 'n2wos' / 'tcnn_nc_wos.hpp'
CUDA = ROOT / 'src' / 'cuda' / 'cubql_bvh.cu'
TOOL = ROOT / 'src' / 'tools' / 'eval_tcnn_nc_wos.cu'
POISSON_HEADER = 'n2wos/nc_poisson_manufactured.hpp'

ENUM_INSERT = '''\n  PoissonMultiscale = 30,\n  PoissonFiglikeHf = 31,'''


def read(p: Path) -> str:
    if not p.exists():
        raise SystemExit(f'not found: {p}')
    return p.read_text(encoding='utf-8')


def write_if_changed(p: Path, old: str, new: str, apply: bool) -> bool:
    if old == new:
        return False
    if apply:
        bak = p.with_suffix(p.suffix + '.pre_poisson')
        if not bak.exists():
            bak.write_text(old, encoding='utf-8')
        p.write_text(new, encoding='utf-8')
    return True


def add_include(s: str) -> str:
    line = f'#include "{POISSON_HEADER}"\n'
    if line.strip() in s:
        return s
    return re.sub(r'(^#include .*$)', r'\1\n' + line.rstrip(), s, count=1, flags=re.M)


def patch_header(s: str) -> str:
    if 'PoissonMultiscale' in s:
        return s
    # Insert before closing brace of enum class NcBoundaryMode.
    m = re.search(r'enum\s+class\s+NcBoundaryMode\s*\{(?P<body>.*?)\n\s*\};', s, re.S)
    if not m:
        raise SystemExit('could not find enum class NcBoundaryMode in header')
    body = m.group('body')
    insert_at = m.start('body') + len(body)
    return s[:insert_at] + ENUM_INSERT + s[insert_at:]


def patch_cli_parser(s: str) -> str:
    if 'poisson_figlike_hf' in s:
        return s
    m = re.search(r'(n2wos::NcBoundaryMode\s+parse_nc_boundary_mode_cli\s*\([^)]*\)\s*\{)', s)
    if not m:
        return s
    insert = '''\n  if (s == "poisson_multiscale" || s == "manufactured_poisson_multiscale" ||\n      s == "poisson_mscale") {\n    return n2wos::NcBoundaryMode::PoissonMultiscale;\n  }\n  if (s == "poisson_figlike_hf" || s == "manufactured_poisson_figlike_hf" ||\n      s == "poisson_hf") {\n    return n2wos::NcBoundaryMode::PoissonFiglikeHf;\n  }\n'''
    return s[:m.end()] + insert + s[m.end():]


def patch_cuda_registration(s: str) -> str:
    s = add_include(s)

    # Boundary evaluator: u(x) is the Dirichlet boundary value and analytic truth.
    if 'nc_poisson_manufactured_u(p, mode)' not in s:
        m = re.search(r'(__host__\s+__device__\s+inline\s+float\s+nc_boundary_device\s*\([^)]*\)\s*\{)', s)
        if not m:
            raise SystemExit('could not find nc_boundary_device(...)')
        insert = '''\n  if (nc_poisson_manufactured_is_mode(mode)) {\n    return nc_poisson_manufactured_u(p, mode);\n  }\n'''
        s = s[:m.end()] + insert + s[m.end():]

    # enum -> device integer mapping.
    if 'NcBoundaryMode::PoissonFiglikeHf' not in re.search(r'int\s+nc_boundary_to_device\s*\([^)]*\)\s*\{.*?\n\}', s, re.S).group(0):
        m = re.search(r'(int\s+nc_boundary_to_device\s*\(\s*NcBoundaryMode\s+mode\s*\)\s*\{)', s)
        if not m:
            raise SystemExit('could not find nc_boundary_to_device(...)')
        insert = '''\n  if (mode == NcBoundaryMode::PoissonMultiscale) {\n    return kNcPoissonMultiscaleMode;\n  }\n  if (mode == NcBoundaryMode::PoissonFiglikeHf) {\n    return kNcPoissonFiglikeHfMode;\n  }\n'''
        s = s[:m.end()] + insert + s[m.end():]

    # Name function.
    name_match = re.search(r'const\s+char\*\s+nc_boundary_mode_name\s*\([^)]*\)\s*\{.*?\n\}', s, re.S)
    if name_match and 'poisson_figlike_hf' not in name_match.group(0):
        m = re.search(r'(const\s+char\*\s+nc_boundary_mode_name\s*\([^)]*\)\s*\{)', s)
        insert = '''\n    case NcBoundaryMode::PoissonMultiscale: return "poisson_multiscale";\n    case NcBoundaryMode::PoissonFiglikeHf: return "poisson_figlike_hf";\n'''
        s = s[:m.end()] + insert + s[m.end():]

    # Library parser.
    parse_match = re.search(r'NcBoundaryMode\s+parse_nc_boundary_mode\s*\([^)]*\)\s*\{.*?\n\}', s, re.S)
    if parse_match and 'poisson_figlike_hf' not in parse_match.group(0):
        m = re.search(r'(NcBoundaryMode\s+parse_nc_boundary_mode\s*\(\s*const\s+char\*\s+text\s*\)\s*\{)', s)
        insert = '''\n  std::string __poisson_s = text ? text : "";\n  for (char& c : __poisson_s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));\n  if (__poisson_s == "poisson_multiscale" || __poisson_s == "manufactured_poisson_multiscale" ||\n      __poisson_s == "poisson_mscale") return NcBoundaryMode::PoissonMultiscale;\n  if (__poisson_s == "poisson_figlike_hf" || __poisson_s == "manufactured_poisson_figlike_hf" ||\n      __poisson_s == "poisson_hf") return NcBoundaryMode::PoissonFiglikeHf;\n'''
        s = s[:m.end()] + insert + s[m.end():]

    return s


def patch_pure_wos_core(s: str) -> str:
    # This patches the core nc_wos_from_point path.  It is intentionally limited
    # to common variable names seen in this branch.  If it cannot find the right
    # pattern, it leaves the file unchanged and the check stage will report it.
    if 'float __poisson_source_accum = 0.0f;' in s:
        return s

    # Add accumulator after the first local point variable in nc_wos_from_point.
    fn = re.search(r'((?:__device__|__host__\s+__device__|static\s+__device__|inline\s+__device__)[^{;]*\s+nc_wos_from_point\s*\([^)]*\)\s*\{)', s)
    if not fn:
        return s
    pos = fn.end()
    s = s[:pos] + '\n  float __poisson_source_accum = 0.0f;\n' + s[pos:]

    # Boundary returns: add accumulator to direct boundary returns in this function.
    start = fn.start()
    next_fn = re.search(r'\n(?:__global__|__device__|int\s+nc_boundary_to_device|const\s+char\*)', s[pos:])
    end = pos + next_fn.start() if next_fn else len(s)
    block = s[start:end]
    block2 = block.replace('return nc_boundary_device(', 'return __poisson_source_accum + nc_boundary_device(')
    # Some code stores boundary in variable before return.
    block2 = re.sub(r'return\s+boundary\s*;', 'return __poisson_source_accum + boundary;', block2)

    # Add center Green source contribution before WoS position update.  Try common
    # radius variable names; the first match wins.
    if 'nc_poisson_center_green_contribution' not in block2:
        patterns = [
            (r'(const\s+float\s+r\s*=\s*[^;]+;)', 'r'),
            (r'(float\s+r\s*=\s*[^;]+;)', 'r'),
            (r'(const\s+float\s+radius\s*=\s*[^;]+;)', 'radius'),
            (r'(float\s+radius\s*=\s*[^;]+;)', 'radius'),
        ]
        for pat, var in patterns:
            mm = re.search(pat, block2)
            if mm:
                add = f"\n    __poisson_source_accum += nc_poisson_center_green_contribution(p0, {var}, boundary_mode);"
                block2 = block2[:mm.end()] + add + block2[mm.end():]
                break

    return s[:start] + block2 + s[end:]


def patch_prefix_best_effort(s: str) -> str:
    # Full Poisson m>0 support requires prefix source A_m.  Because prefix kernels
    # have changed often, this patcher only adds a marker and emits a warning if
    # no obvious prefix function was patched.  The core depth_m=0 and pure label
    # path still works if nc_wos_from_point was patched.
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    if args.check == args.apply:
        raise SystemExit('pass exactly one of --check or --apply')

    changes = []

    h0 = read(HEADER)
    h1 = patch_header(h0)
    if write_if_changed(HEADER, h0, h1, args.apply): changes.append(str(HEADER))

    t0 = read(TOOL)
    t1 = add_include(t0)
    t1 = patch_cli_parser(t1)
    if write_if_changed(TOOL, t0, t1, args.apply): changes.append(str(TOOL))

    c0 = read(CUDA)
    c1 = patch_cuda_registration(c0)
    c1 = patch_pure_wos_core(c1)
    c1 = patch_prefix_best_effort(c1)
    if write_if_changed(CUDA, c0, c1, args.apply): changes.append(str(CUDA))

    print(('Would patch' if args.check else 'Patched') + ':')
    if changes:
        for c in changes: print('  ' + c)
    else:
        print('  no changes needed')

    # Diagnostics.
    diag = read(CUDA) if args.apply else c1
    has_core_accum = 'float __poisson_source_accum = 0.0f;' in diag and 'nc_poisson_center_green_contribution' in diag
    print('\nDiagnostics:')
    print(f'  core nc_wos_from_point source accumulation: {"yes" if has_core_accum else "NO"}')
    print('  prefix A_m accumulation for depth_m>0: not automatically guaranteed by this patcher')
    print('\nRecommended first test: use --boundary poisson_multiscale with --depth-m 0.')
    print('Do not use m>0 for claims until prefix A_m support has been verified in the source.')

if __name__ == '__main__':
    main()
