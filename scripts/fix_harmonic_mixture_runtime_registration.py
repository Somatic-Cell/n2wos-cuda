#!/usr/bin/env python3
"""Patch cubql_bvh.cu so harmonic_mixture_* modes work at runtime.

This is intentionally conservative: it adds early-return branches for the two
new boundary modes rather than trying to rewrite existing switch statements.
Run from the repository root.
"""
from pathlib import Path
import re
import sys

p = Path('src/cuda/cubql_bvh.cu')
if not p.exists():
    raise SystemExit(f'not found: {p}')

s = p.read_text(encoding='utf-8')
orig = s

if '#include "n2wos/nc_harmonic_mixture.hpp"' not in s:
    # Insert after the first include if possible.
    s = re.sub(r'(^#include .*$)', r'\1\n#include "n2wos/nc_harmonic_mixture.hpp"', s, count=1, flags=re.M)

# 1) Device-side boundary evaluator: add early returns for device int modes.
if 'nc_harmonic_mixture_figlike_value(p)' not in s:
    pat = r'(__host__\s+__device__\s+inline\s+float\s+nc_boundary_device\s*\([^)]*\)\s*\{)'
    m = re.search(pat, s)
    if not m:
        raise SystemExit('could not find nc_boundary_device(...) function')
    insert = '''\n  // Harmonic mixture manufactured solutions.  Keep these branches early so\n  // both host and device callers can use the explicit device mode ids.\n  if (mode == static_cast<int>(NcBoundaryMode::HarmonicMixtureSmooth) || mode == 13) {\n    return nc_harmonic_mixture_smooth_value(p);\n  }\n  if (mode == static_cast<int>(NcBoundaryMode::HarmonicMixtureFiglike) || mode == 14) {\n    return nc_harmonic_mixture_figlike_value(p);\n  }\n'''
    s = s[:m.end()] + insert + s[m.end():]

# 2) Host enum -> device int mapping. The error without a trailing ": <mode>"
#    comes from this function when the enum is not mapped.
if 'return static_cast<int>(NcBoundaryMode::HarmonicMixtureFiglike);' not in s:
    pat = r'(int\s+nc_boundary_to_device\s*\(\s*NcBoundaryMode\s+mode\s*\)\s*\{)'
    m = re.search(pat, s)
    if not m:
        raise SystemExit('could not find nc_boundary_to_device(...) function')
    insert = '''\n  if (mode == NcBoundaryMode::HarmonicMixtureSmooth) {\n    return static_cast<int>(NcBoundaryMode::HarmonicMixtureSmooth);\n  }\n  if (mode == NcBoundaryMode::HarmonicMixtureFiglike) {\n    return static_cast<int>(NcBoundaryMode::HarmonicMixtureFiglike);\n  }\n'''
    s = s[:m.end()] + insert + s[m.end():]

# 3) Library parser fallback. eval_tcnn_nc_wos currently has a CLI parser branch,
#    but other tools and fallback paths call n2wos::parse_nc_boundary_mode.
if 'mixture_figlike' not in re.search(r'NcBoundaryMode\s+parse_nc_boundary_mode\s*\([^)]*\)\s*\{.*?\n\}', s, re.S).group(0):
    pat = r'(NcBoundaryMode\s+parse_nc_boundary_mode\s*\(\s*const\s+char\*\s+text\s*\)\s*\{)'
    m = re.search(pat, s)
    if not m:
        raise SystemExit('could not find parse_nc_boundary_mode(...) function')
    insert = '''\n  // Keep parser aliases in sync with eval_tcnn_nc_wos CLI aliases.\n  std::string __hmix_s_for_early_parse = text ? text : "";\n  for (char& c : __hmix_s_for_early_parse) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));\n  if (__hmix_s_for_early_parse == "harmonic_mixture_smooth" ||\n      __hmix_s_for_early_parse == "mixture_smooth" ||\n      __hmix_s_for_early_parse == "hmix_smooth") {\n    return NcBoundaryMode::HarmonicMixtureSmooth;\n  }\n  if (__hmix_s_for_early_parse == "harmonic_mixture_figlike" ||\n      __hmix_s_for_early_parse == "mixture_figlike" ||\n      __hmix_s_for_early_parse == "hmix_figlike") {\n    return NcBoundaryMode::HarmonicMixtureFiglike;\n  }\n'''
    s = s[:m.end()] + insert + s[m.end():]

if s == orig:
    print('No changes needed; harmonic mixture runtime registration appears present.')
else:
    backup = p.with_suffix(p.suffix + '.pre_hmix_fix')
    if not backup.exists():
        backup.write_text(orig, encoding='utf-8')
    p.write_text(s, encoding='utf-8')
    print(f'Patched {p}; backup at {backup}')
