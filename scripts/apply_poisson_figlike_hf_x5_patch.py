#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


HEADER_FUNCTIONS = r'''

// Higher-frequency manufactured Poisson field. This is a bounded Fourier
// mixture. For Poisson we can prescribe an interior source f=-Delta u, so
// internal spatial frequencies need not be generated from boundary data alone.
//
// The returned value is u(x). The source routine returns f(x)=-Delta u. The
// x5 suffix means that the dominant angular frequencies are roughly five times
// larger than the low/mid-frequency Poisson diagnostic modes.
template <typename P>
N2WOS_HD_INLINE float nc_poisson_figlike_hf_x5_value(P p) {
  const float x = p.x;
  const float y = p.y;
  const float z = p.z;

  // Moderate amplitude keeps comparisons from being dominated by dynamic range.
  float u = 0.0f;
  u += 0.090f * sinf(12.0f * x +  4.0f * y +  1.5f * z + 0.30f);
  u += 0.075f * cosf(-5.0f * x + 13.5f * y +  2.0f * z - 0.70f);
  u += 0.060f * sinf( 7.5f * x -  3.5f * y + 15.0f * z + 1.10f);
  u += 0.050f * cosf(14.0f * x - 10.0f * y +  4.0f * z + 2.20f);
  u += 0.045f * sinf(-9.0f * x +  6.5f * y + 12.0f * z - 1.30f);
  u += 0.035f * cosf(18.0f * x +  2.5f * y -  7.0f * z + 0.80f);

  // Weak low-frequency component to make large-scale bias visible.
  u += 0.030f * (0.60f * x - 0.35f * y + 0.20f * z);
  return u;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_figlike_hf_x5_source(P p) {
  const float x = p.x;
  const float y = p.y;
  const float z = p.z;

  float f = 0.0f;
  // For a*sin(k.x+phi) or a*cos(k.x+phi), -Delta gives |k|^2 times the same term.
  f += 0.090f * (12.0f*12.0f +  4.0f* 4.0f +  1.5f* 1.5f) * sinf(12.0f * x +  4.0f * y +  1.5f * z + 0.30f);
  f += 0.075f * ( 5.0f* 5.0f + 13.5f*13.5f +  2.0f* 2.0f) * cosf(-5.0f * x + 13.5f * y +  2.0f * z - 0.70f);
  f += 0.060f * ( 7.5f* 7.5f +  3.5f* 3.5f + 15.0f*15.0f) * sinf( 7.5f * x -  3.5f * y + 15.0f * z + 1.10f);
  f += 0.050f * (14.0f*14.0f + 10.0f*10.0f +  4.0f* 4.0f) * cosf(14.0f * x - 10.0f * y +  4.0f * z + 2.20f);
  f += 0.045f * ( 9.0f* 9.0f +  6.5f* 6.5f + 12.0f*12.0f) * sinf(-9.0f * x +  6.5f * y + 12.0f * z - 1.30f);
  f += 0.035f * (18.0f*18.0f +  2.5f* 2.5f +  7.0f* 7.0f) * cosf(18.0f * x +  2.5f * y -  7.0f * z + 0.80f);
  return f;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_figlike_hf_x5_center_green_contribution(P p, float radius) {
  return (radius * radius / 6.0f) * nc_poisson_figlike_hf_x5_source(p);
}
'''


def find_enum_body(text: str, enum_name: str) -> tuple[int, int]:
    m = re.search(r"enum\s+class\s+" + re.escape(enum_name) + r"\s*\{", text)
    if not m:
        raise RuntimeError(f"enum class {enum_name} not found")
    brace = text.find("{", m.end() - 1)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return brace + 1, i
    raise RuntimeError(f"enum class {enum_name} not closed")


def find_function(text: str, name: str) -> tuple[int, int]:
    m = re.search(r"(^|\n)[\w:<>,\s\*&]+\b" + re.escape(name) + r"\s*\([^;]*?\)\s*\{", text, re.S)
    if not m:
        raise RuntimeError(f"function {name} not found")
    start = m.start() + (1 if text[m.start()] == "\n" else 0)
    brace = text.find("{", m.end() - 1)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise RuntimeError(f"function {name} not closed")


def patch_header_functions(path: Path) -> None:
    text = path.read_text()
    if "nc_poisson_figlike_hf_x5_value" not in text:
        text = text.rstrip() + "\n" + HEADER_FUNCTIONS + "\n"
    path.write_text(text)


def patch_enum(path: Path) -> None:
    text = path.read_text()
    if "PoissonFiglikeHfX5" in text:
        return
    start, end = find_enum_body(text, "NcBoundaryMode")
    body = text[start:end]
    lines = body.splitlines(True)
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if "PoissonFiglikeHf" in line and "PoissonFiglikeHfX5" not in line:
            indent = re.match(r"\s*", line).group(0)
            m = re.search(r"=\s*(\d+)", line)
            if m:
                out.append(f"{indent}PoissonFiglikeHfX5 = {int(m.group(1)) + 1},\n")
            else:
                out.append(f"{indent}PoissonFiglikeHfX5,\n")
            inserted = True
    if not inserted:
        out.append("  PoissonFiglikeHfX5,\n")
    text = text[:start] + "".join(out) + text[end:]
    path.write_text(text)


def patch_eval_cli(path: Path) -> None:
    text = path.read_text()
    start, end = find_function(text, "parse_nc_boundary_mode_cli")
    fn = text[start:end]
    if "poisson_figlike_hf_x5" not in fn:
        marker = "const std::string s = normalize_mode_text(text);"
        pos = fn.find(marker)
        if pos < 0:
            raise RuntimeError("could not find normalized string line in parse_nc_boundary_mode_cli")
        line_end = fn.find("\n", pos)
        insertion = (
            '  if (s == "poisson_figlike_hf_x5" || s == "poisson_hf_x5" || s == "poisson_figlike_x5") {\n'
            '    return n2wos::NcBoundaryMode::PoissonFiglikeHfX5;\n'
            '  }\n'
        )
        fn = fn[:line_end + 1] + insertion + fn[line_end + 1:]
        text = text[:start] + fn + text[end:]
    path.write_text(text)


def patch_cubql(path: Path) -> None:
    text = path.read_text()

    if "kNcPoissonFiglikeHfX5Mode" not in text:
        m = re.search(r"(kNcPoissonFiglikeHfMode\s*=\s*)(\d+)(\s*;)", text)
        if m:
            line_end = text.find("\n", m.end())
            if line_end < 0:
                line_end = len(text)
            text = text[:line_end] + f"\nstatic constexpr int kNcPoissonFiglikeHfX5Mode = {int(m.group(2)) + 1};" + text[line_end:]

    start, end = find_function(text, "nc_boundary_device")
    fn = text[start:end]
    if "nc_poisson_figlike_hf_x5_value" not in fn:
        open_brace = fn.find("{")
        mode_expr = "kNcPoissonFiglikeHfX5Mode" if "kNcPoissonFiglikeHfX5Mode" in text else "static_cast<int>(NcBoundaryMode::PoissonFiglikeHfX5)"
        insertion = f"\n  if (mode == {mode_expr}) {{\n    return nc_poisson_figlike_hf_x5_value(p);\n  }}\n"
        fn = fn[:open_brace + 1] + insertion + fn[open_brace + 1:]
        text = text[:start] + fn + text[end:]

    start, end = find_function(text, "nc_poisson_center_green_contribution")
    fn = text[start:end]
    if "nc_poisson_figlike_hf_x5_center_green_contribution" not in fn:
        open_brace = fn.find("{")
        mode_expr = "kNcPoissonFiglikeHfX5Mode" if "kNcPoissonFiglikeHfX5Mode" in text else "static_cast<int>(NcBoundaryMode::PoissonFiglikeHfX5)"
        insertion = f"\n  if (mode == {mode_expr}) {{\n    return nc_poisson_figlike_hf_x5_center_green_contribution(p, radius);\n  }}\n"
        fn = fn[:open_brace + 1] + insertion + fn[open_brace + 1:]
        text = text[:start] + fn + text[end:]

    start, end = find_function(text, "nc_boundary_to_device")
    fn = text[start:end]
    if "PoissonFiglikeHfX5" not in fn:
        open_brace = fn.find("{")
        if "kNcPoissonFiglikeHfX5Mode" in text:
            insertion = "\n  if (mode == NcBoundaryMode::PoissonFiglikeHfX5) {\n    return kNcPoissonFiglikeHfX5Mode;\n  }\n"
        else:
            insertion = "\n  if (mode == NcBoundaryMode::PoissonFiglikeHfX5) {\n    return static_cast<int>(NcBoundaryMode::PoissonFiglikeHfX5);\n  }\n"
        fn = fn[:open_brace + 1] + insertion + fn[open_brace + 1:]
        text = text[:start] + fn + text[end:]

    start, end = find_function(text, "nc_boundary_mode_name")
    fn = text[start:end]
    if "poisson_figlike_hf_x5" not in fn:
        switch_pos = re.search(r"switch\s*\(\s*mode\s*\)\s*\{", fn)
        if not switch_pos:
            raise RuntimeError("switch(mode) not found in nc_boundary_mode_name")
        insertion = '    case NcBoundaryMode::PoissonFiglikeHfX5: return "poisson_figlike_hf_x5";\n'
        fn = fn[:switch_pos.end()] + "\n" + insertion + fn[switch_pos.end():]
        text = text[:start] + fn + text[end:]

    start, end = find_function(text, "parse_nc_boundary_mode")
    fn = text[start:end]
    if "poisson_figlike_hf_x5" not in fn:
        pos = fn.find("std::string s")
        if pos < 0:
            pos = fn.find("auto s")
        if pos < 0:
            raise RuntimeError("could not find local string s in parse_nc_boundary_mode")
        line_end = fn.find("\n", pos)
        insertion = '  if (s == "poisson_figlike_hf_x5" || s == "poisson_hf_x5" || s == "poisson_figlike_x5") return NcBoundaryMode::PoissonFiglikeHfX5;\n'
        fn = fn[:line_end + 1] + insertion + fn[line_end + 1:]
        text = text[:start] + fn + text[end:]

    path.write_text(text)


def main() -> None:
    root = Path.cwd()
    required = [
        root / "include/n2wos/nc_poisson_manufactured.hpp",
        root / "include/n2wos/tcnn_nc_wos.hpp",
        root / "src/tools/eval_tcnn_nc_wos.cu",
        root / "src/cuda/cubql_bvh.cu",
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"missing required file: {p}")

    patch_header_functions(required[0])
    patch_enum(required[1])
    patch_eval_cli(required[2])
    patch_cubql(required[3])

    print("Added poisson_figlike_hf_x5 mode.")
    print("Please verify:")
    print("  grep -R \"poisson_figlike_hf_x5\\|PoissonFiglikeHfX5\" -n include src scripts | head -120")
    print("Then build:")
    print("  cmake --build ./build/cuda-release-cubql-tcnn --target n2wos_eval_tcnn_nc_wos -j")


if __name__ == "__main__":
    main()
