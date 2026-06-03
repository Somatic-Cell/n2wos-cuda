#!/usr/bin/env python3
"""
Generate human-checkable plane visualizations for Neural Cache experiments.

This postprocess is intentionally independent from the CUDA executable. It reads
an estimates CSV and a slice mask, then writes reference/cache/error images. The
most common n2wos slice output stores only interior pixels in the estimates CSV;
this script reconstructs the full image by filling values into mask-inside
pixels in raster order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import struct
import sys
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


RGB = Tuple[int, int, int]


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_ppm_p6(path: str) -> Tuple[int, int, List[RGB]]:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise RuntimeError(f"unsupported PPM magic in {path}: {magic!r}")

        tokens: List[bytes] = []
        while len(tokens) < 3:
            line = f.readline()
            if not line:
                raise RuntimeError(f"unexpected EOF while reading PPM header: {path}")
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            tokens.extend(line.split())

        width = int(tokens[0])
        height = int(tokens[1])
        maxval = int(tokens[2])
        if maxval != 255:
            raise RuntimeError(f"unsupported PPM maxval in {path}: {maxval}")

        raw = f.read(width * height * 3)
        if len(raw) != width * height * 3:
            raise RuntimeError(f"bad PPM payload size in {path}")
        pixels = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
        return width, height, pixels


def write_ppm_p6(path: str, width: int, height: int, pixels: Sequence[RGB]) -> None:
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        buf = bytearray()
        for r, g, b in pixels:
            buf.extend((max(0, min(255, int(r))),
                        max(0, min(255, int(g))),
                        max(0, min(255, int(b)))))
        f.write(buf)


def write_png_rgb(path: str, width: int, height: int, pixels: Sequence[RGB]) -> None:
    """Write an RGB PNG using only the Python standard library."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type: none
        row = pixels[y * width:(y + 1) * width]
        for r, g, b in row:
            raw.extend((max(0, min(255, int(r))),
                        max(0, min(255, int(g))),
                        max(0, min(255, int(b)))))
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    data += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    if key not in row:
        return None
    s = row[key]
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def mask_inside(mask_pixels: Sequence[RGB]) -> List[bool]:
    # Accept either white-inside masks or grayscale masks.
    return [(r + g + b) > 3 * 127 for (r, g, b) in mask_pixels]


def infer_row_to_pixel(rows: Sequence[Dict[str, str]], mask: Sequence[bool], width: int, height: int) -> List[int]:
    """Return a pixel index for every estimate row."""
    n = len(rows)
    if n == width * height:
        return list(range(n))

    inside_indices = [i for i, m in enumerate(mask) if m]
    if n == len(inside_indices):
        return inside_indices

    first = rows[0] if rows else {}
    xkey = next((k for k in ("pixel_x", "ix", "grid_x") if k in first), None)
    ykey = next((k for k in ("pixel_y", "iy", "grid_y") if k in first), None)
    if xkey and ykey:
        out: List[int] = []
        for r in rows:
            ix = int(r[xkey])
            iy = int(r[ykey])
            if ix < 0 or ix >= width or iy < 0 or iy >= height:
                raise RuntimeError(f"pixel coordinate out of bounds: ({ix}, {iy})")
            out.append(iy * width + ix)
        return out

    raise RuntimeError(
        f"cannot map {n} estimate rows to {width}x{height} mask with "
        f"{len(inside_indices)} inside pixels"
    )


def values_to_image(rows: Sequence[Dict[str, str]], row_to_pixel: Sequence[int], width: int, height: int,
                    column: str) -> List[Optional[float]]:
    vals: List[Optional[float]] = [None] * (width * height)
    for r, pix in zip(rows, row_to_pixel):
        vals[pix] = parse_float(r, column)
    return vals


def reference_values_to_image(ref_rows: Sequence[Dict[str, str]], est_rows: Sequence[Dict[str, str]],
                              row_to_pixel: Sequence[int], width: int, height: int,
                              column: str = "pure_mean") -> List[Optional[float]]:
    ref_by_id = {int(r["point_id"]): r for r in ref_rows if "point_id" in r}
    vals: List[Optional[float]] = [None] * (width * height)
    for er, pix in zip(est_rows, row_to_pixel):
        pid = int(er["point_id"])
        rr = ref_by_id.get(pid)
        vals[pix] = parse_float(rr, column) if rr is not None else None
    return vals


def finite_values(vals: Sequence[Optional[float]], mask: Sequence[bool]) -> List[float]:
    out: List[float] = []
    for v, m in zip(vals, mask):
        if m and v is not None and math.isfinite(v):
            out.append(float(v))
    return out


def quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    x = q * (len(sorted_vals) - 1)
    i0 = int(math.floor(x))
    i1 = int(math.ceil(x))
    if i0 == i1:
        return sorted_vals[i0]
    t = x - i0
    return sorted_vals[i0] * (1 - t) + sorted_vals[i1] * t


def robust_range(vals: Sequence[float], lo_q: float = 0.01, hi_q: float = 0.99) -> Tuple[float, float]:
    if not vals:
        return (-1.0, 1.0)
    s = sorted(vals)
    lo = quantile(s, lo_q)
    hi = quantile(s, hi_q)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = min(s)
        hi = max(s)
    if hi <= lo:
        c = lo
        return (c - 1.0, c + 1.0)
    return (lo, hi)


def symmetric_range(vals: Sequence[float], q: float = 0.99) -> Tuple[float, float]:
    if not vals:
        return (-1.0, 1.0)
    s = sorted(abs(v) for v in vals)
    a = quantile(s, q)
    if not math.isfinite(a) or a <= 0:
        a = max(s) if s else 1.0
    if a <= 0:
        a = 1.0
    return (-a, a)


def gray(v: float, vmin: float, vmax: float) -> RGB:
    if vmax <= vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    g = int(round(255 * t))
    return (g, g, g)


def divmap(v: float, vmin: float, vmax: float) -> RGB:
    if vmax <= vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        u = t / 0.5
        return (int(255 * u), int(255 * u), 255)
    u = (t - 0.5) / 0.5
    return (255, int(255 * (1 - u)), int(255 * (1 - u)))


def render(values: Sequence[Optional[float]], mask: Sequence[bool], width: int, height: int,
           out_prefix: str, cmap: str, value_range: Tuple[float, float]) -> Dict[str, object]:
    pixels: List[RGB] = []
    for v, m in zip(values, mask):
        if not m or v is None or not math.isfinite(v):
            pixels.append((255, 255, 255))
        elif cmap == "gray":
            pixels.append(gray(float(v), *value_range))
        elif cmap == "diverging":
            pixels.append(divmap(float(v), *value_range))
        else:
            raise RuntimeError(f"unknown cmap: {cmap}")

    ppm_path = out_prefix + ".ppm"
    png_path = out_prefix + ".png"
    write_ppm_p6(ppm_path, width, height, pixels)
    write_png_rgb(png_path, width, height, pixels)
    return {"ppm": ppm_path, "png": png_path, "range": list(value_range), "cmap": cmap}


def diff_values(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for x, y in zip(a, b):
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            out.append(None)
        else:
            out.append(float(x) - float(y))
    return out


def abs_values(a: Sequence[Optional[float]]) -> List[Optional[float]]:
    return [None if v is None or not math.isfinite(v) else abs(float(v)) for v in a]


def rmse(a: Sequence[Optional[float]], b: Sequence[Optional[float]], mask: Sequence[bool]) -> float:
    ss = 0.0
    n = 0
    for x, y, m in zip(a, b, mask):
        if m and x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            d = float(x) - float(y)
            ss += d * d
            n += 1
    return math.sqrt(ss / n) if n else float("nan")


def mae(a: Sequence[Optional[float]], b: Sequence[Optional[float]], mask: Sequence[bool]) -> float:
    s = 0.0
    n = 0
    for x, y, m in zip(a, b, mask):
        if m and x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            s += abs(float(x) - float(y))
            n += 1
    return s / n if n else float("nan")


def variance(a: Sequence[Optional[float]], mask: Sequence[bool]) -> float:
    vals = finite_values(a, mask)
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def corr(a: Sequence[Optional[float]], b: Sequence[Optional[float]], mask: Sequence[bool]) -> float:
    aa: List[float] = []
    bb: List[float] = []
    for x, y, m in zip(a, b, mask):
        if m and x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            aa.append(float(x))
            bb.append(float(y))
    if len(aa) < 2:
        return float("nan")
    ma = sum(aa) / len(aa)
    mb = sum(bb) / len(bb)
    va = sum((x - ma) ** 2 for x in aa)
    vb = sum((y - mb) ** 2 for y in bb)
    if va <= 0 or vb <= 0:
        return float("nan")
    cov = sum((x - ma) * (y - mb) for x, y in zip(aa, bb))
    return cov / math.sqrt(va * vb)


@dataclass
class Case:
    estimates_csv: str
    mask_ppm: str
    reference_csv: Optional[str]
    output_dir: str


def guess_mask(estimates_csv: str) -> Optional[str]:
    stem = os.path.splitext(estimates_csv)[0]
    candidates = [
        stem.replace("_estimates", "_slice_mask") + ".ppm",
        stem.replace("_estimates", "") + "_slice_mask.ppm",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    d = os.path.dirname(estimates_csv)
    for fn in os.listdir(d):
        if fn.endswith("_slice_mask.ppm"):
            return os.path.join(d, fn)
    return None


def discover_cases(results_root: str, reference_csv: Optional[str]) -> List[Case]:
    cases: List[Case] = []
    for root, _, files in os.walk(results_root):
        for fn in files:
            if not fn.endswith("_estimates.csv") or fn == "reference_pure_estimates.csv":
                continue
            est = os.path.join(root, fn)
            mask = guess_mask(est)
            if not mask:
                continue
            out = os.path.join(root, "figures")
            cases.append(Case(est, mask, reference_csv, out))
    return sorted(cases, key=lambda c: c.estimates_csv)


def generate(case: Case, require_reference: bool) -> Dict[str, object]:
    os.makedirs(case.output_dir, exist_ok=True)
    rows = read_csv_rows(case.estimates_csv)
    if not rows:
        raise RuntimeError(f"empty estimates CSV: {case.estimates_csv}")

    width, height, mask_pixels = read_ppm_p6(case.mask_ppm)
    mask = mask_inside(mask_pixels)
    row_to_pixel = infer_row_to_pixel(rows, mask, width, height)

    cache = values_to_image(rows, row_to_pixel, width, height, "nc_wos_mean")
    pure = values_to_image(rows, row_to_pixel, width, height, "pure_mean")
    lmc = values_to_image(rows, row_to_pixel, width, height, "nc_2lmc_mean")

    reference: Optional[List[Optional[float]]] = None
    if case.reference_csv and os.path.isfile(case.reference_csv):
        ref_rows = read_csv_rows(case.reference_csv)
        reference = reference_values_to_image(ref_rows, rows, row_to_pixel, width, height)
    elif require_reference:
        raise RuntimeError(f"reference CSV not available for {case.estimates_csv}")

    figures: Dict[str, object] = {}
    metrics: Dict[str, object] = {
        "estimates_csv": case.estimates_csv,
        "mask_ppm": case.mask_ppm,
        "reference_csv": case.reference_csv,
        "inside_pixels": int(sum(mask)),
        "width": width,
        "height": height,
    }

    def render_field(name: str, vals: List[Optional[float]], shared_range: Optional[Tuple[float, float]] = None) -> None:
        fvals = finite_values(vals, mask)
        if not fvals:
            return
        auto_range = robust_range(fvals)
        figures[name] = render(vals, mask, width, height, os.path.join(case.output_dir, name), "gray", auto_range)
        if shared_range is not None:
            figures[name + "_shared_scale"] = render(
                vals, mask, width, height,
                os.path.join(case.output_dir, name + "_shared_scale"),
                "gray", shared_range)

    ref_range: Optional[Tuple[float, float]] = None
    if reference is not None:
        ref_vals = finite_values(reference, mask)
        if ref_vals:
            ref_range = robust_range(ref_vals)
            figures["reference_pure_mean_shared_scale"] = render(
                reference, mask, width, height,
                os.path.join(case.output_dir, "reference_pure_mean_shared_scale"),
                "gray", ref_range)
            metrics["reference"] = {"variance": variance(reference, mask), "robust_range": list(ref_range)}

    render_field("cache_nc_wos_mean", cache, ref_range)
    render_field("pure_wos_mean", pure, ref_range)
    render_field("nc_2lmc_mean", lmc, ref_range)

    if reference is not None:
        for label, vals in [("cache_nc_wos_mean", cache), ("pure_wos_mean", pure), ("nc_2lmc_mean", lmc)]:
            d = diff_values(vals, reference)
            ad = abs_values(d)
            dvals = finite_values(d, mask)
            advals = finite_values(ad, mask)
            if dvals:
                figures[label + "_minus_reference"] = render(
                    d, mask, width, height,
                    os.path.join(case.output_dir, label + "_minus_reference"),
                    "diverging", symmetric_range(dvals))
            if advals:
                figures["abs_" + label + "_minus_reference"] = render(
                    ad, mask, width, height,
                    os.path.join(case.output_dir, "abs_" + label + "_minus_reference"),
                    "gray", robust_range(advals, 0.0, 0.99))
            metrics[label] = {
                "rmse_vs_reference": rmse(vals, reference, mask),
                "mae_vs_reference": mae(vals, reference, mask),
                "corr_vs_reference": corr(vals, reference, mask),
                "variance": variance(vals, mask),
            }

    manifest = {"case": case.__dict__, "figures": figures, "metrics": metrics}
    with open(os.path.join(case.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(case.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate NC cache / reference / error images.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--estimates-csv")
    g.add_argument("--results-root")
    ap.add_argument("--mask-ppm")
    ap.add_argument("--reference-estimates-csv")
    ap.add_argument("--output-dir")
    ap.add_argument("--require-reference", action="store_true")
    ap.add_argument("--print-summary", action="store_true")
    args = ap.parse_args()

    if args.estimates_csv:
        mask = args.mask_ppm or guess_mask(args.estimates_csv)
        if not mask:
            raise RuntimeError(f"could not find mask for {args.estimates_csv}")
        case = Case(
            estimates_csv=os.path.abspath(args.estimates_csv),
            mask_ppm=os.path.abspath(mask),
            reference_csv=os.path.abspath(args.reference_estimates_csv) if args.reference_estimates_csv else None,
            output_dir=os.path.abspath(args.output_dir or os.path.join(os.path.dirname(args.estimates_csv), "figures")),
        )
        manifests = [generate(case, args.require_reference)]
    else:
        cases = discover_cases(args.results_root, args.reference_estimates_csv)
        if not cases:
            raise RuntimeError(f"no visualizable estimates found under {args.results_root}")
        manifests = [generate(c, args.require_reference) for c in cases]
        with open(os.path.join(args.results_root, "nc_visualizations_index.json"), "w", encoding="utf-8") as f:
            json.dump({"cases": manifests}, f, indent=2)

    if args.print_summary:
        for m in manifests:
            case = m["case"]
            metrics = m["metrics"]
            cache = metrics.get("cache_nc_wos_mean", {})
            print(f"{case['estimates_csv']} -> {case['output_dir']} cache_rmse={cache.get('rmse_vs_reference')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
