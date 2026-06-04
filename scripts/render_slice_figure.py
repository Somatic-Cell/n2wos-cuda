#!/usr/bin/env python3
"""Render publication-oriented slice figures for n2wos-cuda outputs.

This script has two separate responsibilities:

1. Convert per-pixel scalar values from an eval_tcnn_nc_wos slice run into an
   RGBA texture PNG.  It uses the slice mask written by the evaluator to map
   rows from *_estimates.csv back to the 2D image grid.
2. Render a clipped mesh with that texture placed on the slice plane.

It deliberately avoids photorealism.  The goal is deterministic raster figures
that can be arranged later in LaTeX, Illustrator, Inkscape, or PowerPoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def _require_matplotlib():
    try:
        from matplotlib import colormaps
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "matplotlib is required for texture/colorbar creation. "
            "Install with: python -m pip install matplotlib"
        ) from exc
    return colormaps


def _require_pyvista():
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "pyvista is required for 3D rendering. Install with: "
            "python -m pip install pyvista trimesh"
        ) from exc
    return pv


def _require_trimesh():
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "trimesh is required for loading OBJ/PLY/STL meshes. Install with: "
            "python -m pip install trimesh"
        ) from exc
    return trimesh


def resolve_path(path: Optional[str], base: Optional[Path] = None) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    if base is not None and (base / p).exists():
        return base / p
    return p


def normalize(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n <= 1.0e-15:
        raise ValueError(f"zero-length vector: {v}")
    return a / n


def view_basis(view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u_axis, v_axis, normal_axis) for a named slice view."""
    view = view.lower()
    if view == "xy":
        return (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    if view == "yz":
        return (
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )
    if view in {"zx", "xz"}:
        # The evaluator uses a two-letter view as the texture frame.  For zx,
        # texture u is z and texture v is x; the plane normal is y.
        return (
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
    raise ValueError(f"unsupported slice view: {view}; expected xy, yz, or zx")


def plane_origin(view: str, plane_value: float) -> np.ndarray:
    _, _, n = view_basis(view)
    return n * float(plane_value)


def load_json(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_from_json(data: dict, base_dir: Optional[Path]) -> dict:
    out: dict = {}
    slice_eval = data.get("slice_eval", {}) if isinstance(data, dict) else {}
    estimate_outputs = data.get("estimate_outputs", {}) if isinstance(data, dict) else {}
    options = data.get("options", {}) if isinstance(data, dict) else {}

    if slice_eval:
        out["width"] = slice_eval.get("width")
        out["height"] = slice_eval.get("height")
        out["view"] = slice_eval.get("view")
        out["plane"] = slice_eval.get("plane")
        out["frame_u_min"] = slice_eval.get("frame_u_min")
        out["frame_u_max"] = slice_eval.get("frame_u_max")
        out["frame_v_min"] = slice_eval.get("frame_v_min")
        out["frame_v_max"] = slice_eval.get("frame_v_max")
        out["mask"] = resolve_path(slice_eval.get("mask_ppm"), base_dir)
        out["points_csv"] = resolve_path(slice_eval.get("points_csv"), base_dir)

    if estimate_outputs:
        out["estimates_csv"] = resolve_path(estimate_outputs.get("estimates_csv"), base_dir)

    if options:
        out["mesh_mode"] = options.get("mesh")
        out["mesh_path"] = resolve_path(options.get("mesh_path"), base_dir)

    mesh_stats = data.get("mesh_stats", {}) if isinstance(data, dict) else {}
    norm = mesh_stats.get("normalization", {}) if isinstance(mesh_stats, dict) else {}
    if isinstance(norm, dict):
        center = norm.get("center")
        scale = norm.get("scale")
        if center is not None and scale is not None:
            out["mesh_normalization_center"] = center
            out["mesh_normalization_scale"] = scale

    return out


def read_csv_values(
    estimates_csv: Path,
    value_column: str,
    subtract_column: Optional[str] = None,
    abs_value: bool = False,
) -> np.ndarray:
    values = []
    with estimates_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {estimates_csv}")
        if value_column not in reader.fieldnames:
            raise ValueError(
                f"column {value_column!r} not found in {estimates_csv}; "
                f"available columns: {', '.join(reader.fieldnames)}"
            )
        if subtract_column and subtract_column not in reader.fieldnames:
            raise ValueError(
                f"subtract column {subtract_column!r} not found in {estimates_csv}; "
                f"available columns: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            v = float(row[value_column])
            if subtract_column:
                v -= float(row[subtract_column])
            if abs_value:
                v = abs(v)
            values.append(v)
    return np.asarray(values, dtype=np.float64)


def mask_inside(mask_image: Path) -> np.ndarray:
    img = Image.open(mask_image).convert("L")
    arr = np.asarray(img)
    # Treat very dark pixels as outside.  Some evaluator masks are written as
    # visualization images whose outside pixels are not exactly zero, so this
    # mask is only a first guess.  If its inside count does not match the
    # estimates CSV, the caller should reconstruct the mask from points_csv.
    return arr > 127


def _first_existing(row: dict, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return None


def _parse_bool_like(text: str) -> bool:
    s = str(text).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "inside", "valid"}


def _round_to_pixel(x: float, n: int) -> int:
    return max(0, min(n - 1, int(round(x))))


def _pixel_from_row_indices(row: dict, width: int, height: int) -> Optional[Tuple[int, int]]:
    x_text = _first_existing(
        row,
        [
            "pixel_x",
            "px",
            "ix",
            "i",
            "col",
            "column",
            "image_x",
            "x_index",
            "u_index",
        ],
    )
    y_text = _first_existing(
        row,
        [
            "pixel_y",
            "py",
            "iy",
            "j",
            "row",
            "image_y",
            "y_index",
            "v_index",
        ],
    )
    if x_text is None or y_text is None:
        return None
    try:
        x = _round_to_pixel(float(x_text), width)
        y = _round_to_pixel(float(y_text), height)
    except ValueError:
        return None
    return x, y


def _pixel_from_row_position(
    row: dict,
    width: int,
    height: int,
    view: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> Optional[Tuple[int, int]]:
    coord_names = {
        "x": ["x", "pos_x", "world_x", "px_world"],
        "y": ["y", "pos_y", "world_y", "py_world"],
        "z": ["z", "pos_z", "world_z", "pz_world"],
    }
    vals = {}
    for axis, names in coord_names.items():
        text = _first_existing(row, names)
        if text is None:
            return None
        try:
            vals[axis] = float(text)
        except ValueError:
            return None

    view = view.lower()
    if view == "xy":
        u, v = vals["x"], vals["y"]
    elif view == "yz":
        u, v = vals["y"], vals["z"]
    elif view in {"zx", "xz"}:
        u, v = vals["z"], vals["x"]
    else:
        return None

    if abs(u_max - u_min) <= 1.0e-15 or abs(v_max - v_min) <= 1.0e-15:
        return None
    x = _round_to_pixel((u - u_min) / (u_max - u_min) * (width - 1), width)
    # Image row 0 is the top.  Slice CSV points are usually generated in
    # bottom-to-top mathematical coordinates, so invert v for image y.
    y = _round_to_pixel((v_max - v) / (v_max - v_min) * (height - 1), height)
    return x, y


def reconstruct_pixels_from_points_csv(
    points_csv: Path,
    expected_rows: int,
    width: int,
    height: int,
    view: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> Tuple[np.ndarray, list[Tuple[int, int]]]:
    """Return an inside mask and per-estimate pixel coordinates in CSV row order.

    The estimates CSV and points CSV are produced from the same slice point list.
    Their rows are paired.  Therefore values must be scattered to the image using
    the point CSV order; assigning values to a boolean mask in row-major order
    creates stripe-like textures whenever the valid slice points are not written
    in ordinary image scanline order.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid slice dimensions from JSON/arguments: {width}x{height}")

    with points_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty points CSV: {points_csv}")
        rows = list(reader)

    inside_key = None
    for key in ["inside", "valid", "is_inside", "mask"]:
        if key in reader.fieldnames:
            inside_key = key
            break
    if inside_key is not None:
        candidate_rows = [r for r in rows if _parse_bool_like(r.get(inside_key, ""))]
    else:
        candidate_rows = rows

    if len(candidate_rows) != expected_rows and len(rows) == expected_rows:
        # Some points CSVs already contain only valid slice points and have no
        # explicit inside column.
        candidate_rows = rows

    if len(candidate_rows) != expected_rows:
        raise ValueError(
            f"points CSV row count does not match estimates CSV: "
            f"points={len(candidate_rows)} all_points={len(rows)} estimates={expected_rows}. "
            f"CSV={points_csv}"
        )

    inside = np.zeros((height, width), dtype=bool)
    pixels: list[Tuple[int, int]] = []
    for row in candidate_rows:
        pix = _pixel_from_row_indices(row, width, height)
        if pix is None:
            pix = _pixel_from_row_position(row, width, height, view, u_min, u_max, v_min, v_max)
        if pix is None:
            raise ValueError(
                f"cannot infer pixel coordinates from points CSV {points_csv}. "
                f"Add pixel_x/pixel_y columns or pass a mask image whose inside count matches the estimates CSV. "
                f"Available columns: {', '.join(reader.fieldnames or [])}"
            )
        x, y = pix
        pixels.append((x, y))
        inside[y, x] = True

    n_inside = int(np.count_nonzero(inside))
    if n_inside != expected_rows:
        raise ValueError(
            f"reconstructed point mask has {n_inside} unique pixels, but estimates CSV has {expected_rows} rows. "
            f"This usually means duplicated or incorrectly rounded pixel coordinates in {points_csv}."
        )
    return inside, pixels


def robust_range(values: np.ndarray, percentile: Optional[float]) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values for range computation")
    if percentile is None:
        return float(np.min(finite)), float(np.max(finite))
    p = float(percentile)
    if not (0.0 < p <= 100.0):
        raise ValueError("--robust-percentile must be in (0, 100]")
    lo = np.percentile(finite, 100.0 - p)
    hi = np.percentile(finite, p)
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def compute_color_range(
    values: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
    symmetric: bool,
    robust_percentile: Optional[float],
) -> Tuple[float, float]:
    lo, hi = robust_range(values, robust_percentile)
    if vmin is not None:
        lo = float(vmin)
    if vmax is not None:
        hi = float(vmax)
    if symmetric:
        m = max(abs(lo), abs(hi), 1.0e-12)
        lo, hi = -m, m
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"non-finite color range: {lo}, {hi}")
    if abs(hi - lo) <= 1.0e-15:
        eps = max(abs(hi), 1.0) * 1.0e-6
        lo -= eps
        hi += eps
    return lo, hi


def values_to_rgba(values_img: np.ndarray, inside: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> Image.Image:
    colormaps = _require_matplotlib()
    cmap = colormaps[cmap_name]
    t = (values_img - float(vmin)) / (float(vmax) - float(vmin))
    t = np.clip(t, 0.0, 1.0)
    rgba = cmap(t, bytes=True)
    rgba[..., 3] = np.where(inside, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def make_texture_from_csv(
    estimates_csv: Path,
    mask_image: Optional[Path],
    points_csv: Optional[Path],
    width: Optional[int],
    height: Optional[int],
    view: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    value_column: str,
    subtract_column: Optional[str],
    abs_value: bool,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    symmetric: bool,
    robust_percentile: Optional[float],
    output: Path,
    meta_output: Optional[Path],
) -> Tuple[Path, float, float, int]:
    values = read_csv_values(estimates_csv, value_column, subtract_column, abs_value)

    inside: Optional[np.ndarray] = None
    pixels: Optional[list[Tuple[int, int]]] = None
    mask_warning: Optional[str] = None

    # Prefer points_csv when available because it preserves the exact row pairing
    # between *_points.csv and *_estimates.csv.  A boolean mask only tells where
    # valid pixels are; it does not encode the order in which CSV rows were
    # written.
    if points_csv is not None:
        if width is None or height is None:
            if mask_image is not None:
                mask_img = Image.open(mask_image)
                width, height = mask_img.size
            else:
                raise ValueError("slice width/height are required when reconstructing texture pixels from points CSV")
        inside, pixels = reconstruct_pixels_from_points_csv(
            points_csv=points_csv,
            expected_rows=int(values.size),
            width=int(width),
            height=int(height),
            view=view,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
        )

        if mask_image is not None:
            mask_guess = mask_inside(mask_image)
            n_mask_inside = int(np.count_nonzero(mask_guess))
            if n_mask_inside != int(values.size):
                mask_warning = (
                    f"mask inside-pixel count ({n_mask_inside}) does not match estimates CSV rows "
                    f"({values.size}); using points CSV row order for texture placement"
                )
                print(f"[render_slice_figure] warning: {mask_warning}", file=sys.stderr)
    elif mask_image is not None:
        inside = mask_inside(mask_image)
        n_mask_inside = int(np.count_nonzero(inside))
        if n_mask_inside != int(values.size):
            raise ValueError(
                f"mask inside-pixel count ({n_mask_inside}) does not match estimates CSV rows "
                f"({values.size}), and no points CSV was available. Pass --points-csv or use --json "
                "from the same slice run so row order can be reconstructed."
            )
    else:
        raise ValueError(
            "cannot map estimates CSV rows to image pixels. Pass --points-csv, --mask-image, "
            "or use --json from the same slice run."
        )

    n_inside = int(np.count_nonzero(inside))
    if n_inside != int(values.size):
        raise ValueError(
            f"inside-pixel count ({n_inside}) does not match number of CSV rows ({values.size}). "
            "Check that JSON, mask/points CSV, and estimates CSV come from the same slice run."
        )

    img_values = np.full(inside.shape, np.nan, dtype=np.float64)
    if pixels is not None:
        for (x, y), value in zip(pixels, values):
            img_values[y, x] = value
    else:
        # Last-resort fallback for older outputs that only have a true binary
        # mask.  This assumes the estimates CSV was written in normal image
        # row-major order.
        img_values[inside] = values
    lo, hi = compute_color_range(values, vmin, vmax, symmetric, robust_percentile)
    tex = values_to_rgba(img_values, inside, cmap, lo, hi)
    output.parent.mkdir(parents=True, exist_ok=True)
    tex.save(output)

    if meta_output is not None:
        meta_output.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "texture_png": str(output),
            "estimates_csv": str(estimates_csv),
            "mask_image": str(mask_image) if mask_image is not None else None,
            "points_csv": str(points_csv) if points_csv is not None else None,
            "value_column": value_column,
            "subtract_column": subtract_column,
            "abs_value": bool(abs_value),
            "colormap": cmap,
            "vmin": lo,
            "vmax": hi,
            "inside_pixels": n_inside,
            "width": int(inside.shape[1]),
            "height": int(inside.shape[0]),
        }
        meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return output, lo, hi, n_inside


def save_colorbar(output: Path, meta_output: Optional[Path], cmap_name: str, vmin: float, vmax: float, width: int, height: int, vertical: bool) -> None:
    colormaps = _require_matplotlib()
    cmap = colormaps[cmap_name]
    if vertical:
        t = np.linspace(1.0, 0.0, height, dtype=np.float64)[:, None]
        t = np.repeat(t, width, axis=1)
    else:
        t = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :]
        t = np.repeat(t, height, axis=0)
    rgba = cmap(t, bytes=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(output)
    if meta_output is not None:
        meta_output.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "colorbar_png": str(output),
            "colormap": cmap_name,
            "vmin": float(vmin),
            "vmax": float(vmax),
            "orientation": "vertical" if vertical else "horizontal",
            "width": int(width),
            "height": int(height),
        }
        meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def generate_procedural_bumpy_sphere(stacks: int = 128, slices: int = 256, amplitude: float = 0.15):
    pv = _require_pyvista()
    # The exact C++ procedural mesh may differ in phase, but this is adequate for
    # paper-layout mockups and figure composition.  For exact geometry, pass the
    # exported OBJ/PLY mesh via --mesh.
    sphere = pv.Sphere(radius=1.0, theta_resolution=slices, phi_resolution=stacks)
    pts = np.asarray(sphere.points, dtype=np.float64)
    dirs = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1.0e-12)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    bump = 1.0 + amplitude * (0.35 * np.sin(5.0 * x + 2.0 * y) + 0.25 * np.cos(7.0 * y - 3.0 * z) + 0.20 * np.sin(9.0 * z + x))
    sphere.points = (dirs * bump[:, None]).astype(np.float32)
    return sphere.triangulate().clean()


def parse_center_triplet(text: Optional[str]) -> Optional[np.ndarray]:
    if text is None or text == "":
        return None
    parts = [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"expected three center coordinates, got {text!r}")
    return np.asarray([float(x) for x in parts], dtype=np.float64)


def load_mesh(
    mesh_arg: Optional[str],
    normalize_unit_radius: bool,
    solver_normalization_center: Optional[Sequence[float]] = None,
    solver_normalization_scale: Optional[float] = None,
):
    if mesh_arg is None or mesh_arg == "none":
        return None
    pv = _require_pyvista()
    if mesh_arg == "procedural_bumpy_sphere":
        mesh = generate_procedural_bumpy_sphere()
    else:
        path = Path(mesh_arg)
        if not path.exists():
            raise FileNotFoundError(path)
        # PyVista directly handles OBJ/PLY/STL; if it fails, fallback to trimesh.
        try:
            mesh = pv.read(path)
            mesh = mesh.extract_surface().triangulate().clean()
        except Exception:
            trimesh = _require_trimesh()
            tri = trimesh.load(path, force="mesh")
            mesh = pv.wrap(tri).extract_surface().triangulate().clean()

    if solver_normalization_center is not None and solver_normalization_scale is not None:
        center = np.asarray(solver_normalization_center, dtype=np.float64)
        if center.shape != (3,):
            raise ValueError(f"solver normalization center must have 3 components, got {center}")
        scale = float(solver_normalization_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid solver normalization scale: {solver_normalization_scale}")
        pts = np.asarray(mesh.points, dtype=np.float64)
        mesh.points = ((pts - center[None, :]) * scale).astype(np.float32)
    elif normalize_unit_radius:
        pts = np.asarray(mesh.points, dtype=np.float64)
        center = 0.5 * (np.min(pts, axis=0) + np.max(pts, axis=0))
        scale = np.max(np.linalg.norm(pts - center[None, :], axis=1))
        if scale <= 0:
            raise ValueError("cannot normalize zero-size mesh")
        mesh.points = ((pts - center[None, :]) / scale).astype(np.float32)
    return mesh

def make_plane_mesh(view: str, plane: float, u_min: float, u_max: float, v_min: float, v_max: float):
    pv = _require_pyvista()
    u_axis, v_axis, n_axis = view_basis(view)
    origin = plane_origin(view, plane)
    p00 = origin + u_axis * u_min + v_axis * v_min
    p10 = origin + u_axis * u_max + v_axis * v_min
    p11 = origin + u_axis * u_max + v_axis * v_max
    p01 = origin + u_axis * u_min + v_axis * v_max
    points = np.asarray([p00, p10, p11, p01], dtype=np.float32)
    faces = np.asarray([4, 0, 1, 2, 3], dtype=np.int64)
    plane_mesh = pv.PolyData(points, faces)
    plane_mesh.active_texture_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    return plane_mesh, origin, u_axis, v_axis, n_axis


def render_scene(
    mesh_arg: Optional[str],
    texture_png: Optional[Path],
    output: Path,
    view: str,
    plane: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    normalize_unit_radius: bool,
    solver_normalization_center: Optional[Sequence[float]],
    solver_normalization_scale: Optional[float],
    mesh_color: str,
    mesh_opacity: float,
    clip_front_half: bool,
    no_clip: bool,
    show_cut: bool,
    show_edges: bool,
    window_size: Tuple[int, int],
    background: str,
    camera_distance_scale: float,
    camera_oblique_u: float,
    camera_oblique_v: float,
    parallel_projection: bool,
    camera_view_angle: float,
) -> None:
    pv = _require_pyvista()
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    mesh = load_mesh(
        mesh_arg,
        normalize_unit_radius=normalize_unit_radius,
        solver_normalization_center=solver_normalization_center,
        solver_normalization_scale=solver_normalization_scale,
    )
    plane_mesh, origin, u_axis, v_axis, n_axis = make_plane_mesh(view, plane, u_min, u_max, v_min, v_max)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)

    if mesh is not None:
        if no_clip:
            clipped = mesh
        else:
            # For slice figures, the desired default is to keep the half behind the
            # textured slice plane so the slice texture sits in front and the mesh is
            # visible behind it.  Users can override this with --clip-front-half.
            keep_back_half = texture_png is not None and not clip_front_half
            clip_invert = keep_back_half
            try:
                clipped = mesh.clip(normal=n_axis, origin=origin, invert=clip_invert)
            except Exception:
                clipped = mesh
        plotter.add_mesh(
            clipped,
            color=mesh_color,
            opacity=float(mesh_opacity),
            smooth_shading=True,
            show_edges=show_edges,
            show_scalar_bar=False,
        )
        if show_cut and not no_clip:
            section = mesh.slice(normal=n_axis, origin=origin)
            if section.n_points > 0:
                plotter.add_mesh(section, color="black", line_width=2, show_scalar_bar=False)

    if texture_png is not None:
        rgba = Image.open(texture_png).convert("RGBA")
        tex = pv.Texture(np.asarray(rgba))
        plotter.add_mesh(plane_mesh, texture=tex, smooth_shading=False, show_scalar_bar=False)

    # Camera is mostly normal-facing, with a small oblique component for depth.
    if mesh is not None:
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        diag = float(np.linalg.norm(bounds[1::2] - bounds[::2]))
    else:
        diag = float(max(abs(u_max - u_min), abs(v_max - v_min), 1.0))
    view_dir = normalize(-n_axis + camera_oblique_u * u_axis + camera_oblique_v * v_axis)
    focal = origin + 0.5 * (u_min + u_max) * u_axis + 0.5 * (v_min + v_max) * v_axis
    cam_pos = focal - float(camera_distance_scale) * diag * view_dir
    plotter.camera_position = [tuple(cam_pos), tuple(focal), tuple(v_axis)]
    if parallel_projection:
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 0.62 * max(abs(u_max - u_min), abs(v_max - v_min))
    else:
        plotter.camera.parallel_projection = False
        plotter.camera.view_angle = float(camera_view_angle)

    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(output), auto_close=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--json", type=Path, help="eval_tcnn_nc_wos JSON from a slice run; used to infer mask/CSV/frame metadata")
    p.add_argument("--mesh", help="mesh path, 'procedural_bumpy_sphere', or 'none'. Overrides JSON mesh_path")
    p.add_argument("--normalize-unit-radius", action="store_true", help="apply visual unit-radius normalization to an input mesh when JSON solver normalization is not used")
    p.add_argument("--no-json-mesh-normalization", action="store_true", help="do not apply mesh_stats.normalization from --json to the rendered mesh")
    p.add_argument("--mesh-normalization-center", help="explicit raw-space solver normalization center as 'cx,cy,cz' or 'cx cy cz'")
    p.add_argument("--mesh-normalization-scale", type=float, help="explicit raw-space solver normalization scale; rendered point is (p-center)*scale")

    p.add_argument("--slice-image", type=Path, help="existing RGBA/RGB image to place on the slice plane")
    p.add_argument("--mask-image", type=Path, help="slice mask PPM/PNG; used when creating a texture from CSV")
    p.add_argument("--points-csv", type=Path, help="slice points CSV; used to reconstruct the mask if the mask image is a visualization mask")
    p.add_argument("--estimates-csv", type=Path, help="*_estimates.csv written by --save-estimates-prefix")
    p.add_argument("--slice-width", type=int, help="slice image width; inferred from JSON or mask image when omitted")
    p.add_argument("--slice-height", type=int, help="slice image height; inferred from JSON or mask image when omitted")
    p.add_argument("--value-column", default="nc_wos_mean", help="CSV column to visualize")
    p.add_argument("--subtract-column", help="optional CSV column to subtract, e.g. analytic_value for signed error")
    p.add_argument("--abs-value", action="store_true", help="visualize abs(value_column - subtract_column)")
    p.add_argument("--texture-output", type=Path, help="write generated slice texture PNG")
    p.add_argument("--texture-meta", type=Path, help="write texture metadata JSON")

    p.add_argument("--view", choices=["xy", "yz", "zx", "xz"], help="slice view; inferred from JSON when omitted")
    p.add_argument("--plane", type=float, help="slice plane coordinate; inferred from JSON when omitted")
    p.add_argument("--frame-u-min", type=float, help="texture-frame u min; inferred from JSON when omitted")
    p.add_argument("--frame-u-max", type=float, help="texture-frame u max; inferred from JSON when omitted")
    p.add_argument("--frame-v-min", type=float, help="texture-frame v min; inferred from JSON when omitted")
    p.add_argument("--frame-v-max", type=float, help="texture-frame v max; inferred from JSON when omitted")

    p.add_argument("--cmap", default="coolwarm", help="matplotlib colormap for generated texture/colorbar")
    p.add_argument("--vmin", type=float)
    p.add_argument("--vmax", type=float)
    p.add_argument("--symmetric-range", action="store_true", help="force color range to [-max_abs, max_abs]")
    p.add_argument("--robust-percentile", type=float, help="use percentile range before optional symmetry, e.g. 99")

    p.add_argument("--colorbar-output", type=Path, help="write standalone colorbar PNG")
    p.add_argument("--colorbar-meta", type=Path, help="write colorbar metadata JSON containing vmin/vmax")
    p.add_argument("--colorbar-width", type=int, default=512)
    p.add_argument("--colorbar-height", type=int, default=48)
    p.add_argument("--colorbar-vertical", action="store_true")

    p.add_argument("--render-output", type=Path, help="write mesh/slice render PNG")
    p.add_argument("--mesh-only", action="store_true", help="render mesh without adding the slice texture")
    p.add_argument("--no-clip", action="store_true", help="render the complete mesh without applying the slice-plane clip")
    p.add_argument("--mesh-color", default="#bdbdbd")
    p.add_argument("--mesh-opacity", type=float, default=1.0)
    p.add_argument("--clip-front-half", action="store_true", help="keep the camera-facing half after clipping; by default, when a slice texture is shown, keep the opposite half so the textured slice is visible in front of the mesh")
    p.add_argument("--show-cut", action="store_true", help="draw the intersection curve")
    p.add_argument("--show-edges", action="store_true")
    p.add_argument("--window-width", type=int, default=1400)
    p.add_argument("--window-height", type=int, default=1000)
    p.add_argument("--background", default="white")
    p.add_argument("--camera-distance-scale", type=float, default=1.55)
    p.add_argument("--camera-oblique-u", type=float, default=0.75, help="horizontal oblique component; larger values give a more side-angled view")
    p.add_argument("--camera-oblique-v", type=float, default=0.08, help="vertical oblique component")
    p.add_argument("--parallel-projection", action="store_true", help="use orthographic projection instead of the default perspective view")
    p.add_argument("--camera-view-angle", type=float, default=24.0, help="perspective field of view in degrees when --parallel-projection is not set")
    return p.parse_args()


def require_value(name: str, value):
    if value is None:
        raise ValueError(f"{name} is required; pass it explicitly or use --json from a slice run")
    return value


def main() -> None:
    args = parse_args()
    json_base = args.json.parent if args.json else None
    data = load_json(args.json)
    inferred = infer_from_json(data, json_base)

    view = args.view or inferred.get("view") or "xy"
    plane = float(args.plane if args.plane is not None else inferred.get("plane", 0.0))
    u_min = float(require_value("--frame-u-min", args.frame_u_min if args.frame_u_min is not None else inferred.get("frame_u_min")))
    u_max = float(require_value("--frame-u-max", args.frame_u_max if args.frame_u_max is not None else inferred.get("frame_u_max")))
    v_min = float(require_value("--frame-v-min", args.frame_v_min if args.frame_v_min is not None else inferred.get("frame_v_min")))
    v_max = float(require_value("--frame-v-max", args.frame_v_max if args.frame_v_max is not None else inferred.get("frame_v_max")))

    mask_image = args.mask_image or inferred.get("mask")
    points_csv = args.points_csv or inferred.get("points_csv")
    estimates_csv = args.estimates_csv or inferred.get("estimates_csv")
    slice_width = args.slice_width if args.slice_width is not None else inferred.get("width")
    slice_height = args.slice_height if args.slice_height is not None else inferred.get("height")
    texture_png = args.slice_image
    used_vmin = args.vmin
    used_vmax = args.vmax

    if texture_png is None and estimates_csv is not None:
        texture_output = args.texture_output or Path(str(estimates_csv).replace("_estimates.csv", f"_{args.value_column}.png"))
        texture_png, used_vmin, used_vmax, _ = make_texture_from_csv(
            estimates_csv=Path(estimates_csv),
            mask_image=Path(mask_image) if mask_image is not None else None,
            points_csv=Path(points_csv) if points_csv is not None else None,
            width=int(slice_width) if slice_width is not None else None,
            height=int(slice_height) if slice_height is not None else None,
            view=view,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
            value_column=args.value_column,
            subtract_column=args.subtract_column,
            abs_value=args.abs_value,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            symmetric=args.symmetric_range,
            robust_percentile=args.robust_percentile,
            output=Path(texture_output),
            meta_output=args.texture_meta,
        )

    if args.colorbar_output is not None:
        if used_vmin is None or used_vmax is None:
            if estimates_csv is None:
                raise ValueError("cannot infer colorbar range without CSV values; pass --vmin and --vmax")
            vals = read_csv_values(Path(estimates_csv), args.value_column, args.subtract_column, args.abs_value)
            used_vmin, used_vmax = compute_color_range(vals, args.vmin, args.vmax, args.symmetric_range, args.robust_percentile)
        save_colorbar(
            output=args.colorbar_output,
            meta_output=args.colorbar_meta,
            cmap_name=args.cmap,
            vmin=float(used_vmin),
            vmax=float(used_vmax),
            width=args.colorbar_width,
            height=args.colorbar_height,
            vertical=args.colorbar_vertical,
        )

    if args.render_output is not None:
        mesh_arg = args.mesh
        if mesh_arg is None:
            mesh_path = inferred.get("mesh_path")
            mesh_mode = inferred.get("mesh_mode")
            if mesh_path:
                mesh_arg = str(mesh_path)
            elif mesh_mode:
                mesh_arg = str(mesh_mode)

        solver_center = parse_center_triplet(args.mesh_normalization_center)
        solver_scale = args.mesh_normalization_scale
        if not args.no_json_mesh_normalization:
            if solver_center is None and inferred.get("mesh_normalization_center") is not None:
                solver_center = np.asarray(inferred.get("mesh_normalization_center"), dtype=np.float64)
            if solver_scale is None and inferred.get("mesh_normalization_scale") is not None:
                solver_scale = float(inferred.get("mesh_normalization_scale"))
        if (solver_center is None) != (solver_scale is None):
            raise ValueError(
                "mesh normalization requires both center and scale. "
                "Use --mesh-normalization-center and --mesh-normalization-scale, "
                "or provide --json with mesh_stats.normalization."
            )

        render_scene(
            mesh_arg=mesh_arg,
            texture_png=None if args.mesh_only else (Path(texture_png) if texture_png is not None else None),
            output=args.render_output,
            view=view,
            plane=plane,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
            normalize_unit_radius=args.normalize_unit_radius,
            solver_normalization_center=solver_center,
            solver_normalization_scale=solver_scale,
            mesh_color=args.mesh_color,
            mesh_opacity=args.mesh_opacity,
            clip_front_half=args.clip_front_half,
            no_clip=args.no_clip,
            show_cut=args.show_cut,
            show_edges=args.show_edges,
            window_size=(args.window_width, args.window_height),
            background=args.background,
            camera_distance_scale=args.camera_distance_scale,
            camera_oblique_u=args.camera_oblique_u,
            camera_oblique_v=args.camera_oblique_v,
            parallel_projection=args.parallel_projection,
            camera_view_angle=args.camera_view_angle,
        )

    if texture_png is None and args.colorbar_output is None and args.render_output is None:
        raise ValueError("nothing to do: pass --texture-output, --colorbar-output, or --render-output")


if __name__ == "__main__":
    main()
