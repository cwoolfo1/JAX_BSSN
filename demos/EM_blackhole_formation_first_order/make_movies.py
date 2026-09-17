"""Make x-z MP4 movies from the first-order EM-collapse snapshots.

Run with the same Python environment used for the demo, plus ffmpeg:
    python make_movies.py
    python make_movies.py --input output --fps 5 --output-dir output/movies

Seven movies cover all BSSN scalars, vectors and independent tensor components.
The eighth plots saved rho_EM on a logarithmic scale, formed from native
quadratic Yee moments.
For legacy files only, it approximates the density from averaged physical D/B:
    rho_EM = (gamma_ij D^i D^j + gamma_ij B^i B^j) / 2,
    gamma_ij = conformal_metric_ij / max(W, 1e-12)^2.
This is the matter density, not an ADM mass or a local gravitational energy.
Color limits are fixed across the selected snapshots. Each snapshot occupies
one video frame; titles display its actual simulation time. No JAX is needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from JAX_BSSN.diagnostics.mesh_coordinates import mesh_axis_coordinates

import openpmd_api as io

DEFAULT_INPUT = Path(__file__).resolve().parent / "output_uniform"
TENSOR_COMPONENTS = ("xx", "xy", "xz", "yy", "yz", "zz")
W_FLOOR = 1.0e-12  # Matches JAX_BSSN.bssn.geometry.W_FLOOR_VALUE.


@dataclass(frozen=True)
class Movie:
    name: str
    title: str
    fields: tuple[tuple[str, str | None, str], ...]


MOVIES = (
    Movie("conformal_factor_W", "Conformal factor", (("W", None, r"$W$"),)),
    Movie("lapse", "Lapse", (("lapse", None, r"$\alpha$"),)),
    Movie("trace_K", "Trace of extrinsic curvature", (("K", None, r"$K$"),)),
    Movie("shift", "Shift", tuple(("shift", c, rf"$\beta^{c}$") for c in "xyz")),
    Movie("conformal_connection", "Conformal connection", tuple(
        ("conformal_connection", c, rf"$\widetilde{{\Gamma}}^{c}$") for c in "xyz")),
    Movie("conformal_metric", "Conformal spatial metric", tuple(
        (f"conformal_metric_{c}", None, rf"$\widetilde{{\gamma}}_{{{c}}}$")
        for c in TENSOR_COMPONENTS)),
    Movie("traceless_K", "Conformal traceless extrinsic curvature", tuple(
        (f"traceless_K_{c}", None, rf"$\widetilde{{A}}_{{{c}}}$")
        for c in TENSOR_COMPONENTS)),
    Movie("mass_energy_density", "Electromagnetic mass-energy density", ()),
)


def energy_density(W, metric, D, B):
    """Contract physical contravariant fields with the physical spatial metric."""
    return 0.5 * (
        np.einsum("ij...,i...,j...->...", metric, D, D)
        + np.einsum("ij...,i...,j...->...", metric, B, B)
    ) / np.maximum(W, W_FLOOR)**2


def input_paths(source, _seen=None):
    source = Path(source).resolve()
    root_call = _seen is None
    seen = set() if _seen is None else _seen
    if source in seen:
        raise ValueError('Cycle in movie restart history')
    seen.add(source)
    if source.is_file():
        return [source]
    summary = source / "run_summary.json"
    if summary.is_file():
        meta = json.loads(summary.read_text())
        paths = []
        if meta.get('parent_run'):
            parent = Path(meta['parent_run'])
            paths.extend(input_paths(parent if parent.is_absolute() else source/parent, seen))
        for name in meta['output_segments']:
            path = Path(name)
            path = path if path.is_absolute() else source/path
            # Existing demos may record a cwd-relative name. The basename also
            # permits relocation of a self-contained output directory.
            if not path.is_file():
                path = source/Path(name).name
            if path not in paths:
                paths.append(path)
    else:
        paths = [source / "EM_blackhole_formation.h5"]
    # An initial-data or recovery segment may deliberately have no snapshots.
    # Its child can still provide a complete movie catalog. Declared files must
    # exist, and a top-level request with no snapshots remains an error.
    if (root_call and not paths) or any(not p.is_file() for p in paths):
        raise FileNotFoundError(f"Cannot locate snapshot files in {source}")
    return paths


def catalog(paths, max_frames):
    # Later restart segments replace duplicate steps at a restart boundary.
    snapshots = {}
    for path in paths:
        series = io.Series(str(path), io.Access.read_only)
        try:
            for step in series.iterations:
                iteration = series.iterations[step]
                time = float(iteration.time)
                if not np.isfinite(time):
                    raise ValueError(f"Nonfinite time in {path}, step {step}")
                snapshots[int(step)] = (path, int(step), time)
                iteration.close()
        finally:
            series.close()
    frames = sorted(snapshots.values(), key=lambda frame: (frame[2], frame[1]))
    if not frames:
        raise ValueError("No snapshots found")
    return frames[:max_frames]


def read_fields(series, iteration, fields):
    pending = []
    geometry = None
    for record, component, _ in fields:
        mesh = iteration.meshes[record]
        data = mesh[component] if component is not None else mesh[next(iter(mesh))]
        if len(data.shape) != 3 or data.shape[1] != 1:
            raise ValueError(f"{record}: expected a singleton-y x-z plane")
        if list(mesh.axis_labels) != ["x", "y", "z"]:
            raise ValueError(f"{record}: expected x,y,z axis order")
        current = (tuple(mesh_axis_coordinates(mesh, data, 0, edges=True)),
                   tuple(mesh_axis_coordinates(mesh, data, 2, edges=True)))
        if geometry is not None and current != geometry:
            raise ValueError("Saved fields are not collocated on a common grid")
        geometry = current
        pending.append(data.load_chunk())
    series.flush()
    return [np.asarray(data)[:, 0, :].copy() for data in pending], geometry


def load_movie(movie, frames):
    fields = movie.fields
    if not fields:
        fields = (("W", None, ""),) + tuple(
            (f"conformal_metric_{c}", None, "") for c in TENSOR_COMPONENTS
        ) + tuple((record, c, "") for record in ("D", "B") for c in "xyz")
    values, geometry = [], None
    series, path_open = None, None
    try:
        for path, step, _ in frames:
            if path != path_open:
                if series is not None:
                    series.close()
                series = io.Series(str(path), io.Access.read_only)
                path_open = path
            iteration = series.iterations[step]
            saved_density = not movie.fields and "rho_EM" in iteration.meshes
            selected_fields = (("rho_EM", None, ""),) if saved_density else fields
            arrays, current = read_fields(series, iteration, selected_fields)
            if geometry is not None and current != geometry:
                raise ValueError("Grid geometry changes between snapshots")
            geometry = current
            if not movie.fields and not saved_density:
                metric = np.empty((3, 3) + arrays[0].shape)
                for component, array in zip(TENSOR_COMPONENTS, arrays[1:7]):
                    i, j = ("xyz".index(c) for c in component)
                    metric[i, j] = metric[j, i] = array
                rho = energy_density(arrays[0], metric, np.stack(arrays[7:10]),
                                     np.stack(arrays[10:13]))
                arrays = [rho]
            values.append(np.stack(arrays))
            iteration.close()
    finally:
        if series is not None:
            series.close()
    return np.stack(values), geometry


def color_scale(values, logarithmic=False):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot color a field with no finite samples")
    if logarithmic:
        positive = finite[finite > 0]
        high = float(positive.max()) if positive.size else 1.0
        low = max(high * 1.0e-8, np.finfo(float).tiny)
        return LogNorm(low, max(high, low * 10)), "magma"
    low, high = float(finite.min()), float(finite.max())
    if low < 0 < high:
        bound = max(abs(low), abs(high))
        return Normalize(-bound, bound), "RdBu_r"
    if low == high:
        padding = max(abs(low), 1.0) * 0.01
        low, high = low - padding, high + padding
    return Normalize(low, high), "viridis"


def render_movie(movie, frames, output, fps, dpi, view_radius=None, run_label=None):
    values, (x_edges, z_edges) = load_movie(movie, frames)
    if view_radius is not None:
        x_edges, z_edges = np.asarray(x_edges), np.asarray(z_edges)
        xi = np.flatnonzero(np.abs(.5*(x_edges[:-1]+x_edges[1:])) <= view_radius)
        zi = np.flatnonzero(np.abs(.5*(z_edges[:-1]+z_edges[1:])) <= view_radius)
        if not xi.size or not zi.size:
            raise ValueError('view radius contains no grid cells')
        values = values[:,:,xi[0]:xi[-1]+1,zi[0]:zi[-1]+1]
        x_edges=x_edges[xi[0]:xi[-1]+2]
        z_edges=z_edges[zi[0]:zi[-1]+2]
    density = not movie.fields
    labels = [r"$\rho_{\rm EM}$ — logarithmic"] if density else [f[2] for f in movie.fields]
    columns = min(3, len(labels))
    rows = math.ceil(len(labels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5*columns, 3.6*rows),
                                squeeze=False, layout="constrained")
    images, scales = [], []
    for index, (axis, label) in enumerate(zip(axes.flat, labels)):
        component = 0 if density else index
        logarithmic = density
        norm, cmap = color_scale(values[:, component], logarithmic)
        plane = values[0, component].T
        masked = np.ma.masked_invalid(plane)
        if logarithmic:
            masked = np.ma.masked_less_equal(masked, 0)
        im = axis.pcolormesh(x_edges, z_edges, masked, shading="flat", norm=norm, cmap=cmap)
        axis.set_aspect("equal")
        axis.set(title=label, xlabel="x", ylabel="z")
        figure.colorbar(im, ax=axis, shrink=0.85)
        images.append(im)
        scales.append({"label": label, "minimum": norm.vmin, "maximum": norm.vmax,
                       "logarithmic": logarithmic})
    for axis in list(axes.flat)[len(labels):]:
        axis.set_visible(False)
    heading = figure.suptitle(movie.title)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=[
        "-pix_fmt", "yuv420p", "-crf", "18", "-threads", "2", "-movflags", "+faststart",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    ])
    try:
        with writer.saving(figure, str(output), dpi=dpi):
            for index, (_, step, time) in enumerate(frames):
                for panel, im in enumerate(images):
                    plane = np.ma.masked_invalid(values[index, 0 if density else panel].T)
                    if density:
                        plane = np.ma.masked_less_equal(plane, 0)
                    im.set_array(plane)
                bad = np.count_nonzero(~np.isfinite(values[index]))
                note = f" | {bad} nonfinite samples masked" if bad else ""
                annotation=f"\n{run_label}" if run_label else ""
                heading.set_text(f"{movie.title}{annotation}\nt = {time:g} | step {step}{note}")
                writer.grab_frame()
    finally:
        plt.close(figure)
    return {"file": output.name, "color_scales": scales,
            "nonfinite_samples": int(np.count_nonzero(~np.isfinite(values)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="run directory or openPMD HDF5 file")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--view-radius", type=float, help="crop to |x|,|z| <= this radius")
    parser.add_argument("--label", help="show a run/validation label on every frame")
    parser.add_argument("--max-frames", type=int, help="use only the first N snapshots")
    parser.add_argument("--only", nargs="+", choices=[m.name for m in MOVIES], help="render selected movies")
    parser.add_argument("--overwrite", action="store_true", help="replace existing movies")
    args = parser.parse_args()
    if args.view_radius is not None and (not math.isfinite(args.view_radius) or args.view_radius <= 0):
        parser.error('view-radius must be positive and finite')
    if args.fps < 1 or args.dpi < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("fps, dpi and max-frames must be positive")
    if not FFMpegWriter.isAvailable():
        parser.error("ffmpeg is required on PATH")
    paths = input_paths(args.input)
    frames = catalog(paths, args.max_frames)
    output = (args.output_dir or paths[0].parent / "movies").resolve()
    movies = [m for m in MOVIES if args.only is None or m.name in args.only]
    targets = [output / (m.name + ".mp4") for m in movies]
    manifest = output / "movies.json"
    if not args.overwrite and any(p.exists() for p in targets + [manifest]):
        parser.error("output already exists; choose another directory or use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for movie, target in zip(movies, targets):
        print(f"Rendering {movie.name}: {len(frames)} frames", flush=True)
        records.append(render_movie(movie, frames, target, args.fps, args.dpi, args.view_radius, args.label))
        print(f"Wrote {target}", flush=True)
    manifest.write_text(json.dumps({
        "inputs": [str(p) for p in paths], "fps": args.fps, "view_radius": args.view_radius,
        "label": args.label,
        "frames": [{"step": step, "time": time} for _, step, time in frames],
        "density_definition": "saved rho_EM (native quadratic moments); approximate averaged-field reconstruction for legacy snapshots without rho_EM",
        "movies": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
