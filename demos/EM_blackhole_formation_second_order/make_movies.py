"""Make x-z MP4 movies from the second-order EM-collapse snapshots.

Run with the same Python environment used for the demo, plus ffmpeg:
    python make_movies.py
    python make_movies.py --input output_uniform --fps 5 --output-dir output_uniform/movies

Seven movies cover all BSSN scalars, vectors and independent tensor components.
The eighth shows Eulerian electromagnetic mass-energy density, computed from
saved cell-centered physical covariant E and B (Lorentz--Heaviside units):
    rho_EM = (gamma^ij E_i E_j + gamma^ij B_i B_j) / 2,
    gamma^ij = max(W, 1e-12)^2 * inverse(conformal_metric)_ij.
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


def energy_density(W, metric, E, B):
    """Contract physical covectors with the inverse physical spatial metric."""
    inverse_metric = np.moveaxis(
        np.linalg.inv(np.moveaxis(metric, (0, 1), (-2, -1))),
        (-2, -1), (0, 1),
    )
    return 0.5 * np.maximum(W, W_FLOOR)**2 * (
        np.einsum("ij...,i...,j...->...", inverse_metric, E, E)
        + np.einsum("ij...,i...,j...->...", inverse_metric, B, B)
    )


def input_paths(source):
    source = Path(source).resolve()
    if source.is_file():
        return [source]
    summary = source / "run_summary.json"
    if summary.is_file():
        names = json.loads(summary.read_text())["output_segments"]
        paths = [source / Path(name).name for name in names]
    else:
        paths = [source / "EM_blackhole_formation.h5"]
    if not paths or any(not p.is_file() for p in paths):
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
        ) + tuple((record, c, "") for record in ("E", "B") for c in "xyz")
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
            arrays, current = read_fields(series, iteration, fields)
            if geometry is not None and current != geometry:
                raise ValueError("Grid geometry changes between snapshots")
            geometry = current
            if not movie.fields:
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


def render_movie(movie, frames, output, fps, dpi):
    values, (x_edges, z_edges) = load_movie(movie, frames)
    density = not movie.fields
    labels = [r"$\rho_{\rm EM}$ — linear", r"$\rho_{\rm EM}$ — logarithmic"] if density else [f[2] for f in movie.fields]
    columns = min(3, len(labels))
    rows = math.ceil(len(labels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5*columns, 3.6*rows),
                                squeeze=False, layout="constrained")
    images, scales = [], []
    for index, (axis, label) in enumerate(zip(axes.flat, labels)):
        component = 0 if density else index
        logarithmic = density and index == 1
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
                    if density and panel == 1:
                        plane = np.ma.masked_less_equal(plane, 0)
                    im.set_array(plane)
                bad = np.count_nonzero(~np.isfinite(values[index]))
                note = f" | {bad} nonfinite samples masked" if bad else ""
                heading.set_text(f"{movie.title}\nt = {time:g} | step {step}{note}")
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
    parser.add_argument("--max-frames", type=int, help="use only the first N snapshots")
    parser.add_argument("--only", nargs="+", choices=[m.name for m in MOVIES], help="render selected movies")
    parser.add_argument("--overwrite", action="store_true", help="replace existing movies")
    args = parser.parse_args()
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
        records.append(render_movie(movie, frames, target, args.fps, args.dpi))
        print(f"Wrote {target}", flush=True)
    manifest.write_text(json.dumps({
        "inputs": [str(p) for p in paths], "fps": args.fps,
        "frames": [{"step": step, "time": time} for _, step, time in frames],
        "density_definition": "rho_EM = 0.5 * max(W,1e-12)^2 * inverse(conformal_metric)^ij * (E_i E_j + B_i B_j)",
        "movies": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
