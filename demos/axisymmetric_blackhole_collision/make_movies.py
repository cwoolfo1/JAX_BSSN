"""Render saved axisymmetric Cartoon BSSN fields as x–z heatmap movies."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
import numpy as np
import openpmd_api as io

from JAX_BSSN.diagnostics.mesh_coordinates import mesh_axis_coordinates


DATA_PATH = Path(__file__).resolve().parent / "output" / "axisymmetric_blackhole_collision.h5"
MOVIE_DIR = DATA_PATH.parent / "movies"
DISPLAY_EXTENT = 12.0
FPS = 5
DPI = 120


@dataclass(frozen=True)
class MovieSpec:
    """Describe one logical BSSN variable and its plotted components."""

    output_name: str
    title: str
    mesh_components: tuple[tuple[str, str | None], ...]
    labels: tuple[str, ...]


MOVIES = (
    MovieSpec("conformal_factor_W.mp4", "Conformal factor W", (("W", None),), (r"$W$",)),
    MovieSpec("trace_K.mp4", "Trace of extrinsic curvature K", (("K", None),), (r"$K$",)),
    MovieSpec("lapse.mp4", "Lapse", (("lapse", None),), (r"$\alpha$",)),
    MovieSpec(
        "shift.mp4",
        "Shift",
        (("shift", "x"), ("shift", "y"), ("shift", "z")),
        (r"$\beta^x$", r"$\beta^y$", r"$\beta^z$"),
    ),
    MovieSpec(
        "conformal_connection.mp4",
        "Conformal connection functions",
        (
            ("conformal_connection", "x"),
            ("conformal_connection", "y"),
            ("conformal_connection", "z"),
        ),
        (r"$\tilde{\Gamma}^x$", r"$\tilde{\Gamma}^y$", r"$\tilde{\Gamma}^z$"),
    ),
    MovieSpec(
        "conformal_metric.mp4",
        "Conformal spatial metric",
        tuple((f"conformal_metric_{name}", None) for name in ("xx", "xy", "xz", "yy", "yz", "zz")),
        tuple(rf"$\tilde{{\gamma}}_{{{name}}}$" for name in ("xx", "xy", "xz", "yy", "yz", "zz")),
    ),
    MovieSpec(
        "traceless_K.mp4",
        "Conformal traceless extrinsic curvature",
        tuple((f"traceless_K_{name}", None) for name in ("xx", "xy", "xz", "yy", "yz", "zz")),
        tuple(rf"$\tilde{{A}}_{{{name}}}$" for name in ("xx", "xy", "xz", "yy", "yz", "zz")),
    ),
    MovieSpec(
        "hamiltonian_constraint.mp4",
        "Hamiltonian constraint",
        (("hamiltonian_constraint", None),),
        (r"$\mathcal{H}$",),
    ),
    MovieSpec(
        "momentum_constraint.mp4",
        "Momentum constraint",
        (
            ("momentum_constraint", "x"),
            ("momentum_constraint", "y"),
            ("momentum_constraint", "z"),
        ),
        (r"$\mathcal{M}_x$", r"$\mathcal{M}_y$", r"$\mathcal{M}_z$"),
    ),
    MovieSpec(
        "det_gamma_constraint.mp4",
        "Unit-determinant constraint",
        (("det_gamma_constraint", None),),
        (r"$\det(\tilde{\gamma})-1$",),
    ),
    MovieSpec(
        "trace_A_constraint.mp4",
        "Trace-free constraint",
        (("trace_A_constraint", None),),
        (r"$\tilde{\gamma}^{ij}\tilde{A}_{ij}$",),
    ),
    MovieSpec(
        "gamma_constraint.mp4",
        "Conformal-connection constraint",
        (
            ("gamma_constraint", "x"),
            ("gamma_constraint", "y"),
            ("gamma_constraint", "z"),
        ),
        (r"$\mathcal{G}^x$", r"$\mathcal{G}^y$", r"$\mathcal{G}^z$"),
    ),
)


def load_frame(series, step, spec, display_extent=DISPLAY_EXTENT):
    """Load just the displayed x–z region for one movie frame."""
    if not np.isfinite(display_extent) or display_extent <= 0:
        raise ValueError("display_extent must be finite and positive")
    iteration = series.iterations[step]
    # The color-limit scan closes each iteration before the rendering pass.
    iteration.open()
    pending = []
    coordinates = None
    try:
        for mesh_name, requested_component in spec.mesh_components:
            mesh = iteration.meshes[mesh_name]
            component = mesh[requested_component if requested_component is not None
                             else io.Mesh_Record_Component.SCALAR]
            if len(component.shape) != 3 or component.shape[1] != 1:
                raise ValueError(f"{mesh_name} must contain a single x–z plane")
            x = mesh_axis_coordinates(mesh, component, 0)
            z = mesh_axis_coordinates(mesh, component, 2)
            ix = np.flatnonzero(np.abs(x) <= display_extent)
            iz = np.flatnonzero(np.abs(z) <= display_extent)
            if not ix.size or not iz.size:
                raise ValueError("No grid samples lie inside the display extent")
            edges = (
                mesh_axis_coordinates(mesh, component, 0, edges=True)[ix[0]:ix[-1] + 2],
                mesh_axis_coordinates(mesh, component, 2, edges=True)[iz[0]:iz[-1] + 2],
            )
            if coordinates is None:
                coordinates = edges
            elif any(not np.array_equal(a, b) for a, b in zip(coordinates, edges)):
                raise ValueError("Movie components have inconsistent mesh coordinates")
            pending.append(component.load_chunk(
                [int(ix[0]), 0, int(iz[0])], [int(ix.size), 1, int(iz.size)]
            ))
        series.flush()
        frames = tuple(np.array(array[:, 0, :].T, copy=True) for array in pending)
        return float(iteration.time), *coordinates, frames
    finally:
        iteration.close()


def render_movie(
    spec,
    path=DATA_PATH,
    movie_dir=MOVIE_DIR,
    display_extent=DISPLAY_EXTENT,
    fps=FPS,
    dpi=DPI,
):
    """Scan color limits, then stream one frame at a time to FFmpeg."""
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg is required to write MP4 movies; install it and make it available on PATH.")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("fps and dpi must be finite and positive")
    path, movie_dir = Path(path), Path(movie_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Collision output not found: {path}")
    series = io.Series(str(path), io.Access.read_only)
    figure = None
    try:
        steps = sorted(series.iterations)
        if not steps:
            raise ValueError(f"No iterations found in {path}")
        limits = np.array([[np.inf, -np.inf]] * len(spec.mesh_components))
        first_bad = None
        coordinates = None
        for step in steps:
            time, x_edges, z_edges, frames = load_frame(series, step, spec, display_extent)
            if coordinates is None:
                coordinates = (x_edges, z_edges)
            elif any(not np.array_equal(a, b) for a, b in zip(coordinates, (x_edges, z_edges))):
                raise ValueError("Movie mesh coordinates changed between iterations")
            for index, frame in enumerate(frames):
                finite = frame[np.isfinite(frame)]
                if finite.size != frame.size and first_bad is None:
                    first_bad = (step, time)
                if finite.size:
                    limits[index, 0] = min(limits[index, 0], float(finite.min()))
                    limits[index, 1] = max(limits[index, 1], float(finite.max()))
        if first_bad is not None:
            print(f"{spec.title}: first nonfinite displayed data at step {first_bad[0]}, t={first_bad[1]:.8g}")

        component_count = len(spec.mesh_components)
        rows, columns = {1: (1, 1), 3: (1, 3), 6: (2, 3)}[component_count]
        figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.8 * rows), squeeze=False)
        images, warnings = [], []
        for axis, label, (lower, upper) in zip(axes.flat, spec.labels, limits):
            if not np.isfinite(lower) or not np.isfinite(upper):
                lower, upper = -1.0, 1.0
            if np.isclose(lower, upper, rtol=1e-12, atol=1e-15):
                padding = 0.05 * max(abs(lower), abs(upper), 1e-12)
                lower, upper = lower - padding, upper + padding
            signed = lower < 0 < upper
            if signed:
                lower, upper = -max(abs(lower), abs(upper)), max(abs(lower), abs(upper))
            plot = axis.pcolormesh(
                x_edges, z_edges, np.zeros_like(frames[0]), shading="flat",
                cmap="RdBu_r" if signed else "viridis", vmin=lower, vmax=upper,
            )
            images.append(plot)
            figure.colorbar(plot, ax=axis)
            axis.set(xlabel="x", ylabel="z", title=label, aspect="equal")
            warnings.append(axis.text(0.5, 0.5, "NON-FINITE DATA", transform=axis.transAxes,
                                      ha="center", color="crimson", visible=False))
        title = figure.suptitle(spec.title)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        movie_dir.mkdir(parents=True, exist_ok=True)
        output_path = movie_dir / spec.output_name
        writer = FFMpegWriter(
            fps=fps, codec="libx264", bitrate=2200,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            metadata={"title": spec.title},
        )
        with writer.saving(figure, output_path, dpi=dpi):
            for step in steps:
                time, _, _, frames = load_frame(series, step, spec, display_extent)
                invalid = False
                for plot, warning, frame in zip(images, warnings, frames):
                    bad = not np.isfinite(frame).all()
                    invalid |= bad
                    plot.set_array(np.ma.masked_invalid(frame).ravel())
                    warning.set_visible(bad)
                title.set_text(f"{spec.title} — step {step}, t = {time:.6f}" +
                               (" — NON-FINITE DATA" if invalid else ""))
                title.set_color("crimson" if invalid else "black")
                writer.grab_frame()
        return output_path
    finally:
        if figure is not None:
            plt.close(figure)
        series.close()


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Collision output not found: {DATA_PATH}")
    series = io.Series(str(DATA_PATH), io.Access.read_only)
    available = []
    try:
        steps = sorted(series.iterations)
        if not steps:
            raise ValueError(f"No iterations found in {DATA_PATH}")
        iteration = series.iterations[steps[0]]
        for spec in MOVIES:
            if all(name in iteration.meshes and
                   (component if component is not None else io.Mesh_Record_Component.SCALAR)
                   in iteration.meshes[name] for name, component in spec.mesh_components):
                available.append(spec)
            else:
                print(f"Skipped fields absent from this file: {spec.title}")
        iteration.close()
    finally:
        series.close()
    for spec in available:
        print(f"Wrote {render_movie(spec)}")


if __name__ == "__main__":
    main()
