"""Rendering and video helpers adapted from videomaker.py and the zoom scripts."""

from dataclasses import dataclass
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch

from .targets import XMAX, XMIN, YMAX, YMIN, mandelbrot_targets, normalize_points


@dataclass(frozen=True)
class View:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float


PRESET_VIEWS = {
    "Full set": View("Full set", XMIN, XMAX, YMIN, YMAX),
    "Left bulb": View("Left bulb", -2.095, -1.095, -0.3158, 0.3158),
    "Mini Mandelbrot": View("Mini Mandelbrot", -1.905, -1.605, -0.0947, 0.0947),
    "Deep mini": View("Deep mini", -1.805, -1.705, -0.028125, 0.028125),
}


def model_device(model):
    return next(model.parameters()).device


def window_resolution(width, view):
    height = round(width * (view.ymax - view.ymin) / (view.xmax - view.xmin))
    return width, height + height % 2


def view_grid(view, width, device, dtype=torch.float32):
    width, height = window_resolution(width, view)
    xs = torch.linspace(view.xmin, view.xmax, width, device=device, dtype=dtype)
    ys = torch.linspace(view.ymin, view.ymax, height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
    return points, height, width


@torch.no_grad()
def render_model(model, view=PRESET_VIEWS["Full set"], width=800, chunk_size=131_072):
    device = model_device(model)
    points, height, width = view_grid(view, width, device)
    was_training = model.training
    model.eval()
    parts = []
    for start in range(0, len(points), chunk_size):
        parts.append(model(normalize_points(points[start:start + chunk_size])).reshape(-1))
    if was_training:
        model.train()
    return torch.cat(parts).reshape(height, width).clamp(0, 1).cpu().numpy()


@torch.no_grad()
def render_target(
    view=PRESET_VIEWS["Full set"], width=800, target="smooth", max_depth=100, precision=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if precision == 64 else torch.float32
    points, height, width = view_grid(view, width, device, dtype=dtype)
    values = mandelbrot_targets(points, max_depth=max_depth, target=target, precision=precision)
    return values.reshape(height, width).cpu().numpy()


def render_comparison(
    model,
    view=PRESET_VIEWS["Full set"],
    width=800,
    target="smooth",
    max_depth=100,
    target_precision=32,
):
    truth = render_target(view, width, target, max_depth, target_precision)
    prediction = render_model(model, view, width)
    return truth, prediction, np.abs(truth - prediction)


def save_comparison(model, view, path, width=800, target="smooth", max_depth=100):
    truth, prediction, error = render_comparison(model, view, width, target, max_depth)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, image, title in zip(
        axes, (truth, prediction, error), ("Target", "Prediction", "Absolute error")
    ):
        axis.imshow(image, cmap="inferno", origin="lower")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(view.name)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


def save_full_and_deep(model, output_dir="outputs/renders", target="smooth", max_depth=100, width=800):
    output_dir = Path(output_dir)
    full = save_comparison(model, PRESET_VIEWS["Full set"], output_dir / "full_view.png", width=width, target=target, max_depth=max_depth)
    deep = save_comparison(model, PRESET_VIEWS["Deep mini"], output_dir / "deep_zoom.png", width=width, target=target, max_depth=max_depth)
    return full, deep


def encode_video(frame_dir, output_path, fps=12):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(Path(frame_dir) / "%05d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(output_path),
        ],
        check=True,
    )
    return output_path


def save_training_frame(model, frame_number, output_dir="outputs/frames/training", width=480):
    views = tuple(PRESET_VIEWS.values())
    view = views[frame_number % len(views)]
    image = render_model(model, view, width)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{frame_number:05d}.png"
    plt.imsave(path, image, cmap="inferno", origin="lower", vmin=0, vmax=1)
    return path


def _zoom_views(destination, frames):
    start = PRESET_VIEWS["Full set"]
    start_cx = (start.xmin + start.xmax) / 2
    start_cy = (start.ymin + start.ymax) / 2
    end_cx = (destination.xmin + destination.xmax) / 2
    end_cy = (destination.ymin + destination.ymax) / 2
    start_w, start_h = start.xmax - start.xmin, start.ymax - start.ymin
    end_w, end_h = destination.xmax - destination.xmin, destination.ymax - destination.ymin
    for index in range(frames):
        progress = index / max(1, frames - 1)
        cx = start_cx + (end_cx - start_cx) * progress
        cy = start_cy + (end_cy - start_cy) * progress
        width = start_w * (end_w / start_w) ** progress
        height = start_h * (end_h / start_h) ** progress
        yield View(destination.name, cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2)


def make_zoom_video(model=None, kind="model", destination="Deep mini", frames=45, width=480, fps=15, target="smooth", max_depth=100, output_dir="outputs"):
    """Create the original-style true-fractal or trained-model zoom video."""
    if kind not in {"model", "target"}:
        raise ValueError("kind must be 'model' or 'target'")
    if kind == "model" and model is None:
        raise ValueError("model is required for a model zoom")
    frame_dir = Path(output_dir) / "frames" / f"{kind}_zoom"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(_zoom_views(PRESET_VIEWS[destination], frames)):
        image = render_model(model, view, width) if kind == "model" else render_target(view, width, target, max_depth)
        plt.imsave(frame_dir / f"{index:05d}.png", image, cmap="inferno", origin="lower", vmin=0, vmax=1)
    return encode_video(frame_dir, Path(output_dir) / "videos" / f"{kind}_zoom.mp4", fps)


# Compatibility names used by the private instructor evaluator.
renderModel = render_model


def render_model_window(model, width=960, **_):
    return render_model(model, PRESET_VIEWS["Full set"], width)


renderModelWindow = render_model_window
