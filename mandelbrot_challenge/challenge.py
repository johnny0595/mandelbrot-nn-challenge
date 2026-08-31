"""Locked competition workflow; students supply only model and training choices."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .exploration import explore_training_data
from .persistence import save_model
from .rendering import (
    encode_video,
    make_zoom_video,
    render_model,
    save_full_and_deep,
    save_training_frame,
)
from .targets import mandelbrot_targets, normalize_points, sample_points
from .widgets import view_picker

TIME_BUDGET_SECONDS = 30
TRAINING_POINT_COUNT = 300_000
PUBLIC_VALIDATION_POINT_COUNT = 50_000
TARGET = "smooth"
MAX_DEPTH = 100
SEED = 42
SNAPSHOT_COUNT = 12
REQUIRED_GPU = "T4"


def require_classroom_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Colab select Runtime > Change runtime type > GPU.")
    gpu_name = torch.cuda.get_device_name(0)
    if REQUIRED_GPU not in gpu_name.upper():
        raise RuntimeError(
            f"This competition requires an NVIDIA T4 for fair timing; this runtime has {gpu_name}. "
            "Reconnect to a T4 runtime before training."
        )
    return gpu_name


@dataclass(frozen=True)
class TrainingResult:
    validation_mae: float
    parameter_count: int
    training_seconds: float
    steps: int
    samples_per_second: float
    gpu_name: str
    model_path: Path
    tensorboard_dir: Path
    training_video: Path
    full_render: Path
    deep_render: Path


class Challenge:
    """Concise public facade for the fixed classroom challenge."""

    def __init__(self):
        self.gpu_name = require_classroom_gpu()
        self.device = torch.device("cuda")
        generator = torch.Generator(device=self.device).manual_seed(SEED)
        self._train_points = sample_points(TRAINING_POINT_COUNT, self.device, generator)
        self._train_targets = mandelbrot_targets(self._train_points, MAX_DEPTH, TARGET)
        self._validation_points = sample_points(PUBLIC_VALIDATION_POINT_COUNT, self.device, generator)
        self._validation_targets = mandelbrot_targets(self._validation_points, MAX_DEPTH, TARGET)

    def _score(self, model):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            predictions = model(normalize_points(self._validation_points))
            score = (predictions.reshape_as(self._validation_targets) - self._validation_targets).abs().mean().item()
        if was_training:
            model.train()
        return score

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _train_for_seconds(self, model, optimizer, loss_function, batch_size, scheduler, seconds):
        if not 1 <= batch_size <= TRAINING_POINT_COUNT:
            raise ValueError(f"batch_size must be between 1 and {TRAINING_POINT_COUNT:,}")
        model.to(self.device).train()
        generator = torch.Generator(device=self.device).manual_seed(SEED + 1)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("outputs/runs") / stamp
        writer = SummaryWriter(run_dir)
        frame_dir = Path("outputs/frames") / stamp
        frame_dir.mkdir(parents=True, exist_ok=True)

        initial_prediction = render_model(model, width=384)
        writer.add_image("Prediction/full_view", initial_prediction, 0, dataformats="HW")
        save_training_frame(model, 0, frame_dir)
        snapshot_number = 1
        next_snapshot = seconds / SNAPSHOT_COUNT
        elapsed = 0.0
        steps = 0
        self._synchronize()
        chunk_started = time.perf_counter()

        while elapsed < seconds:
            indices = torch.randint(
                len(self._train_points), (batch_size,), device=self.device, generator=generator
            )
            predictions = model(normalize_points(self._train_points[indices])).reshape(-1, 1)
            loss = loss_function(predictions, self._train_targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss.detach())
                else:
                    scheduler.step()
            steps += 1
            writer.add_scalar("Loss/train", loss.detach().item(), steps)

            # Check after each complete update so a slow, large model cannot receive up
            # to nine extra updates beyond the same time budget as a small model.
            self._synchronize()
            elapsed += time.perf_counter() - chunk_started
            if elapsed >= next_snapshot and snapshot_number < SNAPSHOT_COUNT:
                image = render_model(model, width=480)
                writer.add_image("Prediction/full_view", image, steps, dataformats="HW")
                save_training_frame(model, snapshot_number, frame_dir)
                snapshot_number += 1
                next_snapshot = seconds * snapshot_number / SNAPSHOT_COUNT
            chunk_started = time.perf_counter()

        final_prediction = render_model(model, width=480)
        writer.add_image("Prediction/full_view", final_prediction, steps, dataformats="HW")
        save_training_frame(model, snapshot_number, frame_dir)
        writer.close()
        return elapsed, steps, run_dir, frame_dir

    def train(self, model, optimizer, loss_function, batch_size, scheduler=None):
        """Train for the locked 30-second optimization budget and create all artifacts."""
        elapsed, steps, run_dir, frame_dir = self._train_for_seconds(
            model, optimizer, loss_function, batch_size, scheduler, TIME_BUDGET_SECONDS
        )
        score = self._score(model)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        model_path = save_model(
            model,
            metadata={"public_validation_mae": score, "parameters": parameters, "steps": steps},
        )
        training_video = encode_video(
            frame_dir, Path("outputs/videos") / f"training_views_{frame_dir.name}.mp4", fps=2
        )
        full_render, deep_render = save_full_and_deep(model, target=TARGET, max_depth=MAX_DEPTH)
        return TrainingResult(
            validation_mae=score,
            parameter_count=parameters,
            training_seconds=elapsed,
            steps=steps,
            samples_per_second=steps * batch_size / elapsed,
            gpu_name=self.gpu_name,
            model_path=model_path,
            tensorboard_dir=run_dir,
            training_video=training_video,
            full_render=full_render,
            deep_render=deep_render,
        )

    def explore_data(self):
        """Display training-only EDA before students choose a model."""
        return explore_training_data(
            self._train_points,
            self._train_targets,
            len(self._validation_points),
            TARGET,
            MAX_DEPTH,
        )

    def explore(self, model):
        return view_picker(model, TARGET, MAX_DEPTH)

    def make_zoom_videos(self, model):
        target_video = make_zoom_video(kind="target", target=TARGET, max_depth=MAX_DEPTH)
        model_video = make_zoom_video(model=model, kind="model", target=TARGET, max_depth=MAX_DEPTH)
        return target_video, model_video


def print_result(result):
    print(f"GPU:                    {result.gpu_name}")
    print(f"Public validation MAE: {result.validation_mae:.6f}")
    print(f"Parameters:            {result.parameter_count:,}")
    print(f"Optimization time:     {result.training_seconds:.1f}s")
    print(f"Training steps:        {result.steps:,}")
    print(f"Throughput:            {result.samples_per_second:,.0f} samples/s")
    print(f"Model:                 {result.model_path}")
    print(f"TensorBoard:           {result.tensorboard_dir}")
    print(f"Training video:        {result.training_video}")
