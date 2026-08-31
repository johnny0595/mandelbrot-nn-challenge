import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from mandelbrot_challenge.persistence import load_model, save_model
from mandelbrot_challenge.challenge import Challenge, require_competition_gpu
from mandelbrot_challenge.exploration import explore_training_data
from mandelbrot_challenge.rendering import (
    PRESET_VIEWS,
    make_zoom_video,
    render_comparison,
    view_grid,
)
from mandelbrot_challenge.targets import mandelbrot_targets, normalize_points, sample_points
from mandelbrot_challenge.widgets import _colab_json, _viewer_html, _zoom_payload


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, points):
        return self.layers(points)


class CoreTests(unittest.TestCase):
    @patch("mandelbrot_challenge.challenge.torch.cuda.get_device_name", return_value="Tesla T4")
    @patch("mandelbrot_challenge.challenge.torch.cuda.is_available", return_value=True)
    def test_competition_gpu_accepts_t4(self, _available, _name):
        self.assertEqual(require_competition_gpu(), "Tesla T4")

    @patch("mandelbrot_challenge.challenge.torch.cuda.get_device_name", return_value="NVIDIA L4")
    @patch("mandelbrot_challenge.challenge.torch.cuda.is_available", return_value=True)
    def test_competition_gpu_rejects_other_hardware(self, _available, _name):
        with self.assertRaisesRegex(RuntimeError, "requires an NVIDIA T4"):
            require_competition_gpu()

    def test_targets_support_both_modes(self):
        generator = torch.Generator().manual_seed(42)
        points = sample_points(128, torch.device("cpu"), generator)
        for target in ("smooth", "periodic"):
            values = mandelbrot_targets(points, max_depth=20, target=target, precision=32)
            self.assertEqual(values.shape, (128, 1))
            self.assertTrue(bool(((0 <= values) & (values <= 1)).all()))

    @patch("mandelbrot_challenge.exploration.plt.show")
    @patch("mandelbrot_challenge.exploration.render_target", return_value=np.zeros((8, 12)))
    def test_eda_reports_shape_balance_and_baseline(self, _render, _show):
        points = torch.tensor([[-2.0, -1.0], [-1.0, 0.0], [0.0, 0.5], [1.0, 1.0]])
        targets = torch.tensor([[0.0], [0.25], [0.75], [1.0]])
        summary = explore_training_data(points, targets, validation_count=2)
        self.assertEqual(summary["training_examples"], 4)
        self.assertEqual(summary["public_validation_examples"], 2)
        self.assertAlmostEqual(summary["target_mean"], 0.5)
        self.assertAlmostEqual(summary["constant_baseline_mae"], 0.375)

    def test_model_render_and_persistence(self):
        model = TinyModel()
        truth, prediction, error = render_comparison(
            model, PRESET_VIEWS["Deep mini"], width=32, max_depth=10
        )
        self.assertEqual(truth.shape, prediction.shape)
        self.assertEqual(error.shape, prediction.shape)
        with tempfile.TemporaryDirectory() as directory:
            path = save_model(model, Path(directory) / "model.pt", {"test": True})
            restored, metadata = load_model(TinyModel(), path)
            self.assertTrue(metadata["test"])
            points = normalize_points(torch.tensor([[-0.75, 0.0]]))
            self.assertTrue(torch.allclose(model(points), restored(points)))

    def test_interactive_viewer_contains_shared_layer_controls(self):
        image = np.zeros((8, 12))
        html = _viewer_html(image, image, image, "challenge.render")
        for text in (
            "Target",
            "Prediction",
            "Absolute error",
            "wheel",
            "pointermove",
            "invokeFunction",
            "challenge.render",
            "Reset view",
            "one fresh render starts when the gesture pauses",
            "scheduleRender",
            "maxPreviewScale",
            "payload?.images",
        ):
            self.assertIn(text, html)
        self.assertNotIn("Re-render this view", html)

    def test_colab_callback_returns_json_mime_payload(self):
        payload = {"images": {"target": "data:image/png;base64,test"}, "max_depth": 100}
        ipython = ModuleType("IPython")
        display = ModuleType("IPython.display")
        display.JSON = lambda data: type("JsonResponse", (), {"data": data})()
        ipython.display = display
        with patch.dict(sys.modules, {"IPython": ipython, "IPython.display": display}):
            response = _colab_json(payload)
        self.assertEqual(response.data, payload)

    def test_deep_target_grid_can_preserve_float64_coordinates(self):
        points, _, _ = view_grid(PRESET_VIEWS["Deep mini"], 16, torch.device("cpu"), torch.float64)
        self.assertEqual(points.dtype, torch.float64)

    def test_zoom_payload_recomputes_a_deeper_view(self):
        view = PRESET_VIEWS["Deep mini"]
        payload = _zoom_payload(
            TinyModel(), (view.xmin, view.xmax, view.ymin, view.ymax), width=16
        )
        self.assertGreater(payload["max_depth"], 100)
        self.assertEqual(set(payload["images"]), {"target", "prediction", "error"})
        self.assertTrue(all(image.startswith("data:image/png;base64,") for image in payload["images"].values()))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for video verification")
    def test_training_and_zoom_video_pipeline(self):
        model = TinyModel()
        challenge = Challenge.__new__(Challenge)
        challenge.device = torch.device("cpu")
        generator = torch.Generator().manual_seed(42)
        challenge._train_points = sample_points(256, challenge.device, generator)
        challenge._train_targets = mandelbrot_targets(challenge._train_points, max_depth=10)
        challenge._validation_points = sample_points(64, challenge.device, generator)
        challenge._validation_targets = mandelbrot_targets(challenge._validation_points, max_depth=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                import os
                os.chdir(directory)
                elapsed, steps, run_dir, frame_dir = challenge._train_for_seconds(
                    model, optimizer, nn.MSELoss(), 32, None, 0.01
                )
                self.assertGreater(elapsed, 0)
                self.assertGreater(steps, 0)
                events = EventAccumulator(str(run_dir)).Reload()
                self.assertEqual(events.Tags()["scalars"], ["Loss/train"])
                self.assertEqual(events.Tags()["images"], ["Prediction/full_view"])
                self.assertGreaterEqual(len(events.Images("Prediction/full_view")), 2)
                video = make_zoom_video(
                    model=model, kind="model", frames=2, width=32, fps=2, output_dir=directory
                )
                self.assertTrue(video.exists())
                self.assertTrue(any(frame_dir.glob("*.png")))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
