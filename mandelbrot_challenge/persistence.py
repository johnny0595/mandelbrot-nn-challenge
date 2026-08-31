"""Model persistence kept outside the editable notebook cells."""

from pathlib import Path

import torch


def save_model(model, path="outputs/models/mandelbrot_model.pt", metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)
    return path


def load_model(model, path="outputs/models/mandelbrot_model.pt", device=None):
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint.get("metadata", {})
