"""Small TensorBoard run logger adapted from mandelbrotnn/logger.py."""

from pathlib import Path


class RunFiles:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def child(self, name):
        path = self.directory / name
        path.mkdir(parents=True, exist_ok=True)
        return path
