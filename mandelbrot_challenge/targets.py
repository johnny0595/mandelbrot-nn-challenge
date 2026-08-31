"""Mandelbrot targets adapted from mandelbrotnn/src/dataset.py."""

import math

import torch

XMIN, XMAX = -2.65, 1.15
YMIN, YMAX = -1.2, 1.2
PERIODIC_ESCAPE_R = 1.0e4
PERIODIC_BETA = 0.050
_TWO_PI = 2.0 * math.pi


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    """Map raw Mandelbrot coordinates to [-1, 1]^2 for student models."""
    x = 2 * (points[:, 0:1] - XMIN) / (XMAX - XMIN) - 1
    y = 2 * (points[:, 1:2] - YMIN) / (YMAX - YMIN) - 1
    return torch.cat((x, y), dim=1)


def sample_points(count: int, device: torch.device, generator=None) -> torch.Tensor:
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] = points[:, 0] * (XMAX - XMIN) + XMIN
    points[:, 1] = points[:, 1] * (YMAX - YMIN) + YMIN
    return points


def _dtypes(precision: int):
    if precision == 32:
        return torch.float32, torch.complex64
    return torch.float64, torch.complex128


def _periodic_target(c: torch.Tensor, max_depth: int) -> torch.Tensor:
    z = torch.zeros_like(c)
    dz = torch.zeros_like(c)
    z_esc = torch.zeros_like(c)
    dz_esc = torch.zeros_like(c)
    alive = torch.ones(c.shape, dtype=torch.bool, device=c.device)

    for _ in range(max_depth):
        dz = torch.where(alive, 2.0 * z * dz + 1.0, dz)
        z = torch.where(alive, z * z + c, z)
        escaped = alive & (z.abs() > PERIODIC_ESCAPE_R)
        z_esc = torch.where(escaped, z, z_esc)
        dz_esc = torch.where(escaped, dz, dz_esc)
        alive &= ~escaped

    zmag = z_esc.abs()
    dzmag = dz_esc.abs().clamp_min(1e-20)
    distance = zmag * torch.log(zmag.clamp_min(1.0001)) / dzmag
    phase = PERIODIC_BETA * torch.log(distance.clamp_min(1e-30))
    values = 0.5 + 0.5 * torch.sin(_TWO_PI * phase)
    values[alive] = 1.0
    return values


@torch.no_grad()
def mandelbrot_targets(
    points: torch.Tensor,
    max_depth: int = 100,
    target: str = "smooth",
    precision: int = 32,
) -> torch.Tensor:
    """Return smooth or periodic targets for raw ``(x, y)`` coordinates."""
    if target not in {"smooth", "periodic"}:
        raise ValueError("target must be 'smooth' or 'periodic'")
    real_dtype, complex_dtype = _dtypes(precision)
    c = torch.complex(points[:, 0], points[:, 1]).to(complex_dtype)
    if target == "periodic":
        return _periodic_target(c, max_depth).to(torch.float32).unsqueeze(1)

    z = torch.zeros_like(c)
    alive = torch.ones(len(points), dtype=torch.bool, device=points.device)
    values = torch.ones(len(points), dtype=real_dtype, device=points.device)
    for iteration in range(max_depth):
        z = torch.where(alive, z * z + c, z)
        escaped = alive & (z.abs() > 2)
        escape_value = 1 - 1 / (iteration / 50 + 1)
        values = torch.where(escaped, escape_value, values)
        alive &= ~escaped
    return values.to(torch.float32).unsqueeze(1)


def mandelbrot_tensor(imag_values, real_values, max_depth, target="smooth", precision=32):
    """Compatibility wrapper used by private instructor tools."""
    points = torch.stack((real_values.reshape(-1), imag_values.reshape(-1)), dim=1)
    return mandelbrot_targets(points, max_depth, target, precision).reshape(real_values.shape)
