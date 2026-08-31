"""Small, beginner-friendly exploratory data analysis for the notebook."""

import matplotlib.pyplot as plt

from .rendering import PRESET_VIEWS, render_target
from .targets import XMAX, XMIN, YMAX, YMIN


def explore_training_data(points, targets, validation_count, target="smooth", max_depth=100):
    """Show spatial structure, target distribution, and an easy baseline."""
    sample_count = min(5_000, len(points))
    sample_points = points[:sample_count].detach().cpu()
    sample_targets = targets[:sample_count].detach().reshape(-1).cpu()
    all_targets = targets.detach().reshape(-1).cpu()
    target_image = render_target(
        PRESET_VIEWS["Full set"], width=600, target=target, max_depth=max_depth
    )

    median = all_targets.median()
    summary = {
        "training_examples": len(points),
        "public_validation_examples": validation_count,
        "target_mean": all_targets.mean().item(),
        "target_median": median.item(),
        "fraction_zero": (all_targets == 0).float().mean().item(),
        "fraction_one": (all_targets == 1).float().mean().item(),
        "constant_baseline_mae": (all_targets - median).abs().mean().item(),
    }

    print(f"Training examples:          {summary['training_examples']:,}")
    print(f"Public validation examples: {summary['public_validation_examples']:,}")
    print(f"Raw x range:                [{XMIN}, {XMAX}]")
    print(f"Raw y range:                [{YMIN}, {YMAX}]")
    print("Network input range:        [-1, 1] for both coordinates")
    print(f"Target mean / median:       {summary['target_mean']:.3f} / {summary['target_median']:.3f}")
    print(f"Targets exactly 0 / 1:      {summary['fraction_zero']:.1%} / {summary['fraction_one']:.1%}")
    print(f"Constant baseline MAE:      {summary['constant_baseline_mae']:.3f}")

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    image = axes[0].imshow(
        target_image,
        cmap="inferno",
        origin="lower",
        extent=(XMIN, XMAX, YMIN, YMAX),
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("The target function")
    axes[0].set_xlabel("x (real)")
    axes[0].set_ylabel("y (imaginary)")
    figure.colorbar(image, ax=axes[0], label="target")

    scatter = axes[1].scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        c=sample_targets,
        cmap="inferno",
        vmin=0,
        vmax=1,
        s=4,
        alpha=0.7,
    )
    axes[1].set_title(f"{sample_count:,} sampled training points")
    axes[1].set_xlabel("x (real)")
    axes[1].set_ylabel("y (imaginary)")
    figure.colorbar(scatter, ax=axes[1], label="target")

    axes[2].hist(all_targets.numpy(), bins=40, range=(0, 1), color="#f06a39")
    axes[2].axvline(median.item(), color="black", linestyle="--", label="median baseline")
    axes[2].set_title("How common is each target value?")
    axes[2].set_xlabel("target value")
    axes[2].set_ylabel("training examples")
    axes[2].legend()

    figure.tight_layout()
    plt.show()
    return summary
