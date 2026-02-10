import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class YCoCg:
    y: np.ndarray
    co: np.ndarray
    cg: np.ndarray


def rgb_to_ycocg(rgb01: np.ndarray) -> YCoCg:
    """Convert RGB in [0,1] to YCoCg.

    Conventions (float):
      Y  = 0.25 R + 0.5 G + 0.25 B
      Co = 0.5 R - 0.5 B
      Cg = -0.25 R + 0.5 G - 0.25 B

    With RGB in [0,1], Y is in [0,1] and Co/Cg are roughly in [-0.5, 0.5].
    """
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {rgb01.shape}")

    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]

    y = 0.25 * r + 0.5 * g + 0.25 * b
    co = 0.5 * r - 0.5 * b
    cg = -0.25 * r + 0.5 * g - 0.25 * b

    return YCoCg(y=y, co=co, cg=cg)


def load_image_rgb01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def sample_every_nth_pixel(arr: np.ndarray, step: int) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be > 0")
    return arr[::step, ::step]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load an image, convert RGB->YCoCg, and plot every Nth pixel as a 3D point "
            "with UP axis = Y (luma)."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=16,
        help="Sample every Nth pixel in both x and y (default: 16)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Scatter point size (default: 2.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Point alpha (default: 0.8)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    rgb01 = load_image_rgb01(input_path)
    height, width, _ = rgb01.shape

    rgb01_s = sample_every_nth_pixel(rgb01, args.step)
    ycocg = rgb_to_ycocg(rgb01_s)

    # Original pixel coordinates for sampled points
    xs = np.arange(0, width, args.step, dtype=np.float32)
    ys = np.arange(0, height, args.step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    x = grid_x.reshape(-1)
    y = grid_y.reshape(-1)
    z = ycocg.y.reshape(-1)  # UP axis

    colors = rgb01_s.reshape(-1, 3)

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        ) from exc

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, c=colors, s=args.point_size, alpha=args.alpha, linewidths=0)

    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_zlabel("Y (UP)")
    ax.set_title(f"YCoCg 3D scatter (step={args.step})\n{input_path.name}")

    ax.set_xlim(0, max(0, width - 1))
    ax.set_ylim(0, max(0, height - 1))
    ax.set_zlim(0.0, 1.0)

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
