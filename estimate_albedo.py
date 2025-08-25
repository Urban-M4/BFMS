#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def linearize_image(img_pil):
    """Convert a PIL RGB image (sRGB) to linear RGB in [0,1]."""
    img_srgb = np.array(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    mask = img_srgb <= 0.04045
    img_lin = np.where(mask, img_srgb / 12.92, ((img_srgb + 0.055) / 1.055) ** 2.4)
    return np.clip(img_lin, 0, 1)


def luminance(img_lin):
    """Compute human-eye weighted luminance (ITU-R BT.709) from linear RGB."""
    Y = 0.2126 * img_lin[..., 0] + 0.7152 * img_lin[..., 1] + 0.0722 * img_lin[..., 2]
    return np.clip(Y, 0, 1)


def lightness_map(img_pil):
    """Compute simple lightness (HSL L) from a PIL RGB image."""
    return linearize_image(img_pil).mean(axis=-1)


def value_map(img_pil):
    """Compute simple brightness (HSV V) from a PIL RGB image."""
    return linearize_image(img_pil).max(axis=-1)


def luminance_map(img_pil):
    """Compute human-eye weighted luminance (ITU-R BT.709) from a PIL RGB image."""
    return luminance(linearize_image(img_pil))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Mask2Former segmentation and save material overlay."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to input image",
    )
    args = parser.parse_args()

    image_path: Path = args.image_path
    return image_path


def main():
    # Parse args
    image_path = parse_args()

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Calculate albedo
    luminance = luminance_map(image)
    albedo = luminance / luminance.max()

    # Create image
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(albedo, vmin=0, vmax=1, cmap="pink")
    ax.set_title("Luminace as proxy for albedo")
    ax.axis("off")

    # Attach colorbar that matches height of axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # adjust size and pad
    plt.colorbar(im, cax=cax)

    output_path = Path.cwd() / (image_path.stem + "_albedo.png")
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 24})
    main()
