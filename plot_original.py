import argparse
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt


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

    # Create image
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    ax.set_title("Original image")
    ax.axis("off")

    output_path = Path.cwd() / (image_path.stem + "_original.png")
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 24})
    main()
