#!/home/peter/urban-m4/streetscapes/.venv/bin/python
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import transformers as tform
import sam2.sam2_image_predictor as sam2_pred

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3

# Default labels (can be customized)
DEFAULT_LABELS = {
    "building": {"window": None, "door": None},
    "road": {
        "street": None,
        "sidewalk": None,
        "pavement": None,
        "crosswalk": None,
    },
    "vegetation": None,
    "car": None,
    "truck": None,
}

# Color palette for visualization
COLORS = np.array(
    [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 128, 128),  # Light Red
        (128, 255, 128),  # Light Green
        (128, 128, 255),  # Light Blue
        (255, 255, 128),  # Light Yellow
        (255, 128, 255),  # Light Magenta
        (128, 255, 255),  # Light Cyan
        (128, 128, 128),  # Gray
    ],
    dtype=np.uint8,
)


def flatten_labels(labels):
    """Convert nested label dictionary to flat list of labels."""
    flat_labels = []

    def _flatten(tree):
        for k, v in tree.items():
            if isinstance(v, dict):
                flat_labels.append(k)
                _flatten(v)
            else:
                flat_labels.append(k)

    _flatten(labels)
    return flat_labels


def create_prompt(labels):
    """Create text prompt from labels."""
    flat_labels = flatten_labels(labels)
    return " ".join([lbl.strip() + "." for lbl in flat_labels if lbl])


def pad_image(image, target_size=2048):
    """Pad image to target size."""
    h, w = image.shape[:2]
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    if image.ndim == 3:
        padded = np.pad(
            image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
        )
    else:
        padded = np.pad(
            image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

    return padded


def unpad_mask(mask, original_shape):
    """Remove padding from mask to match original image size."""
    h, w = original_shape
    return mask[:h, :w]


def plot_overlay(image, masks, labels):
    """Plot image with segmentation masks overlaid."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Start with original image
    overlay = np.array(image)

    # Create legend handles
    legend_handles = []

    # Apply each mask with different color
    for i, (inst_id, mask) in enumerate(masks.items()):
        color = COLORS[i % len(COLORS)]
        label = labels.get(inst_id, f"Object {inst_id}")

        # Create colored overlay
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask] = color

        # Blend with original image
        overlay = np.where(
            mask[..., None],
            (0.7 * overlay + 0.3 * colored_mask).astype(np.uint8),
            overlay,
        )

        # Add to legend
        legend_handles.append(Patch(color=color / 255.0, label=label))

    ax.imshow(overlay)
    ax.axis("off")

    # Add legend if there are masks
    if legend_handles:
        plt.legend(
            handles=legend_handles,
            bbox_to_anchor=(1, 1),
            loc="upper left",
            fontsize="small",
            frameon=False,
        )

    ax.set_title("Instance segmentation with DinoSAM")

    fig.tight_layout()
    return fig, ax


def plot_side_by_side(image, masks, labels):
    """Plot original image and segmentation side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # Original image
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Segmentation mask
    seg_image = np.zeros_like(np.array(image))
    legend_handles = []

    for i, (inst_id, mask) in enumerate(masks.items()):
        color = COLORS[i % len(COLORS)]
        label = labels.get(inst_id, f"Object {inst_id}")

        seg_image[mask] = color
        legend_handles.append(Patch(color=color / 255.0, label=label))

    axs[1].imshow(seg_image)
    axs[1].set_title("Segmentation")
    axs[1].axis("off")

    # Add legend
    if legend_handles:
        plt.legend(
            handles=legend_handles,
            bbox_to_anchor=(1, 1),
            loc="upper left",
            fontsize="small",
            frameon=False,
        )

    fig.tight_layout()
    return fig, axs


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic segmentation and save overlay."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to input image",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)

    # Load models
    print("Loading SAM model...")
    sam_model = sam2_pred.SAM2ImagePredictor.from_pretrained(
        "facebook/sam2.1-hiera-large", device=DEVICE
    )

    print("Loading DINO model...")
    dino_processor = tform.AutoProcessor.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    )
    dino_model = tform.AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(DEVICE)
    dino_model.eval()

    # Load and prepare image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    original_shape = image_np.shape[:2]

    # Pad image for processing
    padded_image = pad_image(image_np)

    # Create prompt from labels
    prompt = create_prompt(DEFAULT_LABELS)
    print(f"Using prompt: {prompt}")

    # Run object detection
    print("Running object detection...")
    inputs = dino_processor(
        images=[padded_image], text=[prompt], return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[padded_image.shape[:2]],
    )[0]

    print(f"Detected {len(results['boxes'])} objects")

    # Run segmentation if objects detected
    if len(results["boxes"]) > 0:
        print("Running segmentation...")
        boxes = results["boxes"].cpu().numpy()

        sam_model.set_image(padded_image)
        masks, _, _ = sam_model.predict(box=boxes, multimask_output=False)

        # Process masks
        masks = np.squeeze(masks, axis=1)  # Remove extra dimension

        # Create final masks and labels (merged by class)
        merged_masks = {}
        for label, mask in zip(results["labels"], masks):
            # Remove padding
            mask = unpad_mask(mask, original_shape)
            mask = mask > 0  # convert to boolean

            # Merge into existing class mask
            if label in merged_masks:
                merged_masks[label] = merged_masks[label] | mask
            else:
                merged_masks[label] = mask

        print(f"Generated {len(merged_masks)} class-level segmentation masks")

        # Create visualizations
        print("Creating overlay...")
        fig, ax = plot_overlay(image, merged_masks, {k: k for k in merged_masks})
        output_filename = f"{image_path.stem}_segmentation_overlay.png"
        output_path = Path.cwd() / output_filename
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved overlay to {output_path}")

        print("Creating side-by-side comparison...")
        fig, axs = plot_side_by_side(image, merged_masks, {k: k for k in merged_masks})
        output_filename = f"{image_path.stem}_segmentation_sidebyside.png"
        output_path = Path.cwd() / output_filename
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved side-by-side image to {output_path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 24})
    main()
