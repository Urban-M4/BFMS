#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Mask2Former segmentation and save material overlay."
    )
    parser.add_argument("image_path", type=Path, help="Path to input image")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("./trained_model"),
        help="Path to model folder",
    )
    args = parser.parse_args()

    image_path: Path = args.image_path
    model_path: Path = args.model_path

    # Load config and model
    config = Mask2FormerConfig.from_pretrained(model_path / "config.json")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_path / "model.safetensors", config=config
    )
    model.eval()

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        model_path / "config.json", use_fast=True
    )

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits
    mask_logits = outputs.masks_queries_logits[0]  # [Q, H, W]
    class_logits = outputs.class_queries_logits[0]  # [Q, C]

    # Convert to probabilities
    masks_probs = torch.sigmoid(mask_logits)
    class_probs = torch.softmax(class_logits, dim=-1)

    # Combine
    pixel_class_probs = torch.einsum("qc,qhw->chw", class_probs, masks_probs)
    semantic_mask = torch.argmax(pixel_class_probs, dim=0).cpu().numpy()

    # Label map
    id2label = {
        0: "Background",
        1: "Wood/Bamboo",
        2: "Ground tile",
        3: "Brick",
        4: "Cardboard/Paper",
        5: "Tree",
        6: "Roof tile",
        7: "Ceramic",
        8: "Chalkboard/Blackboard",
        9: "Asphalt",
        10: "Cement/Concrete",
        11: "Composite decorative board",
        12: "Rammed earth",
        13: "Fabric/Cloth",
        14: "Water",
        15: "Windows with metal fences",
        16: "Foliage",
        17: "Food",
        18: "Fur",
        19: "Pottery",
        20: "Glass",
        21: "Hair",
        22: "Roofing waterproof material",
        23: "Ice",
        24: "Leather",
        25: "Carved brick",
        26: "Metal",
        27: "Mirror",
        28: "Enamel",
        29: "Paint/Coating/Plaster",
        30: "Window screen",
        31: "Whiteboard",
        32: "Photograph/Painting/Airbrushed fabric",
        33: "Plastic, clear",
        34: "Plastic, non-clear",
        35: "Rubber/Latex",
        36: "Sand",
        37: "Skin/Lips",
        38: "Sky",
        39: "Snow",
        40: "Engineered Stone/Imitation Stone",
        41: "Soil/Mud",
        42: "Natural Stone",
    }

    label_colors = np.array(
        [
            (0, 0, 0),
            (139, 69, 19),
            (205, 133, 63),
            (178, 34, 34),
            (210, 180, 140),
            (34, 139, 34),
            (255, 165, 0),
            (255, 215, 0),
            (0, 0, 128),
            (128, 128, 128),
            (192, 192, 192),
            (255, 105, 180),
            (139, 0, 0),
            (75, 0, 130),
            (0, 191, 255),
            (70, 130, 180),
            (0, 128, 0),
            (255, 20, 147),
            (160, 82, 45),
            (184, 134, 11),
            (0, 255, 255),
            (255, 192, 203),
            (0, 100, 0),
            (176, 224, 230),
            (139, 69, 19),
            (205, 92, 92),
            (192, 192, 192),
            (255, 250, 250),
            (255, 0, 255),
            (173, 216, 230),
            (255, 228, 196),
            (245, 245, 245),
            (255, 239, 213),
            (135, 206, 250),
            (105, 105, 105),
            (128, 0, 128),
            (194, 178, 128),
            (255, 182, 193),
            (135, 206, 235),
            (255, 250, 250),
            (128, 128, 0),
            (139, 69, 19),
            (169, 169, 169),
        ],
        dtype=np.uint8,
    )

    # Map semantic mask to RGB
    h, w = semantic_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, color in enumerate(label_colors):
        color_mask[semantic_mask == lbl] = color

    # Overlay
    image_np = np.array(image.resize((w, h)))
    overlay = (0.5 * image_np + 0.5 * color_mask).astype(np.uint8)

    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(overlay)
    plt.axis("off")

    # Legend
    legend_handles = [
        Patch(color=np.array(c) / 255.0, label=id2label[i])
        for i, c in enumerate(label_colors)
    ]
    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
    )
    plt.tight_layout()

    # Output filename
    output_filename = f"{image_path.stem}_materials.png"
    output_path = Path.cwd() / output_filename
    plt.savefig(output_path)
    print(f"Saved overlay to {output_path}")


if __name__ == "__main__":
    main()
