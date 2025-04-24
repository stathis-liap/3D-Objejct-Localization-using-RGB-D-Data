import torch
import numpy as np
import cv2
from fastsam import FastSAM, FastSAMPrompt

DEVICE = "cpu"  # valte "cuda" an exete GPU Nvidia
FASTSAM_CHECKPOINT_PATH = "fastsam-weights/FastSAM-x.pt"

# load FastSAM with weights
def load_sam_model():
    model = FastSAM(FASTSAM_CHECKPOINT_PATH)
    return model

# mask from cropped YOLO image
def get_silhouette(model, image, box, padding=0):
    h, w = image.shape[:2]
    
    # Apply padding to the bounding box
    x1, y1, x2, y2 = map(int, box)
    
    # Adjust the box coordinates based on padding, ensuring they stay within image bounds
    x1 = max(0, min(x1 - padding, w - 1))
    y1 = max(0, min(y1 - padding, h - 1))
    x2 = max(0, min(x2 + padding, w))
    y2 = max(0, min(y2 + padding, h))

    # Crop the image based on the padded bounding box
    cropped = image[y1:y2, x1:x2]
    
    # Run inference without the box parameter
    results = model(
        cropped,
        device=DEVICE,
        retina_masks=True,
        imgsz=256,
        conf=0.4,
        iou=0.9,
    )

    # Pass cropped image and results to FastSAMPrompt (without the box argument)
    processor = FastSAMPrompt(cropped, results, device=DEVICE)
    masks = processor.everything_prompt()

    print("Masks type:", type(masks))
    print("Masks shape:", getattr(masks, "shape", "no shape"))

    if len(masks) == 0:
        print("FastSAM found no masks.")
        return np.zeros(image.shape[:2], dtype=bool)

    masks = np.array(masks)
    if masks.ndim != 3:
        print("Unexpected mask shape:", masks.shape)
        return np.zeros(image.shape[:2], dtype=bool)

    # Compute box center
    box_cx = (x2 - x1) // 2
    box_cy = (y2 - y1) // 2

    best_mask_idx = None
    best_score = float("inf")

    for i, mask in enumerate(masks):
        # Compute mask center
        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        
        mask_cx = int(np.mean(xs))
        mask_cy = int(np.mean(ys))

        # Score is Euclidean distance from box center
        dist = np.sqrt((mask_cx - box_cx) ** 2 + (mask_cy - box_cy) ** 2)

        if dist < best_score:
            best_score = dist
            best_mask_idx = i

    # Fallback to largest mask if nothing is centered
    if best_mask_idx is None:
        print("No good center mask found. Falling back to largest mask.")
        best_mask_idx = np.argmax([np.sum(m) for m in masks])

    selected_mask = masks[best_mask_idx]
    resized_mask = cv2.resize(selected_mask.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = resized_mask

    print("Selected mask. Nonzero pixels:", np.sum(full_mask))
    return full_mask.astype(bool)
