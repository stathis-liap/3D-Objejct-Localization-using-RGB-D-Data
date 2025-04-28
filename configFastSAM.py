import torch
import numpy as np
import cv2
from fastsam import FastSAM, FastSAMPrompt

DEVICE = "cpu"  # "cuda" if you have Nvidia GPU
FASTSAM_CHECKPOINT_PATH = "fastsam-weights/FastSAM-x.pt"

def load_sam_model():
    return FastSAM(FASTSAM_CHECKPOINT_PATH)

def get_silhouette(model, image, box, padding=0):
    h, w = image.shape[:2]
    
    x1, y1, x2, y2 = map(int, box)

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    cropped = image[y1:y2, x1:x2]

    # Run FastSAM on cropped image
    results = model(
        cropped,
        device=DEVICE,
        retina_masks=True,
        imgsz=256,
        conf=0.4,
        iou=0.9,
    )

    processor = FastSAMPrompt(cropped, results, device=DEVICE)
    masks = processor.everything_prompt()

    if masks is None or len(masks) == 0:
        print("FastSAM found no masks.")
        return np.zeros((h, w), dtype=bool)

    masks = np.array(masks)

    # Center of cropped box
    box_cx = (x2 - x1) // 2
    box_cy = (y2 - y1) // 2

    best_mask_idx = None
    best_score = float("inf")

    for i, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue

        mask_cx = int(np.mean(xs))
        mask_cy = int(np.mean(ys))

        dist = np.sqrt((mask_cx - box_cx) ** 2 + (mask_cy - box_cy) ** 2)

        if dist < best_score:
            best_score = dist
            best_mask_idx = i

    selected_mask = masks[best_mask_idx]

    # resize to match the correct frame
    resized_mask = cv2.resize(selected_mask.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    # place resized mask to full image
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = resized_mask

    #print(f"Selected mask. Nonzero pixels: {np.sum(full_mask)}")
    return full_mask.astype(bool)
