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
def get_silhouette(model, image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    cropped = image[y1:y2, x1:x2]

    results = model(
        cropped,
        device=DEVICE,
        retina_masks=True,
        imgsz=256,
        conf=0.4,
        iou=0.9,
    )

    # edw den eimai sigouros gia kapoia an ta xreiazomaste aplws ta phra copy paste, tha to checkaroume sto mellon mallon
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

    combined_mask = np.any(masks, axis=0)
    resized_mask = cv2.resize(combined_mask.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = resized_mask

    print("Full mask created. Nonzero pixels:", np.sum(full_mask))
    return full_mask.astype(bool)

