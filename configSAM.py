import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2

# πρέπει να το βάλεις "cuda" αν έχετε nvidia κάρτα
DEVICE = "cpu"
SAM_CHECKPOINT_PATH = "sam-weights/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b" 

def load_sam_model():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    return predictor

# Get mask from cropped image
def get_silhouette(predictor, image, box):
    input_box = np.array([box], dtype=np.float32)

    # Set image for predictor
    predictor.set_image(image)

    # Predict mask
    masks, scores, _ = predictor.predict(
        box=input_box,
        multimask_output=False
    )

    # Return the best mask
    return masks[0]  
