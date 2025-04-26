import cv2
import numpy as np

def clip_mask_to_box(mask, x1, y1, x2, y2):
    mask_clipped = np.zeros_like(mask)
    mask_clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return mask_clipped

def calculate_average_depth(depth_frame, mask_clipped):
    masked_depth = depth_frame[mask_clipped]
    return np.mean(masked_depth)

def normalize_depth_frame(depth_frame):
    depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

def find_mask_center(mask):
    ys, xs = np.where(mask)
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return cx, cy
