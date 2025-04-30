import cv2
import numpy as np

def clip_mask_to_box(mask, x1, y1, x2, y2):
    """
    Clip the mask to the bounding box defined by (x1, y1) and (x2, y2).
    This ensures that the mask does not extend beyond the specified coordinates.

    Args:

        mask (numpy.ndarray): The mask to be clipped.
        x1 (int): The x-coordinate of the top-left corner of the bounding box.
        y1 (int): The y-coordinate of the top-left corner of the bounding box.
        x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (int): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
        numpy.ndarray: The clipped mask.
    """
    mask_clipped = np.zeros_like(mask)
    # ensure coordinates are within frame bounds
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(mask.shape[0], y2)
    x2 = min(mask.shape[1], x2)
    mask_clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return mask_clipped

def calculate_average_depth(depth_frame, mask_clipped):
    """
    Calculate the average depth from the depth frame using the provided mask.

    The average depth is computed only for the pixels where the mask is greater than 0.
    This is useful for obtaining the average depth for the whole object.

    Args:
        depth_frame (numpy.ndarray): The depth frame from which to calculate the average depth.
        mask_clipped (numpy.ndarray): The mask used to filter the depth frame.

    Returns:
        float: The average depth value for the masked area.
    """
    masked_depth = depth_frame[mask_clipped > 0]
    return np.median(masked_depth)

def normalize_depth_frame(depth_frame):
    """
    Normalize the depth frame for visualization.

    This function scales the depth values to the range [0, 255] for display purposes.
    The depth values are converted to an 8-bit unsigned integer format.

    Args:

        depth_frame (numpy.ndarray): The depth frame to be normalized.

    Returns:
        numpy.ndarray: The normalized depth frame suitable for visualization.
    """
    depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

def find_mask_center(mask):
    """
    Find the center of the mask by calculating the mean of the x and y coordinates of the non-zero pixels.

    This function returns the center coordinates (cx, cy) of the mask.

    Args:
        mask (numpy.ndarray): The mask for which to find the center.

    Returns:
        tuple: The center coordinates (cx, cy) of the mask.
    """
    ys, xs = np.where(mask)
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return cx, cy

#masked_depth = depth_frame[(mask_clipped > 0) & (mask_clipped < max_length)]