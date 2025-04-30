import torch
from ultralytics import YOLO
import torch
import cv2
import random

# Patch torch.load to always use weights_only=False αλλιώς να βάλετε το PiTorch σε version μικρότερο του 2.6 (μου έβγαλε την πίστη αυτό εδώ πέρα)
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = patched_load

# load class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# random color for each class
COLORS = [tuple(random.randint(0, 255) for _ in range(3)) for _ in classNames]

# settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2

# load YOLO model with weights
def load_yolo_model(YOLO_CHECKPOINT_PATH):
    return YOLO(YOLO_CHECKPOINT_PATH)
