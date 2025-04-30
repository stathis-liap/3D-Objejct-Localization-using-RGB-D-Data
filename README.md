# 3D Object Localization using RGB-D Data
A full pipeline for 3D object detection, segmentation, and localization using RGB-D data.
It processes a live feed from an RGB-D camera (or a recorded dataset) to identify objects, extract their silhouettes, and compute real-world coordinates based on depth information.

![pipeline drawio](https://github.com/user-attachments/assets/04e944a2-878d-4af4-b9b3-3a49f142726b)


# Features
 - Live or offline support: use a real camera or pre-recorded datasets.

 - YOLOv8 Object Detection: Fast and accurate object detection with bounding boxes.

 - SAM (Segment Anything Model): Precise object segmentation by generating masks from detected objects.

 - Depth-Based Distance Calculation:

 - Measure the real-world distance from the camera to each detected object.

 - Calculate the distance between multiple objects in 3D space.


# Pipeline Overview
## Input Handling:

 - Capture RGB-D frames either live (camera) or from dataset files.

## Object Detection (YOLO):
 - Detect objects of interest (e.g., cups, bowls, etc.).

 - Filter detections based on class and confidence thresholds.

## Object Segmentation (SAM):

 - Refine the detected regions by generating accurate masks.

 - Isolate the true silhouette of each object.

## Depth Processing:

 - For each object, locate its center using the mask.

 - Fetch the corresponding depth value (in mm).

## 3D Distance Calculation:

- Compute:

   - Distance between camera and object.

   - Distance between multiple objects.

 
## Visualization (GUI):

Show RGB images with bounding boxes and masks.

Show depth frames with object overlays.

Display distances interactively in a beautiful window.



# Technologies Used
Python 3.8+

YOLOv8 for real-time object detection

Segment Anything (SAM) for instance segmentation

OpenCV for frame handling and image processing

DearPyGUI for professional GUI design

Numpy for efficient numerical operations

# Repository Structure

        ├── main.py               # Main script 
        ├── configDepth           # Configure the Depth data
        ├── configYOLO.py         # Load YOLO models and config
        ├── configFastSAM.py      # Load SAM models and helper functions
        ├── coordinates.py        # Computes the World Coordinates
        ├── depthStabilizer.py    # Filters for the depth data
        ├── gui.py                # GUI 
        ├── inputFromCamera.py    # Handles camera or dataset input
        ├── object_tracker.py     # Tracks the trajectory of the object
        ├── pipeline.py           # Coordinates all the scripts
        ├── README.md             # This file
        ├── requirements.txt      # Required Python packages
        ├── data/
            └── meta data from the image
        ├── Fastsam-weights/
           └── pretrained weights for the fastSAM Model
        ├── yolo-Weights/
           └── pretrained weights for the YOLO Model
        ├── images/
           └── images used from the GUI
        └── scene02/
           └── RGB-D dataset
# Setup Instructions
## Clone the repository:

    git clone https://github.com/stathis-liap/3D-Objejct-Localization-using-RGB-D-Data
    cd 3D-Object-Localization-using-RGB-D-Data
    
## Install dependencies:

    pip install -r requirements.txt

## Install FastSAM Models weights
    https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt
    
## Run the main GUI pipeline:

    python main.py

        

🤝 Contributions
This project was a collaboration between:
Stathis Liapodimitris
Kostantinos
Kyriakopoylos
Stavros Stathopoulos

📜 License
This project is licensed under the MIT License
