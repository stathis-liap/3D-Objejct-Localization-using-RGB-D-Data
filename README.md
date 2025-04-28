# 🛰️ 3D Object Localization using RGB-D Data
A full pipeline for 3D object detection, segmentation, and localization using RGB-D data.
It processes a live feed from an RGB-D camera (or a recorded dataset) to identify objects, extract their silhouettes, and compute real-world distances based on depth information.

# ✨ Features
 - Live or offline support: use a real camera or pre-recorded datasets.

 - YOLOv8 Object Detection: Fast and accurate object detection with bounding boxes.

 - SAM (Segment Anything Model): Precise object segmentation by generating masks from detected objects.

 - Depth-Based Distance Calculation:

 - Measure the real-world distance from the camera to each detected object.

 - Calculate the distance between multiple objects in 3D space.


# 🛠️ Pipeline Overview
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



# 🔧 Technologies Used
Python 3.8+

YOLOv8 for real-time object detection

Segment Anything (SAM) for instance segmentation

OpenCV for frame handling and image processing

DearPyGUI for professional GUI design

Numpy for efficient numerical operations

# 📚 Repository Structure

        ├── main.py               # Main script (basic mode)
        ├── main_v2.py            # Main script (GUI-enabled)
        ├── configYOLO.py         # Load YOLO models and config
        ├── configFastSAM.py      # Load SAM models and helper functions
        ├── inputFromCamera.py    # Handles camera or dataset input
        ├── README.md             # This file
        ├── requirements.txt      # Required Python packages
        └── utils/
            └── helpers.py        # (optional) Helper functions
# ⚙️ Setup Instructions
## Clone the repository:

    git clone https://github.com/stathis-liap/3D-Objejct-Localization-using-RGB-D-Data
    cd 3D-Object-Localization-using-RGB-D-Data
    
## Install dependencies:

    pip install -r requirements.txt
    
## Run the main GUI pipeline:

    python main_v2.py

        

🤝 Contributions
This project was a coluboration between:
Stathis Liapodimitris
Kostantinos
Kyriakopoylos
Stavros Stathopoulos

📜 License
This project is licensed under the MIT License
