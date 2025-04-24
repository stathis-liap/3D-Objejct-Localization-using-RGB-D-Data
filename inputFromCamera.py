import cv2
import numpy as np
import os

class InputFromCamera:
    def __init__(self, use_webcam=True, dataset_path=None):
        """
        Depending on input source (webcam or dataset).
        
        -> use_webcam: if webcam feed is used (True) or dataset (False).
        -> dataset_path: path to dataset (only needed if use_webcam is False).
        """
        self.use_webcam = use_webcam
        self.dataset_path = dataset_path

        if self.use_webcam:
            self.cap = cv2.VideoCapture(0) 
            self.cap.set(3, 640)  # width
            self.cap.set(4, 480)  # height
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
        else:
            if not os.path.exists(self.dataset_path):
                raise ValueError(f"Dataset path '{self.dataset_path}' does not exist.")
            self.rgb_files = sorted([f for f in os.listdir(self.dataset_path) if f.endswith('-color.png')]) # etsi ta evala prosorina giati 
            self.depth_files = sorted([f for f in os.listdir(self.dataset_path) if f.endswith('-depth.png')]) # den kserw thn morfi twn arxeiwn
            if not self.rgb_files or not self.depth_files:
                raise ValueError("No RGB or Depth files found in the dataset.")

            self.dataset_idx = 0  # track current frame

    def get_frame(self):
        """
        Get the current frame (either from the webcam or dataset).
        
        :return: RGB and Depth frames as numpy arrays.
        """
        if self.use_webcam:
            success, rgb_frame = self.cap.read()
            if not success:
                raise RuntimeError("Error: Could not read from webcam.")
            # afou sto webcam den exoume depth aplws epistrefei mhdenika
            depth_frame = np.zeros_like(rgb_frame)
            return rgb_frame, depth_frame
        else:
            # RGBD data from dataset
            if self.dataset_idx >= len(self.rgb_files):
                raise IndexError("End of dataset reached.")
            
            rgb_path = os.path.join(self.dataset_path, self.rgb_files[self.dataset_idx])
            depth_path = os.path.join(self.dataset_path, self.depth_files[self.dataset_idx])

            rgb_frame = cv2.imread(rgb_path)
            depth_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Depth image in single channel (den eimai sigouros gia auto alla etsi to eixe o typas pou to phra)

            # Ensure RGB is in BGR format (default for OpenCV)
            if rgb_frame is None or depth_frame is None:
                raise RuntimeError(f"Failed to load RGB or Depth frame from dataset at index {self.dataset_idx}.")

            self.dataset_idx += 1
            return rgb_frame, depth_frame

    def release(self):
        """
        Release resources (for webcam).
        """
        if self.use_webcam:
            self.cap.release()

