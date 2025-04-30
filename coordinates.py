import numpy as np
from scipy.spatial.transform import Rotation as R

class Calculate_Coordinates:
    """
    A class to calculate 3D coordinates from pixel coordinates and depth values.
    The class uses a pinhole camera model to convert pixel coordinates to camera coordinates,
    and then transforms these coordinates to world coordinates using a given pose.
    """
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.length = 170 # cm
        self.pose_data = []

    def transform_camera_to_world(self, u, v, depth, frame_idx, fx=1050, fy=1050, cx=319.5, cy=239.5, scale_factor=1.0, pose_file='data/02.pose'):
        '''
        Convert pixel coordinates (u, v) and depth to world coordinates using the camera pose.
        
        Args:
            u (int): Pixel x-coordinate.
            v (int): Pixel y-coordinate.
            depth (float): Depth value in pixels.
            frame_idx (int): Index of the frame for which the pose is to be used.
            fx (float): Focal length in x direction.
            fy (float): Focal length in y direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.
            scale_factor (float): Scale factor for depth conversion to meters.
            pose_file (str): Path to the file containing camera poses.

        Returns:
            np.ndarray: 3D world coordinates (x, y, z) in meters.
        '''

        # Convert pixel coordinates to camera coordinates
        z = depth * scale_factor  # Convert depth to meters
        x = (u - cx) * z / fx    # Convert pixel X coordinates to camera coordinates
        y = (v - cy) * z / fy   # Convert pixel Y coordinates to camera coordinates

        # Read the pose for the given frame
        with open(pose_file, 'r') as f:
            lines = f.readlines()

        if frame_idx >= len(lines):
            raise IndexError("Frame index out of range.")

        pose_values = list(map(float, lines[frame_idx].strip().split()))
        if len(pose_values) != 7:
            raise ValueError(f"Expected 7 values in pose, got {len(pose_values)}")

        # Calculate the rotation matrix from quaternion
        # Pose values: qx, qy, qz, qw, tx, ty, tz
        qx, qy, qz, qw, tx, ty, tz = pose_values
        rot = R.from_quat([-qx, qy, qz, qw]).as_matrix()

        # Build the full 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [tx, ty, tz]

        # Invert to get camera → world transformation
        T_inv = np.linalg.inv(T)

        # Transform the point
        p_cam = np.array([x, y, z, 1])
        p_world = T_inv @ p_cam

        return p_world[:3]

