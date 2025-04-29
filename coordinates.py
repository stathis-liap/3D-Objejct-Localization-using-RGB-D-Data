import numpy as np

class Calculate_Coordinates:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.length = 170 # cm
        self.pose_data = []

    def transform_camera_to_world(self, u, v, depth, frame_idx, fx=525, fy=525, cx=319.5, cy=319.5, scale_factor=1.0, pose_file='data/02.pose'):
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        # Convert pixel coordinates to camera coordinates
        z = depth * scale_factor  # Convert depth to meters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    
        # Read the pose for the given frame
        with open(pose_file, 'r') as f:
            lines = f.readlines()

        if frame_idx >= len(lines):
            raise IndexError("Frame index out of range.")

        pose_values = list(map(float, lines[frame_idx].strip().split()))
        if len(pose_values) != 7:
            raise ValueError(f"Expected 7 values in pose, got {len(pose_values)}")

        qx, qy, qz, qw, tx, ty, tz = pose_values
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Build the full 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [tx, ty, tz]

        # Transform the point
        p_cam = np.array([x, y, z, 1])
        p_world = T @ p_cam
        return p_world[:3]
