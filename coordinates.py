import numpy as np

class Calculate_Coordinates:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.length = 170 # cm
        self.pose_data = []

    def transform_camera_to_world(self, u, v, depth, frame_idx, fx=525, fy=525, cx=319.5, cy=239.5, scale_factor=1.0, pose_file='data/02.pose'):
        import numpy as np
        from scipy.spatial.transform import Rotation as R


        homogenousPixelVector = np.array([u/depth, v/depth, 1])
        K = np.zeros((3,3))
        K[0,0], K[1,1], K[2,2], K[0,2], K[1,2] = fx, fy, 1.0, cx, cy
        K_inv = np.linalg.inv(K)
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
        rot_inv = np.linalg.inv(rot) 
        t = np.array([tx, ty, tz])
        
        # Build the full 4x4 transformation matrix
        
        scaledPixel = np.dot(scale_factor, homogenousPixelVector)
        correctedFromCameraParameters = np.dot(scaledPixel, K_inv)
        correctedFromCameraPosition = np.subtract(correctedFromCameraParameters, t)
        worldCords = np.dot(correctedFromCameraPosition, rot_inv)

        return worldCords