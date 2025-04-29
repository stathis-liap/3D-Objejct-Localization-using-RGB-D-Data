import numpy as np

class Calculate_Coordinates:
    def __init__(self, width=640, height=480):
        self.x1 = 0
        self.y1 = 0
        self.z1 = 0
        self.width = width
        self.height = height
        self.length = 170 # cm
        self.pose_data = []
        #print("Coordinate calculation initialized.")

    def camera_corection(self, x, y, z):
        #corrects the coordinates based on the camera's field of view and
        #aspect ratio to ensure accurate distance measurements
        correction_factor = self.length / self.width
        x_corrected = x * correction_factor
        y_corrected = y * correction_factor
        z_corrected = z * correction_factor
        # returns the corrected coordinates in the cartesian system x, y, z
        return x_corrected, y_corrected, z_corrected

    def calculate_distance(self, x1, y1, z1, x2, y2, z2):
        X1_m, Y1_m, Z1_m = self.camera_corection(x1, y1, z1/1000)
        X2_m, Y2_m, Z2_m = self.camera_corection(x2, y2, z2/1000)
        # Compute 3D Euclidean distance
        dist = np.sqrt((X1_m - X2_m) ** 2 + (Y1_m - Y2_m) ** 2 + (Z1_m - Z2_m) ** 2)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        return dist, mid_x, mid_y
    
    def calculate_object_center(self, mask):
        # Calculate the center of the object represented by the mask
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
            return center_x, center_y
        return None, None



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


                                    
        
if __name__ == "__main__":
    # Example usage
    calc = Calculate_Coordinates()
    print(calc.transform_camera_to_world(1, 2, 3, 1, 'data/02.pose'))
    print(calc.transform_camera_to_world(1, 2, 3, 2, 'data/02.pose'))
    print(calc.transform_camera_to_world(1, 2, 3, 3, 'data/02.pose'))
    print(calc.transform_camera_to_world(1, 2, 3, 4, 'data/02.pose'))