import numpy as np

class Calculate_Coordinates:
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.z1 = 0
        print("Coordinate calculation initialized.")

    
    def get_coordinates(self, x1, y1, z1, x2, y2, z2):
        #print("Coordinates:")
        r1 = np.sqrt(self.x1**2 + self.y1**2)
        theta1 = np.arctan2(self.y1, self.x1)
        z1 = self.z1

        r2 = np.sqrt(x2**2 + y2**2)
        theta2 = np.arctan2(y2, x2)
        z2 = z2

        r = [r1, r2]
        theta = [theta1, theta2]
        z = [z1, z2]
        #print(f"r: {r}, theta: {theta}, z: {z}")
        return r, theta, z
    



    def calculate_distance(self, x1, y1, z1, x2, y2, z2):
    
        X1_m = x1
        self.x1 = x1
        Y1_m = y1
        self.y1 = y1
        Z1_m = z1 / 1000.0
        self.z1 = z1 / 1000.0

        X2_m = x2
        Y2_m = y2
        Z2_m = z2 / 1000.0

        # Compute 3D Euclidean distance
        dist = np.sqrt((X1_m - X2_m) ** 2 + (Y1_m - Y2_m) ** 2 + (Z1_m - Z2_m) ** 2)/ 1000.0

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
    
    