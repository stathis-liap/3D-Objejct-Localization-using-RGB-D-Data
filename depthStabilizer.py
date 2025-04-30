import numpy as np

class DepthStabilizer:
    """
    A class to stabilize depth values using a moving average filter.
    The class maintains a buffer of the last N depth values and computes the average.
    The oldest value is removed when a new value is added, ensuring the buffer size remains constant.
    """
    def __init__(self, buffer_size=5):
        self.buffer = []
        self.buffer_size = buffer_size
    
    
    def add_and_average(self, value):
        """
        Add a new depth value to the buffer and return the average.
        If the buffer exceeds the specified size, the oldest value is removed.

        Args:
            value (float): The new depth value to add.

        Returns:
            float: The average of the values in the buffer.
        """
        self.buffer.append(value)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return np.mean(self.buffer)
