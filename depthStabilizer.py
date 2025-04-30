import numpy as np

class DepthStabilizer:
    def __init__(self, buffer_size=5):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add_and_average(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return np.mean(self.buffer)
