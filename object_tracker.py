# object_tracker.py
class ObjectTracker:
    '''
    A simple object tracker that uses a dictionary to store the last known coordinates of objects.
    The coordinates are smoothed using an exponential moving average.
    The smoothing factor (alpha) determines how much weight is given to the new coordinates versus the old ones.
    '''
    def __init__(self, alpha=0.2):
        self.last_coords = {}  # key: object label or frame id, value: [x, y, z]
        self.alpha = alpha

    def smooth(self, object_id, coords):
        '''
        Smooth the coordinates of an object using exponential moving average.

        Args:
            object_id (int): Unique identifier for the object.
            coords (list): New coordinates of the object [x, y, z].

        Returns:
            list: Smoothed coordinates of the object [x, y, z].
        '''
        if object_id not in self.last_coords:
            self.last_coords[object_id] = coords
        else:
            prev = self.last_coords[object_id]
            self.last_coords[object_id] = [
                self.alpha * new + (1 - self.alpha) * old
                for new, old in zip(coords, prev)
            ]
        return self.last_coords[object_id]
