# object_tracker.py
class ObjectTracker:
    def __init__(self, alpha=0.2):
        self.last_coords = {}  # key: object label or frame id, value: [x, y, z]
        self.alpha = alpha

    def smooth(self, object_id, coords):
        if object_id not in self.last_coords:
            self.last_coords[object_id] = coords
        else:
            prev = self.last_coords[object_id]
            self.last_coords[object_id] = [
                self.alpha * new + (1 - self.alpha) * old
                for new, old in zip(coords, prev)
            ]
        return self.last_coords[object_id]
