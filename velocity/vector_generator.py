import numpy as np
import time
from collections import deque

class StereoVelocityGenerator:
    def __init__(self, Q_matrix, fps=30, window_size=5):
        """
        Q_matrix: Stereo rectification matrix from cv2.stereoRectify()
        fps: Expected frame rate for fallback calculations
        window_size: Number of frames to average velocity over
        """
        self.Q = Q_matrix
        self.fps = fps
        self.window_size = window_size
        self.tracks = {}  # {track_id: {'positions': deque, 'timestamps': deque}}

    def triangulate(self, point_left, point_right):
        """Convert matched 2D points to 3D coordinates"""
        point_left = np.array([point_left[0], point_left[1]], dtype=float)
        point_right = np.array([point_right[0], point_right[1]], dtype=float)
        
        disparity = abs(point_left[0] - point_right[0])
        
        homog_point = np.ones(4)
        homog_point[0] = point_left[0]
        homog_point[1] = point_left[1]
        homog_point[2] = disparity
        
        point_3d = np.dot(self.Q, homog_point)
        point_3d /= point_3d[3]
        return point_3d[:3]  # Return x,y,z

    def update_track(self, track_id, point_left, point_right, timestamp=None):
        """Update track with new stereo measurement"""
        position = self.triangulate(point_left, point_right)
        
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'positions': deque(maxlen=self.window_size),
                'timestamps': deque(maxlen=self.window_size)
            }
        
        self.tracks[track_id]['positions'].append(position)
        self.tracks[track_id]['timestamps'].append(timestamp or time.time())

    def calculate_velocity(self, track_id):
        """Calculate 3D velocity in m/s (returns None if insufficient data)"""
        track = self.tracks.get(track_id)
        if not track or len(track['positions']) < 2:
            return None

        positions = np.array(track['positions'])
        timestamps = np.array(track['timestamps'])
        
        # Calculate displacement and time differences
        displacements = positions[1:] - positions[:-1]
        time_deltas = timestamps[1:] - timestamps[:-1]
        
        # Handle invalid time deltas
        time_deltas = np.maximum(time_deltas, 1/self.fps)
        
        # Velocity = displacement / time
        velocities = displacements / time_deltas[:, np.newaxis]
        
        return np.mean(velocities, axis=0)  # Average over window