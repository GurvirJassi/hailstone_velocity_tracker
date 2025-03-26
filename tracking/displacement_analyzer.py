import numpy as np
from filterpy.kalman import KalmanFilter

class VelocityCalculator:
    def __init__(self, fps=30, pixel_to_meter_ratio=0.01):
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter_ratio
        self.tracks = {}
        
    def update_track(self, track_id, position_3d, timestamp):
        if track_id not in self.tracks:
            self.tracks[track_id] = {'positions': [], 'timestamps': []}
            
        track = self.tracks[track_id]
        track['positions'].append(position_3d)
        track['timestamps'].append(timestamp)
        
        if len(track['positions']) > 5:
            track['positions'].pop(0)
            track['timestamps'].pop(0)
            
    def calculate_velocity(self, track_id):
        if track_id not in self.tracks or len(self.tracks[track_id]['positions']) < 2:
            return None
            
        positions = np.array(self.tracks[track_id]['positions'])
        timestamps = np.array(self.tracks[track_id]['timestamps'])
        
        displacements = positions[1:] - positions[:-1]
        time_deltas = timestamps[1:] - timestamps[:-1]
        time_deltas[time_deltas == 0] = 1/self.fps
        
        velocities = displacements / time_deltas[:, None]
        return np.mean(velocities, axis=0)