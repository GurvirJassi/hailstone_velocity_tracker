import numpy as np
import csv
import time
from datetime import datetime

class DataLogger:
    def __init__(self, filename="hail_data.csv"):
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp', 'track_id', 'x_pos', 'y_pos', 'z_pos',
            'x_vel', 'y_vel', 'z_vel', 'speed'
        ])
        
    def log(self, track_id, position, velocity, timestamp):
        if position is None:
            return
            
        speed = np.linalg.norm(velocity) if velocity is not None else 0
        human_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
        
        self.writer.writerow([
            human_time, track_id,
            position[0], position[1], position[2],
            velocity[0] if velocity is not None else 0,
            velocity[1] if velocity is not None else 0,
            velocity[2] if velocity is not None else 0,
            speed
        ])
        
    def close(self):
        self.file.close()