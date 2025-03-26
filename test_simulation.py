import cv2
import numpy as np
import os
import json
from collections import defaultdict

class HailSimulator:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.ground_truth = defaultdict(list)
        self.final_metrics = []
        self.baseline = 0.1  # 10cm between cameras
        self.focal_length = 1000   # Focal length in pixels
        self.ground_level = height - 50  # Ground position (pixels from top)
        
    def generate_motion_path(self, num_hailstones=1, max_frames=150):
        """Generate single hailstone falling from top center to bottom corner"""
        paths = []
        
        # Single hailstone parameters
        entry_frame = 0
        vx = 3  # Horizontal velocity
        vz = 6  # Vertical velocity
        start_x = self.width // 2
        start_z = 2.0  # Starting depth in meters
        
        path = {'left': [], 'right': [], 'impact_frame': None}
        
        for frame in range(max_frames):
            # Time since entry
            t = frame - entry_frame
            
            # World coordinates (meters)
            x_world = (start_x - self.width/2)/self.focal_length * start_z + self.width/2
            y_world = 50 + vz * t
            z_world = start_z  # Constant depth for linear fall
            
            # Apply motion
            x_world += vx * t
            
            # Check if hit ground
            if y_world >= self.ground_level and path['impact_frame'] is None:
                path['impact_frame'] = frame
                # Don't break - let object continue falling
                self.final_metrics.append({
                    'hail_id': 0,
                    'entry_frame': entry_frame,
                    'impact_frame': frame,
                    'final_velocity': (vx, vz, 0),
                    'impact_position': (x_world, y_world, z_world),
                })
                        
            # Left camera view (no shift)
            path['left'].append((x_world, y_world))
            
            # Right camera view (with proper disparity)
            x_right = x_world - (self.baseline * self.focal_length) / z_world
            path['right'].append((x_right, y_world))
            
            # Store ground truth
            self.ground_truth[0].append({
                'frame': frame,
                'position': (x_world, y_world, z_world),
                'velocity': (vx, vz, 0),
            })
        
        paths.append(path)
        return paths

    def create_test_videos(self, paths):
        """Generate stereo video pair with proper falling motion"""
        os.makedirs('test_videos', exist_ok=True)
        
        # Video writers
        left_writer = cv2.VideoWriter('test_videos/left.mp4', 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    30, (self.width, self.height))
        right_writer = cv2.VideoWriter('test_videos/right.mp4',
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     30, (self.width, self.height))
        
        max_frames = max(max(len(p['left']), len(p['right'])) for p in paths) if paths else 0
        
        for frame in range(max_frames):
            # Create black frames
            frame_left = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame_right = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw ground plane
            cv2.line(frame_left, (0, self.ground_level), 
                    (self.width, self.ground_level), (0, 255, 255), 2)
            cv2.line(frame_right, (0, self.ground_level), 
                    (self.width, self.ground_level), (0, 255, 255), 2)
            
            for path in paths:
                if frame < len(path['left']):
                    # Left camera - white circle
                    x_left, y_left = path['left'][frame]
                    cv2.circle(frame_left, (int(x_left), int(y_left)), 10, (255, 255, 255), -1)
                    
                if frame < len(path['right']):
                    # Right camera - white circle
                    x_right, y_right = path['right'][frame]
                    cv2.circle(frame_right, (int(x_right), int(y_right)), 10, (255, 255, 255), -1)
            
            left_writer.write(frame_left)
            right_writer.write(frame_right)
        
        left_writer.release()
        right_writer.release()
        
        # Save ground truth data
        with open('test_videos/ground_truth.json', 'w') as f:
            json.dump(self.ground_truth, f)
            
        # Save final metrics
        with open('test_videos/final_metrics.json', 'w') as f:
            json.dump(self.final_metrics, f)

        def create_test_videos(self, paths):
        
            # ... existing video creation code ...
            metadata = {
                'expected_impact_z': 0.15,
                'baseline': self.baseline,
                'frames': [{
                    'frame': i,
                    'left_pos': paths[0]['left'][i] if i < len(paths[0]['left']) else None,
                    'right_pos': paths[0]['right'][i] if i < len(paths[0]['right']) else None,
                    'world_pos': self.ground_truth[0][i]['position'] if i < len(self.ground_truth[0]) else None
                } for i in range(max_frames)]
            }
            with open('test_videos/simulation_metadata.json', 'w') as f:
                json.dump(metadata, f)

if __name__ == "__main__":
    print("Generating test video with single hailstone...")
    sim = HailSimulator()
    paths = sim.generate_motion_path(num_hailstones=1, max_frames=200)
    sim.create_test_videos(paths)
    print("Test videos generated successfully")