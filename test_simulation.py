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
        self.baseline_pixels = 20  # Horizontal distance between cameras in pixels
        self.focal_length = 1000   # Focal length in pixels
        self.ground_level = height - 50  # Ground position (pixels from top)
        
    def generate_motion_path(self, num_hailstones=5, max_frames=150):
        """Generate falling hailstones with varying velocities and entry times"""
        paths = []
        
        for hail_id in range(num_hailstones):
            # Randomize parameters for each hailstone
            entry_frame = np.random.randint(0, 50)  # Staggered entry
            vx = np.random.uniform(-5, 5)  # Horizontal velocity
            vz = np.random.uniform(3, 8)    # Vertical velocity (fall speed)
            start_x = np.random.randint(100, self.width-100)
            start_z = np.random.uniform(80, 120)  # Starting depth
            
            path = {'left': [], 'right': [], 'impact_frame': None}
            
            for frame in range(max_frames):
                if frame < entry_frame:
                    # Not visible yet
                    continue
                    
                # Time since entry
                t = frame - entry_frame
                
                # World coordinates (left camera perspective)
                x_world = start_x + vx * t
                y_world = 50 + vz * t  # Falling from top
                z_world = start_z  # Constant depth for linear fall
                
                # Check if hit ground
                if y_world >= self.ground_level:
                    path['impact_frame'] = frame
                    # Record final metrics before breaking
                    velocity = (vx, vz, 0)
                    mass = 0.01  # 10g hailstone
                    energy = 0.5 * mass * (vx**2 + vz**2)
                    self.final_metrics.append({
                        'hail_id': hail_id,
                        'entry_frame': entry_frame,
                        'impact_frame': frame,
                        'final_velocity': velocity,
                        'kinetic_energy': energy,
                        'impact_position': (x_world, y_world, z_world)
                    })
                    break
                
                # Left camera view
                path['left'].append((x_world, y_world))
                
                # Right camera view (proper disparity)
                x_right = x_world - (self.baseline_pixels * self.focal_length) / z_world
                path['right'].append((x_world, y_world))  # Note: Using same y for proper fall
                
                # Store ground truth
                self.ground_truth[hail_id].append({
                    'frame': frame,
                    'position': (x_world, y_world, z_world),
                    'velocity': (vx, vz, 0)
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
            frame_left = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame_right = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw ground plane
            cv2.line(frame_left, (0, self.ground_level), 
                    (self.width, self.ground_level), (0, 255, 255), 2)
            cv2.line(frame_right, (0, self.ground_level), 
                    (self.width, self.ground_level), (0, 255, 255), 2)
            
            for path in paths:
                if frame < len(path['left']):
                    # Left camera
                    x_left, y_left = path['left'][frame]
                    cv2.circle(frame_left, (int(x_left), int(y_left)), 8, (0, 255, 0), -1)
                    
                if frame < len(path['right']):
                    # Right camera
                    x_right, y_right = path['right'][frame]
                    cv2.circle(frame_right, (int(x_right), int(y_right)), 8, (0, 255, 0), -1)
            
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

def create_test_config():
    """Generate test camera configuration"""
    config = {
        "camera1": {
            "matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0]
        },
        "camera2": {
            "matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0]
        },
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "translation": [-0.1, 0, 0]  # 10cm baseline
    }
    with open("test_config.json", "w") as f:
        json.dump(config, f)

if __name__ == "__main__":
    print("Generating realistic hailstone simulation...")
    
    # Create test configuration
    create_test_config()
    
    # Generate test videos
    sim = HailSimulator()
    paths = sim.generate_motion_path(num_hailstones=5, max_frames=200)
    sim.create_test_videos(paths)
    
    print("Successfully generated:")
    print("- test_videos/left.mp4 (left camera view)")
    print("- test_videos/right.mp4 (right camera view)")
    print("- test_videos/ground_truth.json (full trajectory data)")
    print("- test_videos/final_metrics.json (impact velocities & energies)")
    print("- test_config.json (camera calibration)")