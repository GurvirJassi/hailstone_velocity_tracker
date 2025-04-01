import cv2
import numpy as np
import csv
from collections import defaultdict
from ultralytics import YOLO
import time

# ===== SETTINGS =====
camHeight =  1920 # [pixels]
camWidth =  1080 # [pixels]
scale = 1
realHeight = 1920/1080 * scale # 1.0 # [m]
realWidth = 1 * scale # 1920/1080 # [m]
mperp = realWidth/camWidth
pperm = 1/mperp

VIDEO_PATH_1 = r"sim_videos\left_view.mp4"  # First perpendicular video path
VIDEO_PATH_2 = r"sim_videos\right_view.mp4"  # Second perpendicular video path
FPS = 30                     # Frames per second of both videos
PIXELS_PER_METER = 100       # Conversion factor (pixels/meter)
YOLO_MODEL = "yolov8n.pt"    # YOLO model file
OBJECT_CLASS = 0             # Class ID to track (0 is person in COCO)
DISPLAY_VIDEO = True         # Show video with tracking info
WINDOW_SCALE = 0.5           # Scale factor for display window
CSV_FILENAME = "velocities.csv"  # Output CSV file

# ===== MAIN APPLICATION =====
class ObjectTracker:
    def __init__(self):
        # Initialize YOLO model with verbose=False to suppress output
        self.model = YOLO(YOLO_MODEL)
        self.model.verbose = False
        
        # Tracking storage
        self.prev_positions = defaultdict(dict)
        self.velocities = defaultdict(dict)
        
        # CSV setup
        self.csv_file = open(CSV_FILENAME, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Time', 'ObjectID', 'Vx', 'Vy', 'Vz'])
        
    def __del__(self):
        self.csv_file.close()
        
    def process_videos(self):
        try:
            cap1 = cv2.VideoCapture(VIDEO_PATH_1)
            cap2 = cv2.VideoCapture(VIDEO_PATH_2)
            
            if not cap1.isOpened() or not cap2.isOpened():
                print(f"Error: Could not open video files. Check paths:\n{VIDEO_PATH_1}\n{VIDEO_PATH_2}")
                return
            
            start_time = time.time()
            
            while cap1.isOpened() and cap2.isOpened():
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                    
                current_time = time.time() - start_time
                
                # Detect objects in both views
                detections1 = self.detect_objects(frame1)
                detections2 = self.detect_objects(frame2)
                
                # Match objects between views and calculate velocities
                self.match_and_calculate(detections1, detections2, current_time)
                
                if DISPLAY_VIDEO:
                    self.display_results(frame1, frame2, current_time)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            self.csv_file.close()
            
    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)  # Disable YOLO output
        detections = []
        for box in results[0].boxes:
            if int(box.cls) == OBJECT_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'bbox': (x1, y1, x2, y2)
                })
        return detections
    
    def match_and_calculate(self, detections1, detections2, current_time):
        # Simple matching by closest position (for demo)
        for i, det1 in enumerate(detections1):
            for j, det2 in enumerate(detections2):
                obj_id = f"obj_{i}_{j}"  # Simple ID assignment
                
                # Current positions (x1, z1) and (y2, z2)
                x1, z1 = det1['center']
                y2, z2 = det2['center']
                
                # Store current position
                current_pos = {
                    'x': x1 / PIXELS_PER_METER,
                    'y': y2 / PIXELS_PER_METER,
                    'z': ((z1 + z2) / 2) / PIXELS_PER_METER,
                    'time': current_time
                }
                
                # Calculate velocity if we have previous position
                if obj_id in self.prev_positions:
                    prev = self.prev_positions[obj_id]
                    dt = current_pos['time'] - prev['time']
                    
                    if dt > 0:
                        vx = (current_pos['x'] - prev['x']) / dt
                        vy = (current_pos['y'] - prev['y']) / dt
                        vz = (current_pos['z'] - prev['z']) / dt
                        
                        self.velocities[obj_id] = {'vx': vx, 'vy': vy, 'vz': vz}
                        self.csv_writer.writerow([current_time, obj_id, vx, vy, vz])
                
                # Update previous position
                self.prev_positions[obj_id] = current_pos
    
    def display_results(self, frame1, frame2, current_time):
        # Resize frames
        h, w = frame1.shape[:2]
        new_size = (int(w * WINDOW_SCALE), int(h * WINDOW_SCALE))
        disp1 = cv2.resize(frame1, new_size)
        disp2 = cv2.resize(frame2, new_size)
        
        # Add timestamp
        timestamp = f"Time: {current_time:.2f}s"
        cv2.putText(disp1, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(disp2, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Combine frames
        combined = np.hstack((disp1, disp2))
        cv2.imshow("Hailstone Tracker", combined)
        cv2.resizeWindow("Hailstone Tracker", combined.shape[1], combined.shape[0])

if __name__ == "__main__":
    print("Starting hailstone velocity tracker...")
    print(f"Data will be saved to: {CSV_FILENAME}")
    tracker = ObjectTracker()
    tracker.process_videos()
    print("Tracking complete. CSV file saved.")