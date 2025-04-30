import cv2
import numpy as np
import csv
from collections import defaultdict
import os

# ===== SETTINGS =====
# Camera settings
CAM_WIDTH = 1080   # [pixels]
CAM_HEIGHT = 1920  # [pixels]
SCALE = 1
REAL_WIDTH = 1.0 * SCALE  # [m]
REAL_HEIGHT = (1920/1080) * SCALE  # [m]
M_PER_PIXEL = REAL_WIDTH / CAM_WIDTH

# Video settings
VIDEO_PATH_1 = os.path.join("videos", "left_view.mp4")
VIDEO_PATH_2 = os.path.join("videos", "right_view.mp4")
FPS = 30
DISPLAY_VIDEO = True
WINDOW_SCALE = 0.5  # Reduce this if window is too large
CSV_FILENAME = "hailstone_velocities.csv"

# Detection settings
DETECTION_PARAMS = {
    'blur': 1,             # Odd number (3,5,7,...)
    'thresh': 180,         # 0-255
    'min_area': 1,        # Minimum blob area
    'max_area': 5000,      # Maximum blob area
    'min_circularity': 0.1 # 0-1
}

# ===== MAIN APPLICATION =====
class HailstoneTracker:
    def __init__(self):
        self.tracks = defaultdict(dict)
        self.frame_count = 0
        self.setup_csv()
        self.setup_windows()
        
    def setup_csv(self):
        self.csv_file = open(CSV_FILENAME, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['Time(s)', 'Frame', 'ID', 'X(m)', 'Y(m)', 'Z(m)', 
                            'Vx(m/s)', 'Vy(m/s)', 'Vz(m/s)', 'Diameter(m)'])
        
    def setup_windows(self):
        if DISPLAY_VIDEO:
            cv2.namedWindow("Hailstone Tracker", cv2.WINDOW_NORMAL)
            window_width = int(CAM_WIDTH * 2 * WINDOW_SCALE)
            window_height = int(CAM_HEIGHT * WINDOW_SCALE)
            cv2.resizeWindow("Hailstone Tracker", window_width, window_height)
    
    def run(self):
        # Initialize video captures
        cap1 = cv2.VideoCapture(VIDEO_PATH_1)
        cap2 = cv2.VideoCapture(VIDEO_PATH_2)
        
        # Verify video files
        if not os.path.exists(VIDEO_PATH_1):
            print(f"ERROR: Left video not found at {os.path.abspath(VIDEO_PATH_1)}")
            return
        if not os.path.exists(VIDEO_PATH_2):
            print(f"ERROR: Right video not found at {os.path.abspath(VIDEO_PATH_2)}")
            return
            
        if not cap1.isOpened():
            print("ERROR: Could not open left video")
            return
        if not cap2.isOpened():
            print("ERROR: Could not open right video")
            return
            
        print("=== Hailstone Tracking ===")
        print("Controls: 1/2=Blur | 3/4=Thresh | 5/6=MinArea | 7/8=MaxArea | 9/0=Circ | q=Quit")
        print("Visual feedback:")
        print("- Green circles around detections")
        print("- Red center dots")
        print("- Diameter labels in meters")
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            # Loop videos if they end
            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if not ret2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Process frame
            time = self.frame_count / FPS
            self.frame_count += 1
            
            # Detect hailstones
            circles1 = self.detect_hailstones(frame1)
            circles2 = self.detect_hailstones(frame2)
            
            # Track and calculate velocities
            for i, c1 in enumerate(circles1):
                for j, c2 in enumerate(circles2):
                    obj_id = f"hail-{i}-{j}"
                    x = c1['pos'][0] * M_PER_PIXEL
                    y = c2['pos'][0] * M_PER_PIXEL
                    z = ((c1['pos'][1] + c2['pos'][1])/2) * M_PER_PIXEL
                    diameter = (c1['diameter'] + c2['diameter'])/2
                    
                    current = {'x':x, 'y':y, 'z':z, 'time':time, 'd':diameter}
                    
                    if obj_id in self.tracks:
                        prev = self.tracks[obj_id]
                        dt = time - prev['time']
                        if dt > 0:
                            vx = (current['x'] - prev['x'])/dt
                            vy = (current['y'] - prev['y'])/dt
                            vz = (current['z'] - prev['z'])/dt
                            
                            self.writer.writerow([
                                time, self.frame_count, obj_id,
                                x, y, z, vx, vy, vz, diameter
                            ])
                    
                    self.tracks[obj_id] = current
            
            # Display results
            if DISPLAY_VIDEO:
                self.display_frames(frame1, frame2, circles1, circles2, time)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                self.adjust_parameters(key)
                
        # Cleanup
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        self.csv_file.close()
        print(f"\nData saved to {CSV_FILENAME}")
    
    def detect_hailstones(self, frame):
        """Detect circular objects in frame"""
        blur_size = max(3, DETECTION_PARAMS['blur'])
        blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        _, thresh = cv2.threshold(blurred, DETECTION_PARAMS['thresh'], 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (DETECTION_PARAMS['min_area'] <= area <= DETECTION_PARAMS['max_area']):
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < DETECTION_PARAMS['min_circularity']:
                continue
                
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            detections.append({
                'pos': (int(x), int(y)),
                'radius': int(radius),
                'diameter': 2 * radius * M_PER_PIXEL
            })
            
        return detections
    
    def display_frames(self, frame1, frame2, circles1, circles2, time):
        """Display frames with detections side-by-side"""
        disp1 = frame1.copy()
        disp2 = frame2.copy()
        
        # Draw detections
        for c in circles1:
            cv2.circle(disp1, c['pos'], c['radius'], (0, 255, 0), 3)
            cv2.circle(disp1, c['pos'], 3, (0, 0, 255), -1)
            cv2.putText(disp1, f"{c['diameter']:.2f}m", 
                       (c['pos'][0]+15, c['pos'][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
        for c in circles2:
            cv2.circle(disp2, c['pos'], c['radius'], (0, 255, 0), 3)
            cv2.circle(disp2, c['pos'], 3, (0, 0, 255), -1)
            cv2.putText(disp2, f"{c['diameter']:.2f}m", 
                       (c['pos'][0]+15, c['pos'][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Add labels
        cv2.putText(disp1, "LEFT VIEW", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(disp2, "RIGHT VIEW", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add frame info
        info_text = f"Frame: {self.frame_count} | Time: {time:.2f}s | Detections: {len(circles1)}"
        cv2.putText(disp1, info_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Combine side-by-side
        combined = np.hstack((disp1, disp2))
        
        # Add parameter info
        param_text = f"Blur: {DETECTION_PARAMS['blur']} (1/2) | Thresh: {DETECTION_PARAMS['thresh']} (3/4) | "
        param_text += f"MinA: {DETECTION_PARAMS['min_area']} (5/6) | MaxA: {DETECTION_PARAMS['max_area']} (7/8) | "
        param_text += f"Circ: {DETECTION_PARAMS['min_circularity']:.2f} (9/0)"
        
        text_x = 20
        cv2.putText(combined, param_text, (text_x, combined.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        cv2.imshow("Hailstone Tracker", combined)
    
    def adjust_parameters(self, key):
        """Adjust detection parameters with keyboard"""
        if key == ord('1'): DETECTION_PARAMS['blur'] = max(3, DETECTION_PARAMS['blur']-2)
        elif key == ord('2'): DETECTION_PARAMS['blur'] += 2
        elif key == ord('3'): DETECTION_PARAMS['thresh'] = max(0, DETECTION_PARAMS['thresh']-5)
        elif key == ord('4'): DETECTION_PARAMS['thresh'] = min(255, DETECTION_PARAMS['thresh']+5)
        elif key == ord('5'): DETECTION_PARAMS['min_area'] = max(10, DETECTION_PARAMS['min_area']-10)
        elif key == ord('6'): DETECTION_PARAMS['min_area'] += 10
        elif key == ord('7'): DETECTION_PARAMS['max_area'] = max(100, DETECTION_PARAMS['max_area']-100)
        elif key == ord('8'): DETECTION_PARAMS['max_area'] += 100
        elif key == ord('9'): DETECTION_PARAMS['min_circularity'] = max(0.1, DETECTION_PARAMS['min_circularity']-0.05)
        elif key == ord('0'): DETECTION_PARAMS['min_circularity'] = min(1.0, DETECTION_PARAMS['min_circularity']+0.05)

if __name__ == "__main__":
    tracker = HailstoneTracker()
    tracker.run()