import cv2
import time  # Added this import
import numpy as np
import json  # Added for config loading
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from camera.stereo_capture import StereoCamera


class CameraCalibrator:
    def __init__(self, config_path):
        self.load_config(config_path)
        
    def load_config(self, path):
        with open(path) as f:
            params = json.load(f)
        
        # Convert to numpy arrays with proper dimensions
        self.camera_matrix1 = np.array(params['camera1']['matrix'], dtype=np.float64).reshape(3,3)
        self.dist_coeffs1 = np.array(params['camera1']['distortion'], dtype=np.float64).reshape(-1,1)
        self.camera_matrix2 = np.array(params['camera2']['matrix'], dtype=np.float64).reshape(3,3)
        self.dist_coeffs2 = np.array(params['camera2']['distortion'], dtype=np.float64).reshape(-1,1)
        self.R = np.array(params['rotation'], dtype=np.float64).reshape(3,3)
        self.T = np.array(params['translation'], dtype=np.float64).reshape(3,1)
        
    def rectify_images(self, img1, img2):
        # Validate input images
        if img1 is None or img2 is None:
            raise ValueError("Input images cannot be None")
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions")
            
        # Get image size in (width, height) format
        image_size = (img1.shape[1], img1.shape[0])
        
        # Perform stereo rectification with error handling
        try:
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                cameraMatrix1=self.camera_matrix1,
                distCoeffs1=self.dist_coeffs1,
                cameraMatrix2=self.camera_matrix2,
                distCoeffs2=self.dist_coeffs2,
                imageSize=image_size,
                R=self.R,
                T=self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0
            )
            
            # Create rectification maps
            map1x, map1y = cv2.initUndistortRectifyMap(
                self.camera_matrix1, self.dist_coeffs1, R1, P1,
                image_size, cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(
                self.camera_matrix2, self.dist_coeffs2, R2, P2,
                image_size, cv2.CV_32FC1)
            
            # Apply rectification
            img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
            img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
            
            return img1_rect, img2_rect, Q
            
        except Exception as e:
            raise RuntimeError(f"Stereo rectification failed: {str(e)}")

class HailDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.hail_class_id = 0  # Update this to your hail class ID
        self.min_confidence = 0.5
        self.pixel_to_meter = 0.001  # Conversion factor (adjust based on your setup)
        
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0].item()) == self.hail_class_id and box.conf[0].item() >= self.min_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': box.conf[0].item(),
                        'class': int(box.cls[0].item())
                    })
        return detections
        
class DeepSortTracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)
        
    def track(self, detections, frame):
        tracks = self.tracker.update_tracks(
            [([d['bbox'][0], d['bbox'][1], d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]], d['confidence'], d['class']) 
             for d in detections],
            frame=frame
        )
        return [(t.track_id, t.to_ltrb()) for t in tracks if t.is_confirmed()]

class Triangulator:
    def __init__(self, Q_matrix):
        self.Q = Q_matrix
        
    def triangulate(self, point_left, point_right):
        disparity = abs(point_left[0] - point_right[0])
        homog_point = np.array([point_left[0], point_left[1], disparity, 1.0])
        point_3d = np.dot(self.Q, homog_point)
        return point_3d[:3] / point_3d[3]  # Return x,y,z in meters

class HailstoneTracker:
    def __init__(self, config_path, video_sources=(0, 1)):
        self.calibrator = CameraCalibrator(config_path)
        self.camera = StereoCamera(*video_sources)
        self.detector = HailDetector()
        self.tracker_left = DeepSortTracker()
        self.tracker_right = DeepSortTracker()
        self.tracks_3d = defaultdict(lambda: {
            'positions': [],
            'timestamps': [],
            'velocity': None,
            'energy': None,
            'impact': False,
            'impact_registered': False
        })
        self.ground_z = 0.15  # Adjusted ground threshold (meters)
        self.hail_mass = 0.01  # 10g hail mass
        self.min_impact_speed = 1.5  # Reduced minimum speed (m/s)
        self.impact_data = []
        self.paused = False
        self.show_both_views = True

    def _get_center(self, bbox):
        return ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)

    def _match_tracks(self, tracks_left, tracks_right, max_y_diff=20, max_x_diff=50):
        matched_pairs = []
        used_right = set()
        
        for left_id, left_bbox in tracks_left:
            left_center = self._get_center(left_bbox)
            
            best_match = None
            best_diff = float('inf')
            
            for right_idx, (right_id, right_bbox) in enumerate(tracks_right):
                if right_idx in used_right:
                    continue
                    
                right_center = self._get_center(right_bbox)
                y_diff = abs(left_center[1] - right_center[1])
                x_diff = abs(left_center[0] - right_center[0])
                
                if y_diff < max_y_diff and x_diff < max_x_diff and x_diff < best_diff:
                    best_diff = x_diff
                    best_match = right_idx
                    
            if best_match is not None:
                matched_pairs.append(((left_id, left_bbox), tracks_right[best_match]))
                used_right.add(best_match)
                
        return matched_pairs

    def _update_track(self, track_id, position):
        track = self.tracks_3d[track_id]
        track['positions'].append(position)
        track['timestamps'].append(time.time())
        
        if len(track['positions']) >= 2:
            # Calculate velocity (m/s)
            disp = np.diff(track['positions'], axis=0)[-1]
            dt = track['timestamps'][-1] - track['timestamps'][-2]
            dt = max(dt, 1/30)  # Prevent division by zero
            track['velocity'] = disp / dt
            
            # Calculate kinetic energy (Joules)
            speed = np.linalg.norm(track['velocity'])
            track['energy'] = 0.5 * self.hail_mass * (speed ** 2)
            
            # Check for impact conditions
            if (position[2] <= self.ground_z and 
                not track['impact_registered'] and 
                speed >= self.min_impact_speed):
                
                track['impact'] = True
                track['impact_registered'] = True
                
                self.impact_data.append({
                    'id': track_id,
                    'position': position,
                    'velocity': track['velocity'].tolist(),
                    'speed': speed,
                    'energy': track['energy'],
                    'time': time.time()
                })
                print(f"IMPACT! ID: {track_id} Speed: {speed:.2f}m/s Energy: {track['energy']:.3f}J")

    def _draw_tracking_info(self, frame, bbox, track_id, color=(0, 255, 0)):
        track = self.tracks_3d[track_id]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw tracking info
        info = [
            f"ID: {track_id}",
            f"Vel: {np.linalg.norm(track['velocity']):.2f}m/s" if track['velocity'] is not None else "Vel: -",
            f"Z: {track['positions'][-1][2]:.2f}m" if len(track['positions']) > 0 else "Z: -"
        ]
        
        for i, text in enumerate(info):
            cv2.putText(frame, text, (x1, y1 - 30 + i*20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    def _save_impacts(self):
        if not self.impact_data:
            print("No impacts detected during this run")
            return
            
        with open('impacts.csv', 'w') as f:
            f.write("id,x,y,z,vx,vy,vz,speed,energy,time\n")
            for impact in self.impact_data:
                f.write(
                    f"{impact['id']},"
                    f"{impact['position'][0]},{impact['position'][1]},{impact['position'][2]},"
                    f"{impact['velocity'][0]},{impact['velocity'][1]},{impact['velocity'][2]},"
                    f"{impact['speed']},"
                    f"{impact['energy']},"
                    f"{impact['time']}\n"
                )
        print(f"Saved {len(self.impact_data)} impacts to impacts.csv")

    def run(self):
        try:
            while True:
                if not self.paused:
                    frames = self.camera.get_frames()
                    if frames[0] is None or frames[1] is None:
                        break
                        
                    frame_left, frame_right = frames
                    
                    try:
                        frame_left_rect, frame_right_rect, Q = self.calibrator.rectify_images(frame_left, frame_right)
                    except Exception as e:
                        print(f"Rectification error: {e}")
                        continue
                    
                    triangulator = Triangulator(Q)
                    
                    # Detection and tracking
                    detections_left = self.detector.detect(frame_left_rect)
                    detections_right = self.detector.detect(frame_right_rect)
                    
                    tracks_left = self.tracker_left.track(detections_left, frame_left_rect)
                    tracks_right = self.tracker_right.track(detections_right, frame_right_rect)
                    
                    # Process matched pairs
                    matched_pairs = self._match_tracks(tracks_left, tracks_right)
                    for (left_id, left_bbox), (right_id, right_bbox) in matched_pairs:
                        try:
                            position = triangulator.triangulate(
                                self._get_center(left_bbox), 
                                self._get_center(right_bbox)
                            )
                            self._update_track(left_id, position)
                            self._draw_tracking_info(frame_left_rect, left_bbox, left_id)
                            self._draw_tracking_info(frame_right_rect, right_bbox, right_id)
                        except Exception as e:
                            print(f"Tracking error: {e}")
                            continue
                
                # Display frames
                if self.show_both_views:
                    # Combine frames side by side
                    display_frame = np.hstack((frame_left_rect, frame_right_rect))
                else:
                    display_frame = frame_left_rect
                
                cv2.imshow("Hailstone Tracker", display_frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space to pause/resume
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")
                elif key == ord('v'):  # Toggle view
                    self.show_both_views = not self.show_both_views
                    
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            self._save_impacts()

if __name__ == "__main__":
    tracker = HailstoneTracker(
        config_path="config/camera_params.json",
        video_sources=(
            "test_videos/left.mp4",
            "test_videos/right.mp4"
        )
    )
    tracker.run()