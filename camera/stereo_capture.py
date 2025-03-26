import cv2
import numpy as np
import os

class StereoCamera:
    def __init__(self, left_video_path, right_video_path):
        """
        Initialize stereo video file sources
        
        Args:
            left_video_path: Path to left video file
            right_video_path: Path to right video file
        """
        if not all(isinstance(p, str) for p in [left_video_path, right_video_path]):
            raise ValueError("Video paths must be strings")
            
        if not (os.path.exists(left_video_path) and os.path.exists(right_video_path)):
            raise FileNotFoundError("One or both video files not found")
            
        self.cap_left = cv2.VideoCapture(left_video_path)
        self.cap_right = cv2.VideoCapture(right_video_path)
        
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise ValueError("Could not open video files")

    def get_frames(self):
        """
        Get synchronized frames from both videos
        
        Returns:
            tuple: (left_frame, right_frame) or (None, None) at end of video
        """
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()
        
        # Check if either frame failed to read
        if not ret_left or not ret_right:
            return None, None
            
        return frame_left, frame_right
        
    def release(self):
        """Release video resources"""
        self.cap_left.release()
        self.cap_right.release()