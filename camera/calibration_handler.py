import json
import cv2
import numpy as np

class CameraCalibrator:
    def __init__(self, config_path="config/camera_params.json"):
        self.load_config(config_path)
        
    def load_config(self, path):
        with open(path) as f:
            params = json.load(f)
        self.camera_matrix1 = np.array(params['camera1']['matrix'])
        self.dist_coeffs1 = np.array(params['camera1']['distortion'])
        self.camera_matrix2 = np.array(params['camera2']['matrix'])
        self.dist_coeffs2 = np.array(params['camera2']['distortion'])
        self.R = np.array(params['rotation'])
        self.T = np.array(params['translation'])
        
    def rectify_images(self, img1, img2):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.camera_matrix1, self.dist_coeffs1,
            self.camera_matrix2, self.dist_coeffs2,
            img1.shape[:2], self.R, self.T)
        
        map1x, map1y = cv2.initUndistortRectifyMap(
            self.camera_matrix1, self.dist_coeffs1, R1, P1,
            img1.shape[:2], cv2.CV_32FC1)
            
        map2x, map2y = cv2.initUndistortRectifyMap(
            self.camera_matrix2, self.dist_coeffs2, R2, P2,
            img2.shape[:2], cv2.CV_32FC1)
            
        img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        
        return img1_rect, img2_rect, Q