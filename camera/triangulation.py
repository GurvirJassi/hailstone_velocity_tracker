import numpy as np

class Triangulator:
    def __init__(self, Q_matrix):
        self.Q = Q_matrix
        
    def triangulate(self, point_left, point_right):
        point_left = np.array([point_left[0], point_left[1]], dtype=float)
        point_right = np.array([point_right[0], point_right[1]], dtype=float)
        
        disparity = abs(point_left[0] - point_right[0])
        
        homog_point = np.ones(4)
        homog_point[0] = point_left[0]
        homog_point[1] = point_left[1]
        homog_point[2] = disparity
        
        point_3d = np.dot(self.Q, homog_point)
        point_3d /= point_3d[3]
        
        return point_3d[:3]