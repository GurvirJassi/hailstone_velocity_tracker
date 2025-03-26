import cv2
import numpy as np

class Annotator:
    def __init__(self):
        self.colors = {
            'detection': (255, 0, 0),
            'tracking': (0, 255, 0),
            'velocity': (0, 0, 255)
        }
        
    def draw_3d_info(self, frame, bbox, position_3d, velocity):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['tracking'], 2)
        
        pos_text = f"Pos: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f})m"
        vel_text = f"Vel: ({velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f})m/s" if velocity is not None else "Vel: Calculating..."
        
        cv2.putText(frame, pos_text, (x1, y1 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['velocity'], 1)
        cv2.putText(frame, vel_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['velocity'], 1)
        
        return frame