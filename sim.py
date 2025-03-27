import cv2
import numpy as np
import time 



class Coordinate:
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"

    def move(self, dx: int, dy: int, dz: int) -> None:
        self.x += dx
        self.y += dy
        self.z += dz


size = 500

# Starting coordinates
p1 = Coordinate(0, 0, 0)

frame = np.zeros((size, size, 3), dtype=np.uint8)

end_t = 10
end_f = 1000

fps = 30

spf = 1/30 # second per frame

window_name = "front view"

# Create window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

f = 0
while f <= end_f:
    
    # Clear frame
    frame[:] = 0

    # Draw circle
    cv2.circle(frame,(int(p1.x*size/10), int(p1.z*size/10)), 10, (0, 255, 0), -1)

    # Update Coordinates
    p1.move(0.2, 0.1, 0.4)

    # Display
    cv2.imshow(window_name, frame)
     
    time.sleep(spf)
    f += 1

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("r"):
        p1 = Coordinate(0, 0, 0)
        f = 0

    # Auto-break if object leaves frame
        #if p1.x > size or p1.z > size:
         #   print("Object left frame")
          #  break

    
#cv2.destroyAllWindows()

# taskkill /f /im python.exe







# Create a window
#cv2.namedWindow('Live Video', cv2.WINDOW_NORMAL)

# Video properties
#width, height = 640, 480
#fps = 1


# Main loop to generate and display frames
#frame_count = 0
#while True:

    
    #radius = 30
    #color = (0, 255, 0)  # Green in BGR format
    #thickness = 2

    # Draw the circle
    #cv2.circle(frame, (center_x, center_y), radius, color, thickness)
    
python simv2.py
