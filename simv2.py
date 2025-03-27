import cv2
import numpy as np
import time 
import math

class hail:

    # (position[m], velocity[m/s])
    def __init__(self, position: list, velocity: list):
        
        # save position and velocity vectors
        self.start_pos = list(position)
        self.position = list(position)
        self.velocity = velocity

        self.updateComponents()

        # velocity [m/s]
        self.v = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

        # time [s]
        self.t = 0
        
        self.radius = 0.02 # [m]
        self.color = (255, 255, 255)

    # update accessor variables
    def updateComponents(self):

        # position components [m]
        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]

        # velocity components [m/s]
        self.vx = self.velocity[0]
        self.vy = self.velocity[1]
        self.vz = self.velocity[2]

    # increment position by time [s]
    def fallfor(self, t):
        
        # update positions
        for c in range(3):
            self.position[c] += self.velocity[c] * t 
        
        # update time
        self.t += t
        self.updateComponents()

    # move to position at time [s]
    def moveTo(self, t):
        
        self.t = t
        # update positions
        for c in range(3):
            self.position[c] = self.start_pos[c] + self.velocity[c] * self.t 
        
        # update time
        self.updateComponents()

    # reset hail
    def reset(self):

        self.position = self.start_pos
        self.t = 0
        self.updateComponents()

# python simv2.py

# camera/video settings
camHeight = 1080 # [pixels]
camWidth = 1920 # [pixels]
fps = 30
spf = 1/fps

# physical space settings
realHeight = 1.0 # [m]
realWidth = 1920/1080 # [m]

def simulate(hailstones: list):
    
    # scaling factor
    mperp = realWidth/camWidth
    pperm = 1/mperp

    t = 0  # Time variable
    paused = False
    running = True

    # creating video frames
    left = np.zeros((camHeight, camWidth, 3), dtype=np.uint8)
    right = np.zeros((camHeight, camWidth, 3), dtype=np.uint8)

    separator_width = 5
    separator = np.full((left.shape[0], separator_width, 3), (255, 255, 255), dtype=np.uint8)

   # Calculate combined window size
    window_width = camWidth * 2 + separator_width
    window_height = camHeight
    
    # Create a named window with fixed size
    cv2.namedWindow('Simulation', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Simulation', int(window_width/2), int(window_height/2))
    cv2.setWindowProperty('Simulation', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    
    #print("Controls:")
    #print("Space: Pause/Resume")
    #print("R: Restart simulation")
    #print("Q or Esc: Quit")

    while running:
        
        if not paused:
            # Clear frame
            left[:] = 0
            right[:] = 0

            # iterate through hailstones
            for h in hailstones:
                h.moveTo(t) # critical
                # debug: print(t, h.position)
                # draw on left
                cv2.circle(left, (int(h.y*pperm), int(h.z*pperm)), int(h.radius*pperm), h.color, -1)
                # draw on right
                cv2.circle(right, (camWidth-int(h.x*pperm), int(h.z*pperm)), int(h.radius*pperm), h.color, -1)
            
            # Combine frames
            combined = np.hstack((left, separator, right))

            # Display status text
            status = "Running" if not paused else "Paused"
            cv2.putText(combined, f"Status: {status} | Time: {t:.2f}s", (60, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            
            cv2.imshow('Simulation', combined)

            time.sleep(spf)
            t += spf

       # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            paused = not paused
            #print(f"Simulation {'paused' if paused else 'resumed'} at time {t:.2f}s")
        elif key == ord('r') or key == ord('R'):
            t = 0
            #print("Simulation restarted")
        elif key == 27 or key == ord('q') or key == ord('Q'):
            running = False
            #print(f"Simulation ended at time {t:.2f}s")
        
        if paused:
            cv2.imshow('Simulation', combined)
            time.sleep(0.1)
    
    cv2.destroyAllWindows()


a = hail([0,0,0], [2, 7, 8])
b = hail([0.5,0.5,0.04],[0,0,1])
hailstones = [a, b]

print (a.v, b.v)
simulate(hailstones)