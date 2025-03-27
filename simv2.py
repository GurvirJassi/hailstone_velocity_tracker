import cv2
import numpy as np
import time 
import math

class hail:

    # (position[m], velocity[m/s])
    def __init__(self, position: list, velocity: list):
        
        # save position and velocity vectors
        self.position = position
        self.start_pos = position
        self.velocity = velocity

        self.updateComponents()

        # velocity [m/s]
        self.v = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

        # time [s]
        self.t = 0
        
        self.radius = 0.04 # [m]
        self.color = (244, 255, 220)

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

        # update positions
        for c in range(3):
            self.position[c] = self.start_pos[c] + self.velocity[c] * t 
        
        # update time
        self.t = t
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

def simulate(hailstone: list[hail]):
    
    # scaling factor
    mperp = realWidth/camWidth
    pperm = 1/mperp

    t = 0  # Time variable
    paused = False
    running = True

   
    
    # Create a named window
    cv2.namedWindow('Simulation', cv2.WINDOW_KEEPRATIO)
    
    print("Controls:")
    print("Space: Pause/Resume")
    print("R: Restart simulation")
    print("Q or Esc: Quit")

    while running:
    # creating video frames
        left = np.zeros((camHeight, camWidth, 3), dtype=np.uint8)
        right = np.zeros((camHeight, camWidth, 3), dtype=np.uint8)

        separator_width = 5
        separator = np.full((left.shape[0], separator_width, 3), (255, 255, 255), dtype=np.uint8)
        if not paused:
            # Clear frame
            left[:] = 0
            right[:] = 0

            # iterate through hailstones
            for h in hailstones:
                h.moveTo(t)
                # draw on left
                cv2.circle(left, (int(h.y*pperm), int(h.z*pperm)), int(h.radius*pperm), h.color, 0)
                # draw on right
                cv2.circle(left, (camWidth-int(h.y*pperm), int(h.z*pperm)), int(h.radius*pperm), h.color, 0)
            
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
        key = cv2.waitKey(1)
        
        if key == ord(' '):  # Space toggles pause
            paused = not paused

        elif key == ord('r') or key == ord('R'):  # R restarts
            t = 0
            paused = False

        elif key == 27 or key == ord('q') or key == ord('Q'):  # Esc or Q quits
            running = False
        
        if paused:
            # Still process frames when paused to keep window responsive
            cv2.imshow('Simulation', combined)
            time.sleep(0.1)  # Reduce CPU usage when paused
    
    cv2.destroyAllWindows()




a = hail([0,0,0], [10, 10, 10])
hailstones = [a]

simulate(hailstones)