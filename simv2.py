import cv2
import random
import numpy as np
import time 
import math
import copy

class hail:

    # (position[m], velocity[m/s])
    def __init__(self,
                position: list,
                velocity: list,
                radius: float = 0.02, # [m]
                color: tuple = (244, 255, 220)):
        
        # save position and velocity vectors
        self.start_pos = list(position)
        self.position = list(position)
        self.velocity = list(velocity)
        self.radius = float(radius)
        self.color = tuple(color)

        self.updateComponents()

        # velocity [m/s]
        self.v = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

        # time [s]
        self.t = 0

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
    def fallFor(self, t):
        
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

    # returns refreshed copy of this instance
    def copy(self):
        return hail(self.start_pos, self.velocity, self.radius, self.color)

# python simv2.py

# camera/video settings
camHeight = 1080 # [pixels]
camWidth = 1920 # [pixels]
fps = 120
spf = 1/fps

# physical space settings
realHeight = 1.0 # [m]
realWidth = 1920/1080 # [m]

def simulate(hailstones: list[hail]):
    
    og = [copy.deepcopy(h) for h in hailstones]

    # Statements always run but put away for code cleanliness
    if True:

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

    #creating gradient frame
    ratio = np.linspace(0, 1, camWidth).reshape(1, -1, 1)
    frame = (0,0,230) * (1 - ratio) + (0,200,170) * ratio
    frame = frame.astype(np.uint8)

    while running:
        
        if not paused:

            # Clear frame
            left[:] = frame
            right[:] = frame

            # iterate through hailstones - hailstones is dynamic and is updated through n, m
            for h in hailstones:
                #h.moveTo(t) # critical # old
                h.fallFor(spf)
                # debug: print(t, h.position)
                # draw on left
                cv2.circle(left, (int(h.y*pperm), int(h.z*pperm)), int(h.radius*pperm), h.color, -1)
                # draw on right
                cv2.circle(right, (camWidth-int(h.x*pperm), int(h.z*pperm)), int(h.radius*pperm), h.color, -1)
                if h.z>realHeight*1.05: 
                    hailstones.remove(h)
                    print ("item removed")

            #debug
            '''try: 
                print (hailstones[0].z) #debug
            finally:
                
                print("Nothing in the tank")'''

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
            simulate(og)
            break
            #print("Simulation restarted")

        elif key == 27 or key == ord('q') or key == ord('Q'):
            running = False
            #print(f"Simulation ended at time {t:.2f}s")

        elif key == ord('n') or key == ord('N'):
            #for h in og:
            #    h.reset()
            simulate(generate_hailstones(len(og)))
            break

        elif key == ord('m') or key == ord('M'):
            hailstones += generate_hailstones(len(og))

        if paused:
            cv2.imshow('Simulation', combined)
            time.sleep(0.1)
    
    cv2.destroyAllWindows()

def generate_hailstones(num_hailstones: int, 
                       position_range: tuple = (0, realWidth),
                       sideways_velocity_range: tuple = (0.01, 4),
                       vertical_velocity_range: tuple = (3, 10),
                       radius_range: tuple = (0.00025,0.03),
                       color_variance: int = 30) -> list[hail]:
    """
    Generates multiple hailstone objects with randomized properties
    
    Args:
        num_hailstones: Number of hailstones to generate
        position_range: (min, max) for x,y,z starting positions (meters)
        velocity_range: (min, max) for velocity components (m/s)
        radius_range: (min, max) for hailstone radii (meters)
        color_variance: Max deviation from base whitish color (0-255)
    
    Returns:
        List of hail objects
    """
    hailstones = []
    base_color = (220, 230, 240)  # Base whitish color
    
    for _ in range(num_hailstones):
        # Random position within range
        pos = [random.uniform(*position_range), random.uniform(*position_range), 0]

        # Random velocity within range (all positive)
        vel = [(random.uniform(*sideways_velocity_range)* -1 if random.random() < 0.5 else 1),
               (random.uniform(*sideways_velocity_range)* -1 if random.random() < 0.5 else 1),
                random.uniform(*vertical_velocity_range)] # z component must be positive
        
        # Random radius within range
        radius = random.uniform(*radius_range)
        
        # Slightly randomized whitish color
        color = tuple(
            min(255, max(200, base_color[i] + random.randint(-color_variance, color_variance)))
            for i in range(3)
        )
        
        # Create hailstone with these properties
        hailstones.append(hail(position=pos, velocity=vel, radius=radius, color=color))

        print(vel[2])
    
    return hailstones

# antiquities
a = hail([0,0,0], [2, 7, 8])
b = hail([0.5,0.5,0.04],[10,10,10])
hailstones = [a, b]

#def slideshow(interval: tuple =  (0,5)):




simulate(generate_hailstones(15))