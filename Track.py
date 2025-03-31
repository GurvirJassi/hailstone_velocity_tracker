import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def detect_hailstones(frame):
    processed = preprocess_frame(frame)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hailstones = []
    seen = set()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, radius = int(x), int(y), int(radius)
            if (x, y, radius) not in seen:
                seen.add((x, y, radius))
                hailstones.append((x, y, radius, area))
    
    return hailstones

def track_motion(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def log_hailstone_data(frame_id, detections, file):
    with open(file, "a") as txtfile:
        for x, y, radius, _ in detections:
            txtfile.write(f"Frame {frame_id}: X={x}, Y={y}, Radius={radius}\n")

def main(path):
    cap = cv2.VideoCapture(path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    #Clear hailsontData
    with open("hailstoneData.txt", "w") as f:
        f.write("")  # Overwrite with an empty string
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('tracked_hailstones.avi', fourcc, 30.0, (prev_frame.shape[1], prev_frame.shape[0]))
    
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = track_motion(prev_gray, curr_gray)
        detected_hailstones = detect_hailstones(frame)
        log_hailstone_data(frame_id, detected_hailstones, "hailstoneData.txt")
        
        for (x, y, radius, _) in detected_hailstones:
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.putText(frame, f"({x}, {y}), {radius}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        output_video.write(frame)
        cv2.imshow('Hailstone Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_gray = curr_gray
        frame_id += 1
    
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("sim_videos\left_view.mp4")
    
    import time
    time.sleep(2)

    from SortDetections import main
    main()
    
