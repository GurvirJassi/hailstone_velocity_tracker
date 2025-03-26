from ultralytics import YOLO

class HailDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.hail_class_id = 0  # Update this based on your custom model
        
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0].item()) == self.hail_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': self.hail_class_id
                    })
        return detections