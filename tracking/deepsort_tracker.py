from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)
        
    def track(self, detections, frame):
        # Convert detections to DeepSort format
        ds_detections = []
        for det in detections:
            bbox = det['bbox']
            ds_detections.append(([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], det['confidence'], det['class']))
            
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        tracked_objects = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            tracked_objects.append((track_id, [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]))
            
        return tracked_objects