# inference_node.py
import threading
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

class InferenceNode(threading.Thread):
    def __init__(self, frame_queue, detection_queue, stop_event, model_path='yolov8n.pt'):
        super().__init__(daemon=True, name="InferenceNode")
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.stop_event = stop_event
        self.model_path = model_path

    def setup(self):
        # Load model here (inside thread) so GPU context belongs to this thread
        self.model = YOLO(self.model_path)
        # Warm-up pass — eliminates first-frame latency spike
        import numpy as np
        self.model.predict(np.zeros((720, 1280, 3), dtype=np.uint8), verbose=False)

    def _format_boxes(self, results) -> list:
        custom_boxes = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                tmp = sl.CustomBoxObjectData()
                tmp.unique_object_id = sl.generate_unique_id()
                tmp.probability = float(box.conf)
                tmp.label = int(box.cls)
                tmp.bounding_box_2d = np.array([
                    [coords[0], coords[1]],
                    [coords[2], coords[1]],
                    [coords[2], coords[3]],
                    [coords[0], coords[3]]
                ], dtype=np.float32)
                custom_boxes.append(tmp)
        return custom_boxes

    def run(self):
        self.setup()
        while not self.stop_event.is_set():
            try:
                frame, img_rgb = self.frame_queue.get(timeout=0.5)
            except Exception:
                continue

            results = self.model.predict(img_rgb, conf=0.5, verbose=False)
            custom_boxes = self._format_boxes(results)

            # Drop if downstream is backed up — prefer fresh frames
            if not self.detection_queue.full():
                self.detection_queue.put_nowait((frame, custom_boxes))