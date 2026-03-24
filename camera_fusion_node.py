"""
fusion_node.py  —  Node 3 (Main thread)
─────────────────────────────────────────
Ingests YOLO boxes into ZED SDK to get real 3D positions,
builds a DetectionFrame, hands it to the ROS 2 publisher.
Runs on main thread (required by ZED SDK + cv2.imshow).
"""

import cv2
import numpy as np
import pyzed.sl as sl

from detection_data import DetectedObject, DetectionFrame, COCO_NAMES


PALETTE = [
    (0, 255, 100), (0, 180, 255), (255, 100, 0),
    (255, 0, 180), (180, 255, 0), (0, 255, 255),
]


class FusionNode:
    def __init__(self, detection_queue, zed, stop_event, publisher=None):
        """
        publisher : ROS2DetectionPublisher instance (or None for testing).
                    Any object with a .publish(DetectionFrame) method works.
        """
        self.detection_queue   = detection_queue
        self.zed               = zed
        self.stop_event        = stop_event
        self.publisher         = publisher
        self.objects           = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.frame_id          = 0

    def run(self):
        import queue
        while not self.stop_event.is_set():
            try:
                frame, custom_boxes = self.detection_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── ZED SDK: ingest boxes → get real 3D positions ──
            self.zed.ingest_custom_box_objects(custom_boxes)
            self.zed.retrieve_objects(self.objects, self.obj_runtime_param)

            # ── Build clean DetectionFrame ──────────────────────
            det_frame = DetectionFrame(frame_id=self.frame_id)

            for obj in self.objects.object_list:
                pos  = obj.position          # [x, y, z] metres
                name = (COCO_NAMES[obj.raw_label]
                        if obj.raw_label < len(COCO_NAMES)
                        else str(obj.raw_label))

                det_frame.objects.append(DetectedObject(
                    obj_id     = obj.id,
                    label      = obj.raw_label,
                    class_name = name,
                    confidence = 1.0,
                    x          = float(pos[0]),
                    y          = float(pos[1]),
                    z          = float(pos[2]),
                    bbox_2d    = (obj.bounding_box_2d.tolist()
                                  if obj.bounding_box_2d is not None else []),
                ))

                # Draw on frame
                if obj.bounding_box_2d is not None:
                    color = PALETTE[obj.raw_label % len(PALETTE)]
                    pts   = obj.bounding_box_2d.astype(np.int32)
                    cv2.polylines(frame, [pts], True, color, 2)
                    label_txt = f"{name}  Z:{pos[2]:.1f}m"
                    x1, y1    = pts[0]
                    (tw, th), _ = cv2.getTextSize(
                        label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame,
                                  (x1, y1 - th - 8), (x1 + tw + 4, y1),
                                  color, -1)
                    cv2.putText(frame, label_txt, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 0, 0), 1, cv2.LINE_AA)

            # ── Publish to ROS 2 ────────────────────────────────
            if self.publisher is not None:
                self.publisher.publish(det_frame)

            self.frame_id += 1

            # ── Display ─────────────────────────────────────────
            cv2.imshow("ZED 2i + YOLOv8", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        cv2.destroyAllWindows()