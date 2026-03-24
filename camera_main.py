"""
main.py  —  Pipeline orchestrator
───────────────────────────────────
Wires all nodes together:

  Thread 1  →  CameraNode       (ZED grab)
  Thread 2  →  InferenceNode    (YOLO GPU)
  Thread ROS→  ROS2Spin         (rclpy executor)
  Main      →  FusionNode       (ZED 3D + publish)

Press 'q' in the display window to stop cleanly.
"""

import queue
import threading
import sys

from camera_node          import CameraNode
from camera_inference_node       import InferenceNode
from camera_fusion_node          import FusionNode
from camera_publisher  import ROS2DetectionPublisher


def main():
    # ── Shared queues ────────────────────────────────────────
    # Small maxsize = drop policy (never block the camera or GPU)
    frame_queue     = queue.Queue(maxsize=2)
    detection_queue = queue.Queue(maxsize=4)
    stop_event      = threading.Event()

    # ── ROS 2 publisher (start first so node is ready) ───────
    publisher = ROS2DetectionPublisher(frame_id='zed_left_camera')
    publisher.start()

    # ── Camera node ──────────────────────────────────────────
    camera_node = CameraNode(frame_queue, stop_event)
    print('[Main] Setting up ZED camera...')
    camera_node.setup()          # Opens ZED — raises if it fails

    # ── Inference node ───────────────────────────────────────
    inference_node = InferenceNode(
        frame_queue     = frame_queue,
        detection_queue = detection_queue,
        stop_event      = stop_event,
        model_path      = 'yolov8n.pt',   # swap to yolov8s/m for more accuracy
        conf            = 0.5,
    )

    # ── Fusion node ──────────────────────────────────────────
    fusion_node = FusionNode(
        detection_queue = detection_queue,
        zed             = camera_node.get_zed(),
        stop_event      = stop_event,
        publisher       = publisher,       # ← ROS 2 bridge injected here
    )

    # ── Start threads ────────────────────────────────────────
    print('[Main] Starting pipeline...')
    camera_node.start()
    inference_node.start()

    print('[Main] Pipeline running.')
    print('       Topics:')
    print('         ros2 topic echo /zed_yolo/detections')
    print('         ros2 topic echo /zed_yolo/detections_raw')
    print('       Press q in the display window to stop.')

    # ── FusionNode blocks on main thread ─────────────────────
    try:
        fusion_node.run()
    except KeyboardInterrupt:
        print('\n[Main] Ctrl+C received.')
        stop_event.set()

    # ── Shutdown ─────────────────────────────────────────────
    print('[Main] Shutting down...')
    stop_event.set()
    camera_node.shutdown()
    camera_node.join(timeout=3)
    inference_node.join(timeout=3)
    publisher.stop()
    print('[Main] Done.')


if __name__ == '__main__':
    main()