
"""
ros2_publisher_node.py  —  NEW NODE (ROS 2 bridge)
────────────────────────────────────────────────────
Converts DetectionFrame objects from the pipeline
into ROS 2 messages and publishes them on:
 
  /zed_yolo/detections      → vision_msgs/Detection3DArray
                               (standard, works with any ROS 2 tool)
 
  /zed_yolo/detections_raw  → std_msgs/String  (JSON)
                               (easy to consume from any language)
 
Runs inside the same process as the pipeline (not a separate ROS node).
Call publisher.publish(frame) from FusionNode — thread-safe.
 
Requirements:
  sudo apt install ros-humble-vision-msgs   (or your ROS distro)
  pip install rclpy
"""
 
import json
import threading
import time
from dataclasses import asdict
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
 
# Standard ROS 2 message types
from std_msgs.msg import String, Header
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Vector3, Pose
from vision_msgs.msg import (
    Detection3DArray,
    Detection3D,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray
 
from detection_data import DetectionFrame, COCO_NAMES
 
 
# ─────────────────────────────────────────────────────────────────
# Internal ROS 2 Node class  (spun in its own thread)
# ─────────────────────────────────────────────────────────────────
 
class _ROS2Node(Node):
    """
    The actual rclpy Node. Owns all publishers.
    Do NOT call this directly — use ROS2DetectionPublisher below.
    """
 
    def __init__(self):
        super().__init__('zed_yolo_detection_publisher')
 
        # ── Publisher 1: vision_msgs/Detection3DArray ───────────
        # Standard format — works with RViz2, nav2, any ROS 2 node
        self.pub_detections = self.create_publisher(
            Detection3DArray,
            '/zed_yolo/detections',
            qos_profile=10
        )
 
        # ── Publisher 2: JSON string ─────────────────────────────
        # Easy to subscribe from Python, C++, or any language
        self.pub_json = self.create_publisher(
            String,
            '/zed_yolo/detections_raw',
            qos_profile=10
        )
 
        # ── Publisher 3: RViz2 Marker array ─────────────────────
        # 3D bounding boxes visible directly in RViz2
        self.pub_markers = self.create_publisher(
            MarkerArray,
            '/zed_yolo/markers',
            qos_profile=10
        )
 
        self.get_logger().info(
            'ZED YOLO publisher node started.\n'
            '  Topics:\n'
            '    /zed_yolo/detections      (vision_msgs/Detection3DArray)\n'
            '    /zed_yolo/detections_raw  (std_msgs/String  JSON)\n'
            '    /zed_yolo/markers         (visualization_msgs/MarkerArray)'
        )
 
 
# ─────────────────────────────────────────────────────────────────
# Public interface — use this in your pipeline
# ─────────────────────────────────────────────────────────────────
 
class ROS2DetectionPublisher:
    """
    Thread-safe bridge between your pipeline threads and ROS 2.
 
    Usage in main.py:
        publisher = ROS2DetectionPublisher()
        publisher.start()
        ...
        fusion_node = FusionNode(..., publisher=publisher)
        ...
        publisher.stop()
    """
 
    def __init__(self, frame_id: str = 'zed_left_camera'):
        self._frame_id = frame_id
        self._node     = None
        self._executor = None
        self._thread   = None
        self._lock     = threading.Lock()
        self._started  = False
 
    # ── Lifecycle ─────────────────────────────────────────────
 
    def start(self):
        """Initialize ROS 2 and start the spin thread."""
        if not rclpy.ok():
            rclpy.init()
 
        self._node     = _ROS2Node()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
 
        self._thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name='ROS2Spin'
        )
        self._thread.start()
        self._started = True
        print('[ROS2DetectionPublisher] Started.')
 
    def stop(self):
        """Shut down ROS 2 cleanly."""
        if self._executor:
            self._executor.shutdown()
        if self._node:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print('[ROS2DetectionPublisher] Stopped.')
 
    # ── Main publish method (called from FusionNode) ──────────
 
    def publish(self, frame: DetectionFrame):
        """
        Convert DetectionFrame → ROS 2 messages and publish.
        Thread-safe. Called from the main/fusion thread.
        """
        if not self._started or self._node is None:
            return
 
        stamp = self._to_ros_time(frame.timestamp)
        header = Header()
        header.stamp    = stamp
        header.frame_id = self._frame_id
 
        with self._lock:
            self._publish_detection3d_array(header, frame)
            self._publish_json(header, frame)
            self._publish_markers(header, frame)
 
    # ── vision_msgs/Detection3DArray ──────────────────────────
 
    def _publish_detection3d_array(self, header, frame: DetectionFrame):
        msg             = Detection3DArray()
        msg.header      = header
 
        for obj in frame.objects:
            det = Detection3D()
            det.header = header
 
            # 3D bounding box centre (ZED gives us the centroid)
            det.bbox.center.position.x = obj.x
            det.bbox.center.position.y = obj.y
            det.bbox.center.position.z = obj.z
 
            # ZED doesn't give us box size directly —
            # estimate from 2D bbox area and depth (rough but useful)
            if obj.bbox_2d and len(obj.bbox_2d) == 4:
                w_px = abs(obj.bbox_2d[1][0] - obj.bbox_2d[0][0])
                h_px = abs(obj.bbox_2d[3][1] - obj.bbox_2d[0][1])
                # Convert pixels to metres using pinhole approximation
                # ZED 720p horizontal FOV ≈ 90°, 1280px wide
                scale = obj.z * 2 * 0.7854 / 1280   # tan(45°) * z / half_width_px
                det.bbox.size.x = w_px * scale
                det.bbox.size.y = h_px * scale
                det.bbox.size.z = 0.3   # depth dimension unknown, default 0.3m
 
            # Class hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(obj.label)
            hyp.hypothesis.score    = obj.confidence
            det.results.append(hyp)
 
            msg.detections.append(det)
 
        self._node.pub_detections.publish(msg)
 
    # ── std_msgs/String  (JSON) ───────────────────────────────
 
    def _publish_json(self, header, frame: DetectionFrame):
        payload = {
            'frame_id':  frame.frame_id,
            'timestamp': frame.timestamp,
            'objects': [
                {
                    'id':         obj.obj_id,
                    'label':      obj.label,
                    'class_name': obj.class_name,
                    'confidence': round(obj.confidence, 3),
                    'position': {
                        'x': round(obj.x, 3),
                        'y': round(obj.y, 3),
                        'z': round(obj.z, 3),
                    },
                    'bbox_2d': obj.bbox_2d,
                }
                for obj in frame.objects
            ]
        }
        msg      = String()
        msg.data = json.dumps(payload)
        self._node.pub_json.publish(msg)
 
    # ── visualization_msgs/MarkerArray  (RViz2 boxes) ─────────
 
    def _publish_markers(self, header, frame: DetectionFrame):
        marker_array = MarkerArray()
 
        # Delete all previous markers first
        delete_all        = Marker()
        delete_all.header = header
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
 
        COLORS = [
            (0.0, 1.0, 0.4),   # green
            (0.0, 0.7, 1.0),   # cyan
            (1.0, 0.4, 0.0),   # orange
            (1.0, 0.0, 0.7),   # pink
            (0.7, 1.0, 0.0),   # lime
        ]
 
        for i, obj in enumerate(frame.objects):
            color = COLORS[obj.label % len(COLORS)]
 
            # ── Cube marker (3D box) ──
            cube          = Marker()
            cube.header   = header
            cube.ns       = 'detections'
            cube.id       = i * 2
            cube.type     = Marker.CUBE
            cube.action   = Marker.ADD
 
            cube.pose.position.x = obj.x
            cube.pose.position.y = obj.y
            cube.pose.position.z = obj.z
            cube.pose.orientation.w = 1.0
 
            # Size estimation (same as above)
            if obj.bbox_2d and len(obj.bbox_2d) == 4:
                w_px  = abs(obj.bbox_2d[1][0] - obj.bbox_2d[0][0])
                h_px  = abs(obj.bbox_2d[3][1] - obj.bbox_2d[0][1])
                scale = obj.z * 2 * 0.7854 / 1280
                cube.scale.x = max(w_px * scale, 0.1)
                cube.scale.y = max(h_px * scale, 0.1)
                cube.scale.z = 0.3
            else:
                cube.scale.x = 0.5
                cube.scale.y = 0.5
                cube.scale.z = 0.5
 
            cube.color.r = color[0]
            cube.color.g = color[1]
            cube.color.b = color[2]
            cube.color.a = 0.35          # semi-transparent
            cube.lifetime.sec = 0        # 0 = keep until DELETEALL
            marker_array.markers.append(cube)
 
            # ── Text marker (class label above box) ──
            text          = Marker()
            text.header   = header
            text.ns       = 'labels'
            text.id       = i * 2 + 1
            text.type     = Marker.TEXT_VIEW_FACING
            text.action   = Marker.ADD
 
            text.pose.position.x = obj.x
            text.pose.position.y = obj.y
            text.pose.position.z = obj.z + cube.scale.z / 2 + 0.15
            text.pose.orientation.w = 1.0
 
            text.scale.z  = 0.2          # text height in metres
            text.color.r  = color[0]
            text.color.g  = color[1]
            text.color.b  = color[2]
            text.color.a  = 1.0
            text.text     = f"{obj.class_name}\nZ: {obj.z:.1f}m"
            text.lifetime.sec = 0
            marker_array.markers.append(text)
 
        self._node.pub_markers.publish(marker_array)
 
    # ── Helpers ───────────────────────────────────────────────
 
    @staticmethod
    def _to_ros_time(unix_timestamp: float) -> Time:
        t     = Time()
        t.sec = int(unix_timestamp)
        t.nanosec = int((unix_timestamp - t.sec) * 1e9)
        return t