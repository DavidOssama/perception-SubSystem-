# perception-SubSystem

--> open the screenshot photo to understand the system flow

**Perception Pipeline**
/// camera pipeline ////
- `detection_data.py` — shared data format used by all nodes
- `camera_node.py` — opens ZED camera, grabs frames
- `camera_inference_node.py` — runs YOLO on every frame
- `camera_fusion_node.py` — gets 3D position from ZED depth
- `ros2_publisher_node.py` — publishes results to ROS 2 topics and other subsystems
- `camera_main.py` — starts and connects all nodes together

/// lidar subscriber ////
-`lidar_subscriber.py` - detected objects position according to (x , y and z) of point cloud
