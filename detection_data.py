"""
detection_data.py
─────────────────
Shared data structures passed between the pipeline threads
and handed to the ROS 2 publisher.
"""

from dataclasses import dataclass, field
from typing import List
import time

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

@dataclass
class DetectedObject:
    obj_id:     int
    label:      int
    class_name: str
    confidence: float
    x: float            # metres, ZED camera frame
    y: float            # metres
    z: float            # metres — true depth from ZED
    bbox_2d: list       # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] pixels

@dataclass
class DetectionFrame:
    timestamp: float            = field(default_factory=time.time)
    frame_id:  int              = 0
    objects:   List[DetectedObject] = field(default_factory=list)