# camera_node.py
import threading
import pyzed.sl as sl
import cv2
import numpy as np

class CameraNode(threading.Thread):
    def __init__(self, frame_queue, stop_event):
        super().__init__(daemon=True, name="CameraNode")
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.zed = sl.Camera()

    def setup(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.camera_fps = 30  # Explicit FPS cap

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to open ZED camera")

        pt_params = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(pt_params)

        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        self.zed.enable_object_detection(obj_param)

    def run(self):
        image_zed = sl.Mat()
        while not self.stop_event.is_set():
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                frame = image_zed.get_data().copy()  # Copy BEFORE releasing ZED buffer
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

                # Drop frame if queue is full — never block the camera
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait((frame, img_rgb))

    def shutdown(self):
        self.stop_event.set()
        self.zed.disable_object_detection()
        self.zed.disable_positional_tracking()
        self.zed.close()

    def get_zed(self):
        return self.zed  # Node 3 needs ZED to ingest + retrieve objects