#!/usr/bin/env python3
"""
blue_bin_detector_node.py
=========================
YOLO-based blue bin detector.

Every detection with a valid blue bin bounding box is saved to a
timestamped JPEG inside a per-run subfolder so results accumulate
across simulator runs and are never overwritten.

Save directory layout:
  ~/.ros/blue_bin_detections/
    run_YYYYMMDD_HHMMSS/
      front_0001_conf0.87.jpg
      front_0002_conf0.91.jpg
      bottom_0001_conf0.74.jpg
      ...
    latest_front.jpg      <- always the most recent annotated front frame
    latest_bottom.jpg     <- always the most recent annotated bottom frame
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import cv2
from datetime import datetime
from ament_index_python.packages import get_package_share_directory
import time


class BlueBinDetectorNode(Node):
    # 1 = blue_bin in v7 model
    BLUE_BIN_CLASS_ID = 1

    def __init__(self):
        super().__init__('blue_bin_detector_node')

        self.declare_parameter('confidence_threshold', 0.10)
        self.declare_parameter('debug_image_dir', '')
        self.declare_parameter('model_weights', 'gate_detection_v7.pt')
        self.declare_parameter('process_rate_hz', 5.0)    # max YOLO inference rate
        # Disable debug image writes by default — heavy disk I/O at inference rate
        self.declare_parameter('save_debug_images', False)

        self.p_confidence_threshold = self.get_parameter('confidence_threshold').value
        raw_debug_dir = self.get_parameter('debug_image_dir').value
        base_dir = (
            raw_debug_dir if raw_debug_dir
            else os.path.join(os.path.expanduser('~'), '.ros', 'blue_bin_detections')
        )
        p_model_weights = self.get_parameter('model_weights').value
        self.process_interval = 1.0 / self.get_parameter('process_rate_hz').value
        self.p_save_debug = self.get_parameter('save_debug_images').value

        # ── Per-run save directory ──────────────────────────────────────────
        run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir    = os.path.join(base_dir, f'run_{run_stamp}')
        self.latest_dir = base_dir   # latest_front.jpg / latest_bottom.jpg go here
        os.makedirs(self.run_dir,    exist_ok=True)
        os.makedirs(self.latest_dir, exist_ok=True)

        self.detection_count = {'front': 0, 'bottom': 0}

        self.cv_bridge = CvBridge()

        package_share_directory = get_package_share_directory('auv_vision')
        weights_path = os.path.join(package_share_directory, 'weights', p_model_weights)

        self.get_logger().info(f'Loading YOLO model from: {weights_path}')
        self.model = YOLO(weights_path)
        self.get_logger().info(f'YOLO model loaded! ({p_model_weights})')
        self.get_logger().info(f'Detection images -> {self.run_dir}')

        self.front_pub  = self.create_publisher(Float32MultiArray, '/auv/blue_bin_detection_front_2d',  10)
        self.bottom_pub = self.create_publisher(Float32MultiArray, '/auv/blue_bin_detection_bottom_2d', 10)

        self.front_sub  = self.create_subscription(
            Image, '/model/auv_box/stereo_front/left/image_raw', self.front_callback,  10)
        self.bottom_sub = self.create_subscription(
            Image, '/model/auv_box/bottom/image_raw',            self.bottom_callback, 10)

        self.last_front_time  = 0.0
        self.last_bottom_time = 0.0

    # ------------------------------------------------------------------
    def process_image(self, msg, publisher, source_name):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert {source_name} image: {e}')
            return

        results = self.model(cv_image, verbose=False, conf=self.p_confidence_threshold)

        # Generate annotated frame (only render when debug saving is enabled)
        annotated = cv_image
        if self.p_save_debug:
            try:
                annotated = results[0].plot()
                cv2.imwrite(os.path.join(self.latest_dir, f'latest_{source_name}.jpg'), annotated)
            except Exception as e:
                self.get_logger().warn(f'Failed to save latest {source_name} image: {e}')

        # Gather valid blue bin detections
        valid_detections = []
        for box in results[0].boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id == self.BLUE_BIN_CLASS_ID:
                valid_detections.append((confidence, box))

        valid_detections.sort(key=lambda x: x[0], reverse=True)

        msg_out = Float32MultiArray()
        if len(valid_detections) == 0:
            msg_out.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            conf, box = valid_detections[0]
            cx, cy, w, h = box.xywh[0].tolist()
            msg_out.data = [1.0, float(cx), float(cy), float(w), float(h), float(conf), 1.0]

            # ── Save individual detection frames only when debug mode on ──
            if self.p_save_debug:
                try:
                    self.detection_count[source_name] += 1
                    n     = self.detection_count[source_name]
                    fname = f'{source_name}_{n:04d}_conf{conf:.2f}.jpg'
                    fpath = os.path.join(self.run_dir, fname)
                    cv2.imwrite(fpath, annotated)
                    self.get_logger().info(
                        f'[BLUE BIN] {source_name} detection #{n} saved -> {fname}  '
                        f'(cx={cx:.0f}, cy={cy:.0f}, conf={conf:.2f})'
                    )
                except Exception as e:
                    self.get_logger().warn(f'Failed to save detection image: {e}')

        publisher.publish(msg_out)

    # ------------------------------------------------------------------
    def front_callback(self, msg):
        now = time.time()
        if now - self.last_front_time >= self.process_interval:
            self.last_front_time = now
            self.process_image(msg, self.front_pub, 'front')

    def bottom_callback(self, msg):
        now = time.time()
        if now - self.last_bottom_time >= self.process_interval:
            self.last_bottom_time = now
            self.process_image(msg, self.bottom_pub, 'bottom')


def main(args=None):
    rclpy.init(args=args)
    node = BlueBinDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
