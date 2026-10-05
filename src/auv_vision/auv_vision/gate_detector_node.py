#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import cv2
from ament_index_python.packages import get_package_share_directory

class GateDetectorNode(Node):
    # 3 = gate in v8 model, 0 = gate in v2 model
    GATE_CLASS_IDS = {3}

    def __init__(self):
        super().__init__('gate_detector_node')

        # ------------------------------------------------------------------ #
        #  Declare all tunable parameters (overridable via YAML or CLI)       #
        # ------------------------------------------------------------------ #
        self.declare_parameter('confidence_threshold', 0.10)
        # debug_image_dir: '' = auto-select ~/.ros/yolo_debug; set abs path to override
        self.declare_parameter('debug_image_dir', '')
        # model_weights: filename of the .pt file inside the package weights/ dir
        self.declare_parameter('model_weights', 'gate_detection_v8.pt')
        # process_rate_hz: max YOLO inference rate — camera may tick faster than this.
        # 5 Hz is plenty for the gate FSM (navigator runs at 10 Hz with 1.5 s freshness).
        self.declare_parameter('process_rate_hz', 5.0)
        # save_debug_images: write annotated frame to disk each inference cycle.
        # Disable during normal runs to avoid continuous disk I/O.
        self.declare_parameter('save_debug_images', False)

        self.p_confidence_threshold = self.get_parameter('confidence_threshold').value
        raw_debug_dir               = self.get_parameter('debug_image_dir').value
        self.debug_dir = (
            raw_debug_dir if raw_debug_dir
            else os.path.join(os.path.expanduser('~'), '.ros', 'yolo_debug')
        )
        p_model_weights             = self.get_parameter('model_weights').value
        self.p_min_period_sec       = 1.0 / max(self.get_parameter('process_rate_hz').value, 0.1)
        self.p_save_debug           = self.get_parameter('save_debug_images').value
        self._last_infer_time: float = 0.0
        # ------------------------------------------------------------------ #

        # 1. Initialize cv_bridge
        self.cv_bridge = CvBridge()

        # 2. Locate and load the YOLOv8 model ONCE at startup
        package_share_directory = get_package_share_directory('auv_vision')
        weights_path = os.path.join(package_share_directory, 'weights', p_model_weights)

        self.get_logger().info(f'Loading YOLOv8 weights from: {weights_path}')
        self.model = YOLO(weights_path)
        self.get_logger().info(f'YOLOv8 model loaded successfully! ({p_model_weights})')

        # Create debug output directory
        os.makedirs(self.debug_dir, exist_ok=True)
        self.get_logger().info(f'Debug images will be saved to {self.debug_dir}')
        
        # 3. Create Publisher for custom 2D detection array
        self.detection_pub = self.create_publisher(
            Float32MultiArray, 
            '/auv/gate_detection_2d', 
            10
        )
        
        # 4. Create Subscription to Gazebo left camera
        self.image_sub = self.create_subscription(
            Image,
            '/model/auv_box/stereo_front/left/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        # ── Rate limiter: skip frames so YOLO runs at most process_rate_hz ──
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self._last_infer_time) < self.p_min_period_sec:
            return
        self._last_infer_time = now

        # Convert ROS Image to OpenCV BGR image
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Run YOLO inference
        results = self.model(cv_image, verbose=False, conf=self.p_confidence_threshold)
        
        # Generate and save debug image (only when explicitly enabled)
        if self.p_save_debug:
            try:
                annotated_frame = results[0].plot()
                debug_img_path = os.path.join(self.debug_dir, 'latest_detection.jpg')
                cv2.imwrite(debug_img_path, annotated_frame)
            except Exception as e:
                self.get_logger().warn(f'Failed to save debug image: {e}')
        
        # Collect all valid detections
        valid_detections = []
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Check against the set of valid gate class IDs
            if class_id in self.GATE_CLASS_IDS:
                valid_detections.append((confidence, box))
                
        # Sort by confidence descending so the best matches are first
        valid_detections.sort(key=lambda x: x[0], reverse=True)
        
        # Prepare the Float32MultiArray message
        # Format: [detected(0/1), cx, cy, w, h, conf, 1.0]
        msg_out = Float32MultiArray()
        
        if len(valid_detections) == 0:
            # No gate found in this frame
            msg_out.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
        else:
            # Take the highest-confidence gate detection (whole gate as one box)
            conf, box = valid_detections[0]
            cx, cy, w, h = box.xywh[0].tolist()
            
            # num_poles field is always 1.0 — v6 model sees gate as a single object
            msg_out.data = [1.0, float(cx), float(cy), float(w), float(h), float(conf), 1.0]
            self.get_logger().info(f'Gate detected! Center: ({cx:.1f}, {cy:.1f}), Size: {w:.0f}x{h:.0f}, Conf: {conf:.2f}')
            
        # Publish the array
        self.detection_pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = GateDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()