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
    # 0 = gate (whole gate detected as a single box in v2 model)
    GATE_CLASS_IDS = {0}

    def __init__(self):
        super().__init__('gate_detector_node')
        
        # 1. Initialize cv_bridge
        self.cv_bridge = CvBridge()
        
        # 2. Locate and load the YOLOv8 model ONCE at startup
        package_share_directory = get_package_share_directory('auv_vision')
        weights_path = os.path.join(package_share_directory, 'weights', 'gate_detector_v2.pt')
        
        self.get_logger().info(f'Loading YOLOv8 weights from: {weights_path}')
        self.model = YOLO(weights_path)
        self.get_logger().info('YOLOv8 model loaded successfully!')

        # Create debug output directory
        self.debug_dir = '/home/pai/yolo_debug'
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
        # Convert ROS Image to OpenCV BGR image
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Run YOLO inference
        results = self.model(cv_image, verbose=False, conf=0.10)
        
        # Generate and save debug image
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
            
            # num_poles field is always 1.0 — v2 model sees gate as a single object
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