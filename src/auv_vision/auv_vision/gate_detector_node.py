#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory

class GateDetectorNode(Node):
    def __init__(self):
        super().__init__('gate_detector_node')
        
        # 1. Initialize cv_bridge
        self.cv_bridge = CvBridge()
        
        # 2. Locate and load the YOLOv8 model ONCE at startup
        # This safely finds the weights file installed by setup.py
        package_share_directory = get_package_share_directory('auv_vision')
        weights_path = os.path.join(package_share_directory, 'weights', 'gate_detector.pt')
        
        self.get_logger().info(f'Loading YOLOv8 weights from: {weights_path}')
        self.model = YOLO(weights_path)
        self.get_logger().info('YOLOv8 model loaded successfully!')
        
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

        # Run YOLO inference (verbose=False prevents console spam for every single frame)
        results = self.model(cv_image, verbose=False, conf=0.15)
        
        best_conf = -1.0
        best_box = None
        
        # Process the detections
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Filter ONLY for class ID 9 (qual_gate)
            if class_id == 9:
                # Keep track of the highest confidence detection in this frame
                if confidence > best_conf:
                    best_conf = confidence
                    best_box = box
                    
        # Prepare the Float32MultiArray message
        msg_out = Float32MultiArray()
        
        if best_box is not None:
            # Ultralytics xywh provides [center_x, center_y, width, height]
            cx, cy, w, h = best_box.xywh[0].tolist()
            
            # Array Layout: [detected, center_x, center_y, width, height, confidence]
            msg_out.data = [1.0, float(cx), float(cy), float(w), float(h), float(best_conf)]
            
            # Log the successful detection cleanly
            self.get_logger().info(f'Gate detected! Center: ({cx:.1f}, {cy:.1f}), Conf: {best_conf:.2f}')
        else:
            # No gate found in this frame
            msg_out.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
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