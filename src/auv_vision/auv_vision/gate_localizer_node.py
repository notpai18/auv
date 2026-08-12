#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from std_msgs.msg import Float32MultiArray
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import numpy as np
import math

class GateLocalizerNode(Node):
    def __init__(self):
        super().__init__('gate_localizer_node')
        self.cv_bridge = CvBridge()
        
        # Store the latest YOLO detection here since it doesn't have a timestamp
        self.latest_detection = None
        
        # Minimum confidence to accept a detection and compute 3D position
        self.CONFIDENCE_THRESHOLD = 0.10
        
        # 1. Standard subscriber for the YOLO 2D detection array
        self.det_sub = self.create_subscription(
            Float32MultiArray,
            '/auv/gate_detection_2d',
            self.detection_callback,
            10
        )
        
        # 2. Synchronized subscribers for disparity and camera info
        self.disp_sub = message_filters.Subscriber(self, DisparityImage, '/disparity')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/model/auv_box/stereo_front/left/camera_info')
        
        # Synchronize the two topics based on their header timestamps
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.disp_sub, self.info_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.pos_pub = self.create_publisher(Float32MultiArray, '/auv/gate_position_3d', 10)
        
        self.get_logger().info(f"Gate 3D Localizer Node started! Confidence threshold: {self.CONFIDENCE_THRESHOLD}")

    def detection_callback(self, msg):
        self.latest_detection = msg.data

    def sync_callback(self, disp_msg, info_msg):
        # Ensure we have received a detection at least once
        if not self.latest_detection:
            return
            
        detected = self.latest_detection[0]
        
        # If no gate is detected this cycle, do nothing
        if detected == 0.0:
            return
        
        # --- Confidence gate: reject low-quality detections ---
        confidence = self.latest_detection[5]
        if confidence < self.CONFIDENCE_THRESHOLD:
            self.get_logger().debug(f"Detection rejected: confidence {confidence:.2f} < {self.CONFIDENCE_THRESHOLD}")
            return
            
        center_x_px = self.latest_detection[1]
        center_y_px = self.latest_detection[2]
        width_px    = self.latest_detection[3]
        height_px   = self.latest_detection[4]
        
        # a. Convert disparity image to OpenCV numpy array (32FC1)
        try:
            cv_disp = self.cv_bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
            
        # b & c. Disparity sampling: target the physical gate structure,
        # NOT the open water gap in the centre.
        # - left/right poles are at roughly ±50% of the box width
        # - crossbar is at the top edge of the box
        # Sample with a 9x9 patch for more valid pixels on thin poles.
        h, w = cv_disp.shape
        left_pole_x  = center_x_px - width_px * 0.48
        right_pole_x = center_x_px + width_px * 0.48
        top_bar_y    = center_y_px - height_px * 0.45   # near top edge (crossbar)
        candidates = [
            ("left_pole",  left_pole_x,  center_y_px),
            ("right_pole", right_pole_x, center_y_px),
            ("crossbar",   center_x_px,  top_bar_y),
            ("center",     center_x_px,  center_y_px),   # last resort
        ]
        
        disparity_value = None
        used_candidate = "none"
        
        for name, cx, cy in candidates:
            px_x = int(round(cx))
            px_y = int(round(cy))
            
            # 9x9 patch bounds (clipped to image)
            x_min = max(0, px_x - 4)
            x_max = min(w, px_x + 5)
            y_min = max(0, px_y - 4)
            y_max = min(h, px_y + 5)
            
            patch = cv_disp[y_min:y_max, x_min:x_max]
            
            # Filter for valid values (>0 and not NaN)
            valid_mask = (patch > 0.0) & (~np.isnan(patch))
            valid_pixels = patch[valid_mask]
            
            # Need at least 3 valid pixels for a reliable median
            if len(valid_pixels) >= 3:
                disparity_value = np.median(valid_pixels)
                used_candidate = name
                break
                
        if disparity_value is None:
            self.get_logger().warn(
                f"No valid disparity across all patches for gate at ({int(center_x_px)}, {int(center_y_px)})"
            )
            return
            
        # d. Compute forward depth (Z in camera frame = X in ROS body frame)
        fx = info_msg.k[0]
        baseline = 0.12  # stereo baseline: 12 cm
        depth_z = (fx * baseline) / disparity_value
        
        # e & f. Compute lateral and vertical offsets using pinhole projection
        cam_cx = info_msg.k[2]
        cam_cy = info_msg.k[5]
        x_offset = (center_x_px - cam_cx) * depth_z / fx
        y_offset = (center_y_px - cam_cy) * depth_z / fx
        
        # Camera frame: Z-forward, X-right, Y-down
        # ROS body frame: X-forward, Y-left, Z-up
        x_out = float(depth_z)
        y_out = float(-x_offset)
        z_out = float(-y_offset)
        
        # Publish [x_forward, y_left, z_up, confidence]
        pt_msg = Float32MultiArray()
        pt_msg.data = [x_out, y_out, z_out, float(confidence)]
        self.pos_pub.publish(pt_msg)
        
        self.get_logger().info(
            f"Gate 3D -> X: {x_out:.2f}m  Y: {y_out:.2f}m  Z: {z_out:.2f}m "
            f"| conf: {confidence:.2f}  disp_src: {used_candidate}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = GateLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()