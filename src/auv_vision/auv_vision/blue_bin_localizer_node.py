#!/usr/bin/env python3
"""
blue_bin_localizer_node.py
==========================
Converts 2D blue bin detections into a 3D body-frame position vector
using two different depth sources depending on phase.
"""

import rclpy
from rclpy.node import Node
import message_filters
from std_msgs.msg import Float32MultiArray
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import math


class BlueBinLocalizerNode(Node):

    def __init__(self):
        super().__init__('blue_bin_localizer_node')
        self.cv_bridge = CvBridge()

        # ------------------------------------------------------------------ #
        #  ROS 2 Parameters                                                    #
        # ------------------------------------------------------------------ #
        self.declare_parameter('stereo_baseline',         0.12)   # m
        # pool_floor_z is the Z height (world/odom frame) of the TARGET SURFACE
        # the bottom camera should treat as "ground". In camera_test.sdf the
        # blue drum centres are at z=-4.75, radius=0.3 → top face at z=-4.45.
        # Using -4.5 (top of drums) instead of -5.0 (actual pool floor) makes
        # the altitude in _localize_bottom() reflect the camera-to-target
        # distance, not the camera-to-floor distance.
        self.declare_parameter('pool_floor_z',           -4.5)    # m (world frame — bin top)
        self.declare_parameter('bottom_cam_z_offset',    -0.28)   # m from base_link
        self.declare_parameter('min_depth_m',             0.3)    # plausibility guard
        self.declare_parameter('max_depth_m',            25.0)
        self.declare_parameter('patch_radius',            4)      # disparity sampling patch
        self.declare_parameter('min_valid_pixels',        3)
        self.declare_parameter('confidence_threshold',    0.5)  # min confidence to accept

        p = self.get_parameter
        self.p_baseline            = p('stereo_baseline').value
        self.p_pool_floor_z        = p('pool_floor_z').value
        self.p_bottom_cam_z_offset = p('bottom_cam_z_offset').value
        self.p_min_depth           = p('min_depth_m').value
        self.p_max_depth           = p('max_depth_m').value
        self.p_patch_radius        = p('patch_radius').value
        self.p_min_valid_pixels    = p('min_valid_pixels').value
        self.p_conf_thresh         = p('confidence_threshold').value
        # ------------------------------------------------------------------ #

        self.latest_front_det  = None
        self.latest_bottom_det = None
        self.latest_odom_z     = None

        self.front_fx = None
        self.front_cx = None
        self.front_cy = None
        self.bottom_fx = None
        self.bottom_fy = None
        self.bottom_cx = None
        self.bottom_cy = None

        self.create_subscription(Float32MultiArray, '/auv/blue_bin_detection_front_2d', self._front_det_callback, 10)
        self.create_subscription(Float32MultiArray, '/auv/blue_bin_detection_bottom_2d', self._bottom_det_callback, 10)
        self.create_subscription(Odometry, '/auv/odom', self._odom_callback, 10)
        self.create_subscription(CameraInfo, '/model/auv_box/stereo_front/left/camera_info', self._front_info_callback, 10)
        self.create_subscription(CameraInfo, '/model/auv_box/bottom/camera_info', self._bottom_info_callback, 10)

        self.disp_sub  = message_filters.Subscriber(self, DisparityImage, '/disparity')
        self.finfo_sub = message_filters.Subscriber(self, CameraInfo, '/model/auv_box/stereo_front/left/camera_info')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.disp_sub, self.finfo_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self._disp_sync_callback)

        self.latest_disp_msg  = None
        self.latest_finfo_msg = None

        self.pos_pub = self.create_publisher(Float32MultiArray, '/auv/blue_bin_position_3d', 10)

        self.create_timer(0.1, self._process)

        self.get_logger().info('Blue Bin 3D Localizer started!')

    def _front_det_callback(self, msg):
        self.latest_front_det = list(msg.data)

    def _bottom_det_callback(self, msg):
        self.latest_bottom_det = list(msg.data)

    def _odom_callback(self, msg):
        self.latest_odom_z = msg.pose.pose.position.z

    def _front_info_callback(self, msg):
        if self.front_fx is None:
            self.front_fx = msg.k[0]
            self.front_cx = msg.k[2]
            self.front_cy = msg.k[5]

    def _bottom_info_callback(self, msg):
        if self.bottom_fx is None:
            self.bottom_fx = msg.k[0]
            self.bottom_fy = msg.k[4]
            self.bottom_cx = msg.k[2]
            self.bottom_cy = msg.k[5]

    def _disp_sync_callback(self, disp_msg, info_msg):
        self.latest_disp_msg  = disp_msg
        self.latest_finfo_msg = info_msg

    def _sample_disparity(self, cv_disp, px_x, px_y):
        h, w = cv_disp.shape[:2]
        px_x = int(round(px_x))
        px_y = int(round(px_y))
        r    = self.p_patch_radius

        x_min = max(0, px_x - r)
        x_max = min(w, px_x + r + 1)
        y_min = max(0, px_y - r)
        y_max = min(h, px_y + r + 1)

        patch = np.asarray(cv_disp[y_min:y_max, x_min:x_max], dtype=np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0.0)]

        if len(valid) < self.p_min_valid_pixels:
            return None
        return float(np.median(valid))

    def _localize_front(self):
        det = self.latest_front_det
        if det is None or det[0] < 0.5:
            return None
        if self.front_fx is None or self.latest_disp_msg is None:
            return None

        cx_px = det[1]
        cy_px = det[2]
        conf  = det[5]

        if conf < self.p_conf_thresh:
            return None

        try:
            cv_disp = self.cv_bridge.imgmsg_to_cv2(self.latest_disp_msg.image, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Phase1 CV Bridge error: {e}')
            return None

        disparity_value = self._sample_disparity(cv_disp, cx_px, cy_px)
        if disparity_value is None:
            return None

        fx       = self.front_fx
        baseline = self.p_baseline
        slant    = (fx * baseline) / disparity_value

        if not math.isfinite(slant) or not (self.p_min_depth <= slant <= self.p_max_depth):
            return None

        theta = math.atan2(cy_px - self.front_cy, fx)
        horizontal_dist = slant * math.cos(abs(theta))
        lateral_offset  = (cx_px - self.front_cx) * slant / fx

        x_fwd  = float(horizontal_dist)
        y_left = float(-lateral_offset)
        return x_fwd, y_left, float(conf)

    def _localize_bottom(self):
        det = self.latest_bottom_det
        if det is None or det[0] < 0.5:
            return None
        if self.bottom_fx is None or self.latest_odom_z is None:
            return None

        cx_px    = det[1]
        cy_px    = det[2]
        conf     = det[5]

        if conf < self.p_conf_thresh:
            return None

        auv_z      = self.latest_odom_z
        cam_z      = auv_z + self.p_bottom_cam_z_offset
        altitude   = abs(cam_z - self.p_pool_floor_z)

        if altitude < 0.01:
            return None

        x_fwd  = float( (cy_px - self.bottom_cy) * altitude / self.bottom_fy)
        y_left = float(-(cx_px - self.bottom_cx) * altitude / self.bottom_fx)

        return x_fwd, y_left, float(conf)

    def _process(self):
        x_fwd  = None
        y_left = None
        conf   = 0.0
        source = 0.0

        bottom_det = self.latest_bottom_det
        if bottom_det is not None and bottom_det[0] > 0.5 and bottom_det[5] >= self.p_conf_thresh:
            result = self._localize_bottom()
            if result is not None:
                x_fwd, y_left, conf = result
                source = 2.0

        if x_fwd is None:
            result = self._localize_front()
            if result is not None:
                x_fwd, y_left, conf = result
                source = 1.0

        if x_fwd is None:
            return

        msg = Float32MultiArray()
        msg.data = [float(x_fwd), float(y_left), 0.0, float(conf), float(source)]
        self.pos_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BlueBinLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
