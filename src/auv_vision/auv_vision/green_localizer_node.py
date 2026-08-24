#!/usr/bin/env python3
"""
green_localizer_node.py
=======================
Converts 2D green mat detections into a 3D body-frame position vector
using two different depth sources depending on phase:

  Phase 1 — Front stereo camera (long-range, mat in lower FOV)
    Uses stereo disparity to get slant range, then projects the pixel
    angular position to recover the horizontal distance and lateral offset
    to the mat centre.

    Geometry (camera horizontal, mat on floor below):
      slant_range       = fx * baseline / disparity
      look_down_angle θ = arctan((cy_px - cam_cy) / fx)   [rad, +ve = below]
      horizontal_dist   = slant_range * cos(θ)
      lateral_offset    = (cx_px - cam_cx) * slant_range / fx

  Phase 2 — Bottom camera (AUV directly above mat)
    Uses the AUV's Z coordinate from /auv/odom as altitude above the pool
    floor, then applies standard pinhole projection to get the lateral error.

      altitude = |auv_z - pool_floor_z|   (pool floor at Z = -5.0 m in SDF)
      x_fwd    =  (cy_px - cam_cy) * altitude / fy
      y_left   = -(cx_px - cam_cx) * altitude / fx

  Source selection:
    If bottom camera coverage_frac >= phase2_coverage_thresh → Phase 2
    Else if front detection is valid → Phase 1

Publications
------------
  /auv/green_position_3d  — Float32MultiArray
      [x_fwd_m, y_left_m, z_up_m, confidence, source]
      source: 1.0 = front stereo, 2.0 = bottom camera

Sim-to-real: all geometric constants are ROS 2 parameters.
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


class GreenLocalizerNode(Node):

    def __init__(self):
        super().__init__('green_localizer_node')
        self.cv_bridge = CvBridge()

        # ------------------------------------------------------------------ #
        #  ROS 2 Parameters                                                    #
        # ------------------------------------------------------------------ #
        self.declare_parameter('stereo_baseline',         0.12)   # m
        self.declare_parameter('pool_floor_z',           -5.0)    # m (world frame)
        self.declare_parameter('bottom_cam_z_offset',    -0.28)   # m from base_link
        self.declare_parameter('phase2_coverage_thresh',  0.03)   # fraction of bottom frame
        self.declare_parameter('min_depth_m',             0.3)    # plausibility guard
        self.declare_parameter('max_depth_m',            25.0)
        self.declare_parameter('patch_radius',            4)      # disparity sampling patch
        self.declare_parameter('min_valid_pixels',        3)
        self.declare_parameter('confidence_threshold',    0.001)  # min coverage to accept

        p = self.get_parameter
        self.p_baseline            = p('stereo_baseline').value
        self.p_pool_floor_z        = p('pool_floor_z').value
        self.p_bottom_cam_z_offset = p('bottom_cam_z_offset').value
        self.p_phase2_thresh       = p('phase2_coverage_thresh').value
        self.p_min_depth           = p('min_depth_m').value
        self.p_max_depth           = p('max_depth_m').value
        self.p_patch_radius        = p('patch_radius').value
        self.p_min_valid_pixels    = p('min_valid_pixels').value
        self.p_conf_thresh         = p('confidence_threshold').value
        # ------------------------------------------------------------------ #

        # Latest data store (callbacks are not time-synchronised to each other)
        self.latest_front_det  = None   # [detected, cx, cy, w, h, conf, 1.0]
        self.latest_bottom_det = None   # [detected, cx, cy, area, cov, conf, 2.0]
        self.latest_odom_z     = None   # AUV z position in world frame (m)

        # Camera intrinsics — populated from camera_info callbacks
        self.front_fx = None
        self.front_cx = None
        self.front_cy = None
        self.bottom_fx = None
        self.bottom_fy = None
        self.bottom_cx = None
        self.bottom_cy = None

        # ------------------------------------------------------------------ #
        #  Subscriptions                                                       #
        # ------------------------------------------------------------------ #
        # Simple (async) subscriptions — no tight time-sync needed here
        self.create_subscription(
            Float32MultiArray,
            '/auv/green_detection_front_2d',
            self._front_det_callback, 10)

        self.create_subscription(
            Float32MultiArray,
            '/auv/green_detection_bottom_2d',
            self._bottom_det_callback, 10)

        self.create_subscription(
            Odometry, '/auv/odom',
            self._odom_callback, 10)

        # Front camera info (for fx, cx, cy)
        self.create_subscription(
            CameraInfo,
            '/model/auv_box/stereo_front/left/camera_info',
            self._front_info_callback, 10)

        # Bottom camera info (for fx, fy, cx, cy)
        self.create_subscription(
            CameraInfo,
            '/model/auv_box/bottom/camera_info',
            self._bottom_info_callback, 10)

        # Disparity (synchronized with front camera info for Phase 1)
        self.disp_sub  = message_filters.Subscriber(
            self, DisparityImage, '/disparity')
        self.finfo_sub = message_filters.Subscriber(
            self, CameraInfo,
            '/model/auv_box/stereo_front/left/camera_info')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.disp_sub, self.finfo_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self._disp_sync_callback)

        # Latest disparity + front info (populated by ts callback)
        self.latest_disp_msg  = None
        self.latest_finfo_msg = None

        # Publisher
        self.pos_pub = self.create_publisher(
            Float32MultiArray, '/auv/green_position_3d', 10)

        # Main processing timer — 10 Hz, same cadence as navigator
        self.create_timer(0.1, self._process)

        self.get_logger().info(
            f'Green 3D Localizer started! '
            f'baseline={self.p_baseline}m | '
            f'pool_floor_z={self.p_pool_floor_z}m | '
            f'phase2_thresh={self.p_phase2_thresh:.3f}'
        )

    # ------------------------------------------------------------------ #
    #  Simple store callbacks                                              #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Disparity sampling (identical to gate_localizer implementation)    #
    # ------------------------------------------------------------------ #

    def _sample_disparity(self, cv_disp, px_x, px_y):
        """Return median disparity in a patch around (px_x, px_y).
        Returns None when fewer than min_valid_pixels finite+positive values."""
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

    # ------------------------------------------------------------------ #
    #  Phase 1 — front stereo slant projection                            #
    # ------------------------------------------------------------------ #

    def _localize_front(self):
        """
        Returns (x_fwd, y_left, conf) in body frame using the front stereo
        camera and the latest disparity image, or None on failure.
        """
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

        # Convert disparity image to numpy
        try:
            cv_disp = self.cv_bridge.imgmsg_to_cv2(
                self.latest_disp_msg.image, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Phase1 CV Bridge error: {e}')
            return None

        disparity_value = self._sample_disparity(cv_disp, cx_px, cy_px)
        if disparity_value is None:
            self.get_logger().warn('Phase1: no valid disparity at green centroid')
            return None

        fx       = self.front_fx
        baseline = self.p_baseline
        slant    = (fx * baseline) / disparity_value

        if not math.isfinite(slant) or not (self.p_min_depth <= slant <= self.p_max_depth):
            self.get_logger().warn(f'Phase1: slant {slant:.2f}m out of range')
            return None

        # Look-down angle from camera horizontal axis
        # cy_px > cam_cy means the blob is BELOW the optical axis
        # (camera is pitched 0°, so below = down in world)
        theta = math.atan2(cy_px - self.front_cy, fx)   # positive = below horizon
        horizontal_dist = slant * math.cos(abs(theta))
        lateral_offset  = (cx_px - self.front_cx) * slant / fx

        # Body frame: X forward, Y left, Z up
        x_fwd  = float(horizontal_dist)
        y_left = float(-lateral_offset)

        self.get_logger().info(
            f'[Phase1-FRONT] slant={slant:.2f}m θ={math.degrees(theta):.1f}° '
            f'→ x_fwd={x_fwd:.2f}m y_left={y_left:.2f}m conf={conf:.4f}'
        )
        return x_fwd, y_left, float(conf)

    # ------------------------------------------------------------------ #
    #  Phase 2 — bottom camera altitude projection                         #
    # ------------------------------------------------------------------ #

    def _localize_bottom(self):
        """
        Returns (x_fwd, y_left, coverage) using the bottom camera and odom Z,
        or None on failure.
        """
        det = self.latest_bottom_det
        if det is None or det[0] < 0.5:
            return None
        if self.bottom_fx is None or self.latest_odom_z is None:
            return None

        cx_px    = det[1]
        cy_px    = det[2]
        coverage = det[4]   # green pixel fraction of bottom frame

        # Altitude above pool floor: AUV body is at odom_z,
        # bottom camera sits 0.28 m below base_link.
        auv_z      = self.latest_odom_z
        cam_z      = auv_z + self.p_bottom_cam_z_offset   # camera world Z
        altitude   = abs(cam_z - self.p_pool_floor_z)

        if altitude < 0.1:
            self.get_logger().warn('Phase2: altitude < 0.1m, skipping')
            return None

        # Standard pinhole projection: pixel error → metric error
        # Bottom camera: image X → AUV Y-left, image Y → AUV X-forward
        # (because camera is rotated 90° pitch, looking straight down)
        x_fwd  = float( (cy_px - self.bottom_cy) * altitude / self.bottom_fy)
        y_left = float(-(cx_px - self.bottom_cx) * altitude / self.bottom_fx)

        self.get_logger().info(
            f'[Phase2-BOTTOM] alt={altitude:.2f}m '
            f'cx={cx_px:.1f} cy={cy_px:.1f} '
            f'→ x_fwd={x_fwd:.2f}m y_left={y_left:.2f}m coverage={coverage:.3f}'
        )
        return x_fwd, y_left, float(coverage)

    # ------------------------------------------------------------------ #
    #  Main processing timer                                               #
    # ------------------------------------------------------------------ #

    def _process(self):
        """
        Select the best available source, compute 3D position, publish.
        Priority: Phase 2 (bottom cam) > Phase 1 (front stereo).
        """
        x_fwd  = None
        y_left = None
        conf   = 0.0
        source = 0.0

        # --- Check bottom camera first (Phase 2) ---
        bottom_det = self.latest_bottom_det
        if (bottom_det is not None and
                bottom_det[0] > 0.5 and
                bottom_det[4] >= self.p_phase2_thresh):
            result = self._localize_bottom()
            if result is not None:
                x_fwd, y_left, conf = result
                source = 2.0

        # --- Fall back to front stereo (Phase 1) ---
        if x_fwd is None:
            result = self._localize_front()
            if result is not None:
                x_fwd, y_left, conf = result
                source = 1.0

        if x_fwd is None:
            return   # No valid detection this tick — publish nothing

        msg = Float32MultiArray()
        msg.data = [float(x_fwd), float(y_left), 0.0, float(conf), float(source)]
        self.pos_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GreenLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
