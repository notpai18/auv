#!/usr/bin/env python3
"""
green_detector_node.py
======================
Detects the green target zone mat using HSV colour segmentation.
Runs on two camera feeds simultaneously:

  Phase 1 — Front stereo left camera (long-range approach)
    Only the bottom ROI_FRAC of the image is processed, because the
    mat (on the pool floor) appears in the lower portion of the
    horizontally-mounted camera's field of view.

  Phase 2 — Bottom camera (fine centering when AUV is above the mat)
    Full frame is processed.

Publications
------------
  /auv/green_detection_front_2d  — Float32MultiArray
      [detected, cx_px, cy_px, w_px, h_px, conf_approx, 1.0]
      conf_approx = largest_contour_area / image_area  (clamped 0-1)

  /auv/green_detection_bottom_2d — Float32MultiArray
      [detected, cx_px, cy_px, area_px2, coverage_frac, conf_approx, 2.0]

Both formats intentionally mirror the gate_detection_2d layout so the
logger and any downstream nodes can treat them uniformly.

Sim-to-real: all HSV thresholds and ROI fractions are ROS 2 parameters.
  • Sim:  clean uniform green  → tight HSV range works well.
  • Real: tune hsv_s_low down, widen hue range to cope with colour shift.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import os


class GreenDetectorNode(Node):

    def __init__(self):
        super().__init__('green_detector_node')
        self.cv_bridge = CvBridge()

        # ------------------------------------------------------------------ #
        #  ROS 2 Parameters — all overridable via YAML or CLI                 #
        # ------------------------------------------------------------------ #
        # HSV thresholds for the green mat  (OpenCV: H 0-179, S/V 0-255)
        # Gazebo mat colour: ambient 0.2 0.8 0.2  →  RGB(51,204,51)
        # → HSV approx (60°, 75%, 80%) → OpenCV HSV (60, 191, 204)
        self.declare_parameter('hsv_h_low',  38)    # hue lower bound
        self.declare_parameter('hsv_h_high', 82)    # hue upper bound
        self.declare_parameter('hsv_s_low',  80)    # saturation lower bound
        self.declare_parameter('hsv_v_low',  60)    # value lower bound

        # Front camera: only process this fraction of the image from the bottom
        # (mat only visible in lower part of horizontally-mounted camera)
        self.declare_parameter('front_roi_top_frac',   0.60)  # ignore top 60%
        self.declare_parameter('min_contour_area_px',  200.0) # px² noise gate

        # Debug image output directory ('' → auto ~/.ros/yolo_debug)
        self.declare_parameter('debug_image_dir', '')
        # Max processing rate per camera (camera ticks faster than needed)
        self.declare_parameter('process_rate_hz', 5.0)
        # Disable debug image writes by default — heavy disk I/O at process rate
        self.declare_parameter('save_debug_images', False)

        # Bind parameters
        p = self.get_parameter
        h_low  = p('hsv_h_low').value
        h_high = p('hsv_h_high').value
        s_low  = p('hsv_s_low').value
        v_low  = p('hsv_v_low').value

        self.lower_green = np.array([h_low,  s_low, v_low],  dtype=np.uint8)
        self.upper_green = np.array([h_high, 255,   255],    dtype=np.uint8)
        self.front_roi_frac    = p('front_roi_top_frac').value
        self.min_contour_area  = p('min_contour_area_px').value

        raw_debug_dir = p('debug_image_dir').value
        self.debug_dir = (
            raw_debug_dir if raw_debug_dir
            else os.path.join(os.path.expanduser('~'), '.ros', 'yolo_debug')
        )
        os.makedirs(self.debug_dir, exist_ok=True)
        _min_period = 1.0 / max(p('process_rate_hz').value, 0.1)
        self._front_min_period = _min_period
        self._bottom_min_period = _min_period
        self._front_last_t  = 0.0
        self._bottom_last_t = 0.0
        self.p_save_debug = p('save_debug_images').value
        # ------------------------------------------------------------------ #

        # Publishers
        self.front_pub = self.create_publisher(
            Float32MultiArray, '/auv/green_detection_front_2d', 10)
        self.bottom_pub = self.create_publisher(
            Float32MultiArray, '/auv/green_detection_bottom_2d', 10)

        # Subscriptions
        self.front_sub = self.create_subscription(
            Image,
            '/model/auv_box/stereo_front/left/image_raw',
            self.front_image_callback,
            10)
        self.bottom_sub = self.create_subscription(
            Image,
            '/model/auv_box/bottom/image_raw',
            self.bottom_image_callback,
            10)

        self.get_logger().info(
            f'Green Detector started! '
            f'HSV: H[{h_low},{h_high}] S[{s_low},255] V[{v_low},255] | '
            f'Front ROI top-frac: {self.front_roi_frac:.2f} | '
            f'Min contour: {self.min_contour_area:.0f} px²'
        )

    # ------------------------------------------------------------------ #
    #  Shared HSV detection logic                                          #
    # ------------------------------------------------------------------ #

    def _detect_green(self, bgr_image):
        """
        Run HSV thresholding on bgr_image.

        Returns (contour, cx_px, cy_px, w_px, h_px, area_px2) of the
        largest valid green contour, or None if nothing is found.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Morphological cleanup — remove tiny salt-and-pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, mask

        # Pick the largest contour
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)

        if area < self.min_contour_area:
            return None, mask

        x, y, w, h = cv2.boundingRect(best)
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)

        return (best, cx, cy, float(w), float(h), float(area)), mask

    def _build_msg(self, result, img_area, phase_id):
        """Build a Float32MultiArray from _detect_green result."""
        msg = Float32MultiArray()
        if result is None:
            msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(phase_id)]
        else:
            _, cx, cy, w, h, area = result
            conf_approx = min(area / max(img_area, 1.0), 1.0)
            coverage    = conf_approx  # same thing for the mat
            msg.data = [1.0, cx, cy, w, h, float(conf_approx), float(phase_id)]
        return msg

    def _save_debug(self, bgr_image, mask, result, filename, roi_y0=0):
        """Save an annotated debug image."""
        try:
            vis = bgr_image.copy()
            # Draw the green mask as a transparent overlay
            overlay = vis.copy()
            h_m, w_m = mask.shape[:2]
            overlay[roi_y0:roi_y0+h_m, :w_m][mask > 0] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

            if result is not None:
                contour, cx, cy, w, h, _ = result
                # Adjust contour/bbox back to full-image coords if using ROI
                x = int(cx - w / 2)
                y = int(cy - h / 2) + roi_y0
                cv2.rectangle(vis, (x, y), (x + int(w), y + int(h)),
                              (0, 200, 0), 2)
                cv2.circle(vis, (int(cx), int(cy + roi_y0)), 6,
                           (0, 255, 0), -1)
                cv2.putText(
                    vis,
                    f'GREEN ({cx:.0f},{cy + roi_y0:.0f})',
                    (x, max(y - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if roi_y0 > 0:
                cv2.line(vis, (0, roi_y0), (vis.shape[1], roi_y0),
                         (255, 255, 0), 1)

            path = os.path.join(self.debug_dir, filename)
            cv2.imwrite(path, vis)
        except Exception as e:
            self.get_logger().warn(f'Debug image save failed: {e}')

    # ------------------------------------------------------------------ #
    #  Front camera callback — Phase 1                                     #
    # ------------------------------------------------------------------ #

    def front_image_callback(self, msg):
        # Rate limiter
        import time as _time
        now = _time.monotonic()
        if (now - self._front_last_t) < self._front_min_period:
            return
        self._front_last_t = now

        try:
            full_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Front CV Bridge error: {e}')
            return

        h_full, w_full = full_image.shape[:2]
        img_area = float(h_full * w_full)

        # Crop to the bottom ROI only — mat visible in lower portion
        roi_y0 = int(h_full * self.front_roi_frac)
        roi = full_image[roi_y0:, :]

        result, mask = self._detect_green(roi)

        # Publish — pixel coords are relative to the ROI crop.
        # Downstream (green_localizer) receives the full camera_info,
        # so we must shift cy back to full-image pixel space.
        pub_msg = Float32MultiArray()
        if result is None:
            pub_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        else:
            _, cx, cy_roi, w, h, area = result
            cy_full     = cy_roi + roi_y0   # shift to full-image coords
            conf_approx = min(area / max(img_area, 1.0), 1.0)
            pub_msg.data = [1.0, cx, cy_full, w, h,
                            float(conf_approx), 1.0]
            self.get_logger().debug(
                f'[FRONT] Green mat: cx={cx:.1f} cy={cy_full:.1f} '
                f'w={w:.0f} h={h:.0f} conf={conf_approx:.4f}')

        self.front_pub.publish(pub_msg)
        if self.p_save_debug:
            self._save_debug(full_image, mask, result,
                             'latest_green_front.jpg', roi_y0=roi_y0)

    # ------------------------------------------------------------------ #
    #  Bottom camera callback — Phase 2                                    #
    # ------------------------------------------------------------------ #

    def bottom_image_callback(self, msg):
        # Rate limiter
        import time as _time
        now = _time.monotonic()
        if (now - self._bottom_last_t) < self._bottom_min_period:
            return
        self._bottom_last_t = now

        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Bottom CV Bridge error: {e}')
            return

        h, w = image.shape[:2]
        img_area = float(h * w)

        result, mask = self._detect_green(image)

        pub_msg = Float32MultiArray()
        if result is None:
            pub_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
        else:
            _, cx, cy, bw, bh, area = result
            coverage    = area / max(img_area, 1.0)
            conf_approx = min(coverage, 1.0)
            # Format: [detected, cx, cy, area_px2, coverage_frac, conf, phase=2]
            pub_msg.data = [1.0, cx, cy, float(area),
                            float(coverage), float(conf_approx), 2.0]
            self.get_logger().debug(
                f'[BOTTOM] Green mat: cx={cx:.1f} cy={cy:.1f} '
                f'coverage={coverage:.3f}')

        self.bottom_pub.publish(pub_msg)
        if self.p_save_debug:
            self._save_debug(image, mask, result, 'latest_green_bottom.jpg')


def main(args=None):
    rclpy.init(args=args)
    node = GreenDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
