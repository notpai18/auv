#!/usr/bin/env python3
"""
blue_bin_cv_detector_node.py
============================
Classical HSV colour-segmentation detector for the blue bin drum, replacing
the bottom-camera YOLO path that suffered from blue-water false positives.

Algorithm (bottom camera only — front camera keeps using YOLO):
  1. Convert frame to HSV.
  2. Threshold -> GREEN mask   (same HSV range as green_detector_node).
  3. Threshold -> BLUE mask    (tunable HSV range for the blue drum).
  4. Find blue contours.  Keep only those whose bounding box overlaps the
     dilated green region by at least `min_overlap_frac` of the blue blob's
     own area.
  5. Pick the largest qualifying blue blob -> centroid = bin pixel position.

Why this beats YOLO for this camera:
  - Open blue water cannot pass the "must overlap green" gate -- false positives
    become structurally impossible instead of relying on the fragile
    _bottom_on_green() workaround in the planner.
  - Zero training data needed.
  - Runs on CPU at full 15 Hz without GPU overhead.
  - In sim, colours are deterministic; HSV ranges are perfectly reliable.
  - On real hardware: retune HSV ranges via the YAML parameters -- no retraining.

Publications
------------
  /auv/blue_bin_detection_bottom_2d  -- Float32MultiArray
      [detected(0/1), cx_px, cy_px, w_px, h_px, conf, 2.0]
      conf = blue_blob_area / (img_area * 0.05)  clamped 0-1

  Same format as blue_bin_detector_node so blue_bin_localizer_node and
  green_navigator_node need zero changes.

Debug images
------------
  <debug_dir>/latest_blue_bin_cv_bottom.jpg  -- always the most recent frame
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time


class BlueBinCvDetectorNode(Node):

    def __init__(self):
        super().__init__('blue_bin_cv_detector_node')
        self.cv_bridge = CvBridge()

        # ------------------------------------------------------------------ #
        #  ROS 2 Parameters -- all tunable without recompiling               #
        # ------------------------------------------------------------------ #

        # Green mat HSV thresholds (OpenCV: H 0-179, S/V 0-255)
        # Match green_detector_node so both nodes agree on what is "green mat".
        # Sim mat colour: ambient 0.2 0.8 0.2 -> RGB(51,204,51)
        #                 -> HSV approx (60 deg, 75%, 80%) -> OpenCV (60, 191, 204)
        self.declare_parameter('green_h_low',  38)
        self.declare_parameter('green_h_high', 82)
        self.declare_parameter('green_s_low',  80)
        self.declare_parameter('green_v_low',  60)

        # Blue drum HSV thresholds
        # Sim drum colour: ambient 0 0 1 -> pure RGB blue (0,0,255)
        #                  -> HSV (240 deg, 100%, 100%) -> OpenCV H=120, S=255, V=255
        # Add margin for underwater lighting attenuation and slight fog.
        self.declare_parameter('blue_h_low',   100)
        self.declare_parameter('blue_h_high',  130)
        self.declare_parameter('blue_s_low',   120)
        # V=140 minimum: the blue drum in sim is bright (V~255) but the
        # underwater background water is dark (V~115 for Gazebo ambient
        # 0.05,0.25,0.45).  Threshold at 140 cleanly separates them while
        # still catching the drum even with mild lighting attenuation.
        self.declare_parameter('blue_v_low',   140)

        # Detection quality gates
        self.declare_parameter('min_blue_area_px',  80.0)
        # Fraction of blue blob area that must overlap the dilated green mask.
        self.declare_parameter('min_overlap_frac',   0.20)
        # Green dilation: 20px catches drums sitting right at the mat edge
        # without reaching far enough into open water to cause false positives.
        self.declare_parameter('green_dilate_px',    20)
        # Morphological cleanup kernel size for both masks.
        self.declare_parameter('morph_kernel_px',     5)

        # Misc
        self.declare_parameter('debug_image_dir', '')
        self.declare_parameter('process_rate_hz', 5.0)
        self.declare_parameter('save_debug_images', False)

        # Bind ----------------------------------------------------------------
        p = self.get_parameter
        self.lower_green = np.array(
            [p('green_h_low').value, p('green_s_low').value, p('green_v_low').value],
            dtype=np.uint8)
        self.upper_green = np.array(
            [p('green_h_high').value, 255, 255], dtype=np.uint8)

        self.lower_blue = np.array(
            [p('blue_h_low').value, p('blue_s_low').value, p('blue_v_low').value],
            dtype=np.uint8)
        self.upper_blue = np.array(
            [p('blue_h_high').value, 255, 255], dtype=np.uint8)

        self.min_blue_area    = p('min_blue_area_px').value
        self.min_overlap_frac = p('min_overlap_frac').value
        self.green_dilate_px  = int(p('green_dilate_px').value)
        self.morph_k          = int(p('morph_kernel_px').value)
        self.process_interval = 1.0 / p('process_rate_hz').value
        self.p_save_debug     = p('save_debug_images').value

        raw_debug = p('debug_image_dir').value
        self.debug_dir = (
            raw_debug if raw_debug
            else os.path.join(os.path.expanduser('~'), '.ros', 'blue_bin_cv_detections')
        )
        os.makedirs(self.debug_dir, exist_ok=True)
        # ------------------------------------------------------------------ #

        self.pub = self.create_publisher(
            Float32MultiArray, '/auv/blue_bin_detection_bottom_2d', 10)

        self.sub = self.create_subscription(
            Image,
            '/model/auv_box/bottom/image_raw',
            self._image_callback,
            10)

        self._last_proc_time = 0.0

        self.get_logger().info(
            f'Blue Bin CV Detector started! '
            f'Green HSV: H[{p("green_h_low").value},{p("green_h_high").value}] '
            f'S[{p("green_s_low").value},255] V[{p("green_v_low").value},255] | '
            f'Blue HSV: H[{p("blue_h_low").value},{p("blue_h_high").value}] '
            f'S[{p("blue_s_low").value},255] V[{p("blue_v_low").value},255] | '
            f'Min area: {self.min_blue_area:.0f}px2  '
            f'Overlap: {self.min_overlap_frac:.0%}  '
            f'Green dilate: {self.green_dilate_px}px'
        )

    # ------------------------------------------------------------------ #
    #  Core detection                                                      #
    # ------------------------------------------------------------------ #

    def _detect(self, bgr):
        """
        Detect a blue blob that overlaps the green mat region.

        Returns
        -------
        best_blob : (area, cx, cy, w, h, overlap_frac) or None
        vis       : annotated BGR debug image
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_k, self.morph_k))

        # Green mask
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  k, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k, iterations=2)

        # Dilate green mask so drums at the mat edge still qualify
        if self.green_dilate_px > 0:
            dk = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.green_dilate_px + 1, 2 * self.green_dilate_px + 1))
            green_dilated = cv2.dilate(green_mask, dk, iterations=1)
        else:
            green_dilated = green_mask

        # Blue mask
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  k, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k, iterations=1)

        # Find blue contours that overlap the green region
        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_blob = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_blue_area:
                continue

            # Single-blob mask for overlap calculation
            blob_mask = np.zeros_like(blue_mask)
            cv2.drawContours(blob_mask, [cnt], -1, 255, cv2.FILLED)

            overlap_px   = float(cv2.countNonZero(
                cv2.bitwise_and(blob_mask, green_dilated)))
            overlap_frac = overlap_px / max(float(area), 1.0)

            if overlap_frac < self.min_overlap_frac:
                continue   # not on the green mat

            x, y, w, h = cv2.boundingRect(cnt)
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)

            if best_blob is None or area > best_blob[0]:
                best_blob = (area, cx, cy, float(w), float(h), overlap_frac)

        # Annotated debug image
        vis = bgr.copy()
        ov1 = vis.copy()
        ov1[green_dilated > 0] = (160, 255, 160)
        cv2.addWeighted(ov1, 0.22, vis, 0.78, 0, vis)
        ov2 = vis.copy()
        ov2[blue_mask > 0] = (255, 80, 80)
        cv2.addWeighted(ov2, 0.35, vis, 0.65, 0, vis)

        if best_blob is not None:
            area, cx, cy, w, h, overlap_frac = best_blob
            x0 = int(cx - w / 2)
            y0 = int(cy - h / 2)
            cv2.rectangle(vis, (x0, y0), (x0 + int(w), y0 + int(h)),
                          (0, 120, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 7, (0, 60, 255), -1)
            cv2.putText(
                vis,
                f'BLUE BIN ({cx:.0f},{cy:.0f}) ovlp={overlap_frac:.0%}',
                (x0, max(y0 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        else:
            cv2.putText(vis, 'no blue bin on green', (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 200), 2)

        return best_blob, vis

    # ------------------------------------------------------------------ #
    #  ROS callback                                                        #
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg):
        now = time.time()
        if now - self._last_proc_time < self.process_interval:
            return
        self._last_proc_time = now

        try:
            bgr = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        h_img, w_img = bgr.shape[:2]
        img_area = float(h_img * w_img)

        best_blob, vis = self._detect(bgr)

        # Publish
        out = Float32MultiArray()
        if best_blob is None:
            out.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
        else:
            area, cx, cy, w, h, overlap_frac = best_blob
            # conf proxy: area relative to 5% of image area -> 0..1
            conf = float(min(area / max(img_area * 0.05, 1.0), 1.0))
            out.data = [1.0, float(cx), float(cy), float(w), float(h), conf, 2.0]
            self.get_logger().debug(
                f'[BLUE BIN CV] cx={cx:.1f} cy={cy:.1f} '
                f'w={w:.0f} h={h:.0f} conf={conf:.2f} ovlp={overlap_frac:.0%}')

        self.pub.publish(out)

        # Debug image (only when explicitly enabled)
        if self.p_save_debug:
            try:
                cv2.imwrite(
                    os.path.join(self.debug_dir, 'latest_blue_bin_cv_bottom.jpg'),
                    vis)
            except Exception as e:
                self.get_logger().warn(f'Debug image save failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BlueBinCvDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
