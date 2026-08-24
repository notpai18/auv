#!/usr/bin/env python3
"""
underwater_view_node.py — makes the camera feed look like it is underwater.

Applies per-channel Beer-Lambert attenuation to the left stereo image using the
depth stereo_image_proc already computes, so distant objects wash out into the
water colour and red disappears before blue. This is the same model DAVE's
UnderwaterCamera plugin uses:

    e      = exp(-range * attenuation[c])
    out[c] = e * in[c] + (1 - e) * background[c]

WHY THIS EXISTS AS A ROS NODE RATHER THAN IN GAZEBO
---------------------------------------------------
Two things that look like they should handle this do not:

  * <fog> in the world's <scene> is a NO-OP in Gazebo Harmonic. SDFormat parses
    it, but the ogre2 render engine never implements it — "fog" does not appear
    anywhere in libgz-rendering8-ogre2.so or libgz-sim8.so, and a camera 18 m
    from the drums with end=18.0 still renders them fully saturated.

  * The water_volume visual in the world cannot tint these cameras, because
    they sit INSIDE it. A hollow box only tints what is seen THROUGH a face;
    from inside, the near face is behind the camera and the far faces are
    behind the props. That is geometry, not tuning — no transparency fixes it.

DAVE's own UnderwaterCamera plugin does work, but adopting it means converting
all three sensors to rgbd_camera, which renames every image topic and forces a
rewiring of the bridge, stereo_image_proc, and every vision node. This node
gets the same result using depth the stack already produces.

OUTPUT IS A SEPARATE TOPIC — NOTHING IS RETUNED
------------------------------------------------
Publishes to /auv/underwater_view/image_raw and leaves the raw feed untouched,
so the YOLO gate detector and the HSV green/blue detectors keep receiving
exactly the pixels they were tuned against. Point image_view or RViz at the new
topic for a submerged-looking view; leave the detectors where they are.

If you ever DO want the detectors to run on hazed images, remap their input to
this topic — and expect to retune every HSV threshold, because the whole point
of this node is that it shifts hue with distance.

A NOTE ON CHANNEL ORDER
-----------------------
DAVE's plugin converts its image to BGR (UnderwaterCamera.cc:328) and then
applies attenuation[0], loaded from <attenuationR>, to channel 0 — which in BGR
is BLUE. With its defaults (R 0.8, G 0.5, B 0.2) that attenuates blue fastest,
which is backwards for water. This node maps each coefficient to the channel it
is named for, so attenuation_r really does attenuate red.

Usage
-----
  ros2 run auv_vision underwater_view_node
  ros2 run image_view image_view --ros-args -r image:=/auv/underwater_view/image_raw
"""

import numpy as np
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge


class UnderwaterViewNode(Node):
    def __init__(self):
        super().__init__('underwater_view_node')
        self.cv_bridge = CvBridge()

        # ------------------------------------------------------------------ #
        #  Tunable parameters (overridable via YAML or CLI)                   #
        # ------------------------------------------------------------------ #
        # Attenuation coefficients, per metre, per colour channel. Red must be
        # the largest — losing red first is the single strongest visual cue
        # that a shot is underwater. Raise all three together for murkier
        # water. DAVE's "murky coastal" preset is 0.8 / 0.5 / 0.2, which is far
        # too aggressive for a pool: at 8 m it leaves 0.2% of the red.
        #
        # These defaults were chosen by rendering the real world at the AUV's
        # spawn pose with a depth camera and comparing presets. Reference
        # points, red retained by distance at 0.24/m:
        #     2 m 62%   5 m 30%   8 m 15%   15 m 3%
        # Two alternatives that were tried on the same frame:
        #     0.30/0.10/0.05 with bg 18/70/100  — moodier, gate gets dim
        #     0.18/0.06/0.03 with bg 45/120/155 — brighter, barely submerged
        self.declare_parameter('attenuation_r', 0.24)
        self.declare_parameter('attenuation_g', 0.08)
        self.declare_parameter('attenuation_b', 0.04)

        # The colour an infinitely distant surface fades to — i.e. the water
        # itself. 0-255 per channel. Keep it near the world's <background>
        # (0.05 0.25 0.45 -> about 13/64/115) so hazed objects blend into the
        # far field instead of banding against it.
        self.declare_parameter('background_r', 35)
        self.declare_parameter('background_g', 105)
        self.declare_parameter('background_b', 140)

        # Stereo geometry. baseline MUST match the camera separation in
        # auv_description/models/auv_box/model.sdf, same value the gate
        # localizer uses, or ranges (and therefore haze) come out scaled.
        self.declare_parameter('stereo_baseline', 0.12)

        # Range clamp. Pixels with no valid disparity are treated as max_range,
        # i.e. fully hazed — the correct default, since disparity drops out on
        # exactly the textureless far walls that should wash out anyway.
        self.declare_parameter('min_range', 0.2)
        self.declare_parameter('max_range', 19.1)   # matches the camera far clip

        self.declare_parameter('image_topic',       '/model/auv_box/stereo_front/left/image_raw')
        self.declare_parameter('disparity_topic',   '/disparity')
        self.declare_parameter('camera_info_topic', '/model/auv_box/stereo_front/left/camera_info')
        self.declare_parameter('output_topic',      '/auv/underwater_view/image_raw')

        g = lambda n: self.get_parameter(n).value
        # RGB order throughout; converted to the image's BGR order once, below.
        self.p_attenuation = np.array(
            [g('attenuation_r'), g('attenuation_g'), g('attenuation_b')], dtype=np.float32)
        self.p_background = np.array(
            [g('background_r'), g('background_g'), g('background_b')], dtype=np.float32)
        self.p_baseline  = g('stereo_baseline')
        self.p_min_range = g('min_range')
        self.p_max_range = g('max_range')
        out_topic        = g('output_topic')
        # ------------------------------------------------------------------ #

        # cv_bridge hands us bgr8, so flip both triples to BGR once here rather
        # than reindexing inside the per-frame path.
        self._att_bgr = self.p_attenuation[::-1].copy().reshape(1, 1, 3)
        self._bg_bgr = self.p_background[::-1].copy().reshape(1, 1, 3)

        # Depth->range scale factor, built lazily from the first CameraInfo.
        # A pixel's range is longer than its depth everywhere except the
        # principal point, by 1/cos(angle from the optical axis); this is the
        # same depth2range lookup table DAVE precomputes.
        self._range_scale = None
        self._fx = None

        self.info_sub = self.create_subscription(
            CameraInfo, g('camera_info_topic'), self._on_camera_info, 10)

        self.image_sub = message_filters.Subscriber(self, Image, g('image_topic'))
        self.disp_sub = message_filters.Subscriber(self, DisparityImage, g('disparity_topic'))
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.disp_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self._on_pair)

        self.pub = self.create_publisher(Image, out_topic, 10)

        self._warned_no_info = False
        self.get_logger().info(
            'Underwater view node started. Publishing %s '
            '(attenuation RGB %.2f/%.2f/%.2f per m, background RGB %d/%d/%d)' % (
                out_topic, *self.p_attenuation, *self.p_background.astype(int)))

    # ---------------------------------------------------------------------- #
    def _on_camera_info(self, msg: CameraInfo):
        """Build the depth->range scale table once; intrinsics do not change."""
        if self._range_scale is not None:
            return
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        if fx == 0 or fy == 0:
            self.get_logger().warn('CameraInfo has zero focal length; ignoring')
            return
        if cx == 0:
            cx = msg.width / 2.0
        if cy == 0:
            cy = msg.height / 2.0

        u = np.arange(msg.width, dtype=np.float32)
        v = np.arange(msg.height, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        self._range_scale = np.sqrt(
            ((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1.0).astype(np.float32)
        self._fx = fx
        self.get_logger().info(
            'Intrinsics locked: fx=%.1f fy=%.1f cx=%.1f cy=%.1f (%dx%d)' % (
                fx, fy, cx, cy, msg.width, msg.height))

    # ---------------------------------------------------------------------- #
    def _on_pair(self, img_msg: Image, disp_msg: DisparityImage):
        if self._range_scale is None:
            if not self._warned_no_info:
                self.get_logger().warn(
                    'Waiting for CameraInfo before hazing; no output yet')
                self._warned_no_info = True
            return

        img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        disp = self.cv_bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding='32FC1')

        if disp.shape[:2] != img.shape[:2] or disp.shape[:2] != self._range_scale.shape:
            self.get_logger().warn(
                'Size mismatch image%s disparity%s — skipping frame' % (
                    img.shape[:2], disp.shape[:2]), throttle_duration_sec=5.0)
            return

        # depth = fx * baseline / disparity, then range = depth / cos(theta).
        # Non-positive or non-finite disparity means "no match", which we treat
        # as maximum range so those pixels haze out fully rather than flicker.
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (self._fx * self.p_baseline) / disp
        rng = depth * self._range_scale
        rng = np.where(np.isfinite(rng) & (disp > 0.0), rng, self.p_max_range)
        np.clip(rng, self.p_min_range, self.p_max_range, out=rng)

        # out = e*in + (1-e)*background, evaluated per channel.
        e = np.exp(-rng[:, :, None] * self._att_bgr)
        out = e * img.astype(np.float32) + (1.0 - e) * self._bg_bgr

        out_msg = self.cv_bridge.cv2_to_imgmsg(
            np.clip(out, 0, 255).astype(np.uint8), encoding='bgr8')
        out_msg.header = img_msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = UnderwaterViewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
