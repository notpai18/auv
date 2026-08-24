#!/usr/bin/env python3
"""
trail_mapper_node.py  —  AUV Trail Mapper + Gate/Bin Marker Mapping

Subscribes to /auv/odom and builds a 3-D trajectory trail.
Also subscribes to /auv/gate_position_3d and /auv/bin_position_3d
(Float32MultiArray, BODY frame: [x_fwd, y_left, z_up, confidence] —
same format gate_localizer_node already publishes) and converts each
detection into a fixed ODOM-frame marker, using the same transform
math already validated in gate_navigator_node.py (camera-offset
correction + rotation by current yaw).

Publishes:
    /auv/trail_map       nav_msgs/Path              — full trail (1 Hz)
    /auv/object_markers  visualization_msgs/MarkerArray
                          — gate = red X, bin = blue circle
                          (viewable live in RViz)

Subscribes:
    /auv/odom              nav_msgs/Odometry
    /auv/gate_position_3d  std_msgs/Float32MultiArray  [x_fwd, y_left, z_up, conf]
                            (published by gate_localizer_node)
    /auv/blue_bin_position_3d  std_msgs/Float32MultiArray  [x_fwd, y_left, z_up, conf, source]
                            (published by blue_bin_localizer_node)
    /auv/green_position_3d std_msgs/Float32MultiArray  [x_fwd, y_left, z_up, conf, source]
                            (published by green_localizer_node)

On shutdown, saves:
    - trail_<timestamp>.csv     full pose trail
    - markers_<timestamp>.csv   every deduped gate/bin detection
    - map_<timestamp>.png       2D top-down map: trail line + gate (X)
                                 + bin (O) — Bonus Mapping Challenge deliverable

Usage
-----
ros2 run auv_telemetry trail_mapper_node
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import math
import os
import csv
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')   # headless-safe backend — no display needed onboard
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mcollections

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TrailMapperNode(Node):

    # ── Tunable parameters ────────────────────────────────────────────────
    MIN_DIST_M = 0.10       # min AUV movement before a new trail point is stored
    PUBLISH_HZ = 1.0        # trail/marker publish rate
    MAP_EXPORT_PERIOD_SEC = 30.0   # periodic map re-save, independent of shutdown

    # Minimum confidence to accept a gate/bin detection at all.
    # Matches gate_localizer_node / gate_navigator_node's own threshold.
    # Gate and bin detections both come from YOLO, whose confidence is a
    # real 0-1 detection probability (observed ~0.7-0.95), so this bar is
    # meaningful for them.
    CONFIDENCE_THRESHOLD = 0.10

    # green_detector_node computes "confidence" as contour_area / full_image
    # area — a raw pixel-coverage fraction, not a detection probability. Even
    # a real, tracking detection of the green mat only reaches ~0.02-0.05 by
    # that formula (confirmed from live mission logs), so the shared
    # CONFIDENCE_THRESHOLD above would silently discard every green
    # detection. green_localizer_node already knows this and uses 0.001 as
    # its own accept bar — match that here instead of YOLO's scale.
    GREEN_CONFIDENCE_THRESHOLD = 0.001

    # Persistent bin/green tracker. Each is one physical object in the pool,
    # so repeated detections should smooth-update the same estimate instead
    # of scattering into many markers as odometry drift shifts each frame's
    # projected position — the same problem the gate tracker below solves.
    OBJECT_ASSOCIATION_DIST_M = 1.0
    OBJECT_MAX_JUMP_M = 2.0
    OBJECT_SMOOTH_ALPHA = 0.20

    # Persistent gate tracker. A gate is one physical object, so repeated
    # detections should update the same estimate instead of creating many Xs.
    GATE_ASSOCIATION_DIST_M = 1.25
    GATE_MAX_JUMP_M = 1.50
    GATE_SMOOTH_ALPHA = 0.20
    GATE_MIN_UPDATES = 3

    # Gate geometry — fixed dimensions from the pool world file (camera_test.sdf):
    # two 0.05m-radius posts at X=17.5 / X=20.5 (Y=12 fixed), each 1.524m tall.
    GATE_WIDTH_M = 3.0        # post-to-post spacing (transverse axis)
    GATE_HEIGHT_M = 1.524     # post length (vertical axis)
    GATE_THICKNESS_M = 0.10   # post diameter (2 * 0.05m radius)

    # The gate is a fixed, non-moving fixture built perpendicular to the
    # 25m east/west walls (its post-to-post axis runs along world X, same
    # as the pool's north/south walls) — confirmed from camera_test.sdf.
    # gate_localizer_node only ever publishes [x,y,z,confidence] (no live
    # yaw), so rather than guessing orientation from noisy vision data we
    # use this known, constant orientation for every gate marker/render.
    GATE_FIXED_YAW_RAD = 0.0

    # Pool bounds (from camera_test.sdf: pool_floor size 28 x 25, corner at origin).
    POOL_LENGTH_M = 28.0   # X extent (spanned by the north/south 28m walls)
    POOL_WIDTH_M = 25.0    # Y extent (spanned by the east/west 25m walls)
    POOL_FLOOR_Z = -5.0
    WATER_SURFACE_Z = 0.0

    # Green target-mat geometry (fixed 8m x 2m mat, long axis along world X —
    # from camera_test.sdf target_zone box size).
    GREEN_LENGTH_M = 8.0
    GREEN_WIDTH_M = 2.0

    # Bin (drum) geometry — fixed cylinder, from camera_test.sdf drum_blue.
    BIN_RADIUS_M = 0.3
    BIN_HEIGHT_M = 0.5

    # Camera mounting offset relative to base_link (from model.sdf),
    # same constants gate_navigator_node.py uses to correct localizer
    # output before treating it as a base_link-relative measurement.
    #   Left camera pose: (0.30, +0.06, 0) relative to base_link
    CAM_OFFSET_FWD  = 0.30   # metres forward
    CAM_OFFSET_LEFT = 0.06   # metres left
    CAM_OFFSET_UP   = 0.00   # metres up; stereo camera is currently at z=0
    # ─────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('trail_mapper_node')

        # ── Trail storage ────────────────────────────────────────────────
        self._trail: list[tuple] = []
        self._last_x = self._last_y = self._last_z = None
        self._total_dist: float = 0.0

        # ── Current pose (needed to transform body-frame detections) ──────
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.current_yaw = None

        # ── Marker storage: separate lists per class, in ODOM frame ────────
        # Each entry: (x, y, z, stamp)
        # One persistent gate estimate:
        # (x, y, z, yaw_odom, stamp, update_count)
        self._gate_track = None
        self._gate_markers: list[tuple] = []   # retained for export compatibility
        self._bin_markers: list[tuple] = []
        self._green_markers: list[tuple] = []

        # ── Log files ────────────────────────────────────────────────────
        log_dir = os.path.join(os.path.expanduser('~'), 'auv_ws', 'mission_logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        self._csv_path = os.path.join(log_dir, f'trail_{ts}.csv')
        self._csv_file = open(self._csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ['index', 'x_m', 'y_m', 'z_m', 'yaw_deg', 'dist_from_prev_m', 'total_dist_m']
        )

        self._markers_csv_path = os.path.join(log_dir, f'markers_{ts}.csv')
        self._markers_csv_file = open(self._markers_csv_path, 'w', newline='')
        self._markers_csv_writer = csv.writer(self._markers_csv_file)
        self._markers_csv_writer.writerow(['class', 'x_m', 'y_m', 'z_m', 'stamp_sec'])

        self._map_png_path = os.path.join(log_dir, f'map_{ts}.png')
        self._map_html_path = os.path.join(log_dir, f'map_3d_{ts}.html')

        # ── ROS interfaces ───────────────────────────────────────────────
        self._odom_sub = self.create_subscription(
            Odometry, '/auv/odom', self._odom_callback, 10
        )
        self._gate_sub = self.create_subscription(
            Float32MultiArray, '/auv/gate_position_3d', self._gate_callback, 10
        )
        self._bin_sub = self.create_subscription(
            Float32MultiArray, '/auv/blue_bin_position_3d', self._bin_callback, 10
        )
        self._green_sub = self.create_subscription(
            Float32MultiArray, '/auv/green_position_3d', self._green_callback, 10
        )

        self._path_pub = self.create_publisher(Path, '/auv/trail_map', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/auv/object_markers', 10)

        self.create_timer(1.0 / self.PUBLISH_HZ, self._publish_path)
        self.create_timer(1.0 / self.PUBLISH_HZ, self._publish_markers)
        # Re-save the map periodically, not just on clean shutdown. A run
        # killed forcefully (closed terminal, crash, second Ctrl+C) skips
        # destroy_node() entirely and loses the map for that run even though
        # the trail/markers CSVs (flushed every write) survive fine — this
        # keeps a recent snapshot on disk no matter how the run ends.
        self.create_timer(self.MAP_EXPORT_PERIOD_SEC, self._periodic_map_export)

        self.get_logger().info(
            f'Trail Mapper started | MIN_DIST={self.MIN_DIST_M}m | '
            f'gate_tracking={self.GATE_ASSOCIATION_DIST_M}m | '
            f'publish={self.PUBLISH_HZ}Hz | csv={self._csv_path}'
        )

    # ── Odometry callback ────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.current_x, self.current_y, self.current_z, self.current_yaw = x, y, z, yaw

        # ── Trail downsampling (unchanged from original) ──────────────────
        if self._last_x is not None:
            dist = math.sqrt(
                (x - self._last_x) ** 2 +
                (y - self._last_y) ** 2 +
                (z - self._last_z) ** 2
            )
            if dist < self.MIN_DIST_M:
                return
            self._total_dist += dist
        else:
            dist = 0.0

        yaw_deg = math.degrees(yaw)
        self._trail.append((x, y, z, yaw_deg, msg.header.stamp, q))
        self._last_x, self._last_y, self._last_z = x, y, z

        idx = len(self._trail)
        self._csv_writer.writerow([
            idx, f'{x:.4f}', f'{y:.4f}', f'{z:.4f}',
            f'{yaw_deg:.2f}', f'{dist:.4f}', f'{self._total_dist:.4f}',
        ])
        self._csv_file.flush()

        if idx % 20 == 0:
            self.get_logger().info(
                f'[Trail] {idx} poses recorded | '
                f'total_dist={self._total_dist:.2f}m | '
                f'pos=({x:.2f}, {y:.2f}, {z:.2f})'
            )

    # ── Detection callbacks ──────────────────────────────────────────────
    # msg.data = [x_fwd, y_left, z_up, confidence] — BODY frame,
    # same shape gate_localizer_node already publishes on
    # /auv/gate_position_3d. bin_localizer_node (once built) should
    # publish the same shape on /auv/bin_position_3d.
    #
    # x/y/z are BODY-frame offsets from the camera. X/Y are corrected for
    # camera mounting offset and rotated by current yaw; Z is anchored to
    # current odometry Z because yaw does not affect the vertical axis.

    def _gate_callback(self, msg: Float32MultiArray):
        self._handle_gate_detection(msg)

    def _bin_callback(self, msg: Float32MultiArray):
        self._handle_tracked_detection('bin', msg, self._bin_markers)

    def _green_callback(self, msg: Float32MultiArray):
        self._handle_tracked_detection('green', msg, self._green_markers)

    def _handle_gate_detection(self, msg: Float32MultiArray):
        """
        Track the gate as one persistent physical object.

        gate_localizer_node only ever publishes [x, y, z, confidence] — it
        does not measure orientation. The gate is a fixed, non-moving
        fixture whose real-world orientation is known in advance (built
        perpendicular to the pool's 25m walls, its post-to-post axis along
        world X — see GATE_FIXED_YAW_RAD), so that known constant is used
        instead of trying to infer yaw from noisy vision data.
        Position is exponentially smoothed; isolated large jumps are rejected.
        """
        if self.current_x is None or self.current_y is None or self.current_z is None:
            return

        if len(msg.data) < 4:
            return

        x_fwd, y_left, z_up, confidence = msg.data[0:4]

        if confidence < self.CONFIDENCE_THRESHOLD:
            return

        if not all(math.isfinite(v) for v in (x_fwd, y_left, z_up)):
            return

        # Same camera -> base_link correction used by the navigator.
        x_body = x_fwd - self.CAM_OFFSET_FWD
        y_body = y_left - self.CAM_OFFSET_LEFT
        z_body = z_up - self.CAM_OFFSET_UP

        yaw = self.current_yaw

        odom_x = self.current_x + x_body * math.cos(yaw) - y_body * math.sin(yaw)
        odom_y = self.current_y + x_body * math.sin(yaw) + y_body * math.cos(yaw)
        odom_z = self.current_z + z_body

        if not all(math.isfinite(v) for v in (odom_x, odom_y, odom_z)):
            return

        # Orientation is a known constant, not a per-frame measurement.
        gate_yaw_odom = self.GATE_FIXED_YAW_RAD

        # First valid detection initializes the persistent gate.
        if self._gate_track is None:
            self._gate_track = (
                odom_x, odom_y, odom_z,
                gate_yaw_odom,
                self.get_clock().now().to_msg(),
                1
            )
            self._sync_gate_marker_storage()
            return

        gx, gy, gz, gyaw, old_stamp, update_count = self._gate_track

        jump = math.sqrt(
            (odom_x - gx) ** 2 +
            (odom_y - gy) ** 2 +
            (odom_z - gz) ** 2
        )

        # Reject a clearly bad stereo/localization frame.
        if jump > self.GATE_MAX_JUMP_M:
            self.get_logger().debug(
                f'[GateTrack] Rejected jump={jump:.2f}m '
                f'> {self.GATE_MAX_JUMP_M:.2f}m'
            )
            return

        # If the detection is close enough, it is the same physical gate.
        # Smooth position instead of creating another marker.
        if jump <= self.GATE_ASSOCIATION_DIST_M:
            a = self.GATE_SMOOTH_ALPHA

            gx = (1.0 - a) * gx + a * odom_x
            gy = (1.0 - a) * gy + a * odom_y
            gz = (1.0 - a) * gz + a * odom_z
            gyaw = self.GATE_FIXED_YAW_RAD   # known constant — nothing to smooth

            update_count += 1
            stamp = self.get_clock().now().to_msg()

            self._gate_track = (
                gx, gy, gz, gyaw, stamp, update_count
            )
            self._sync_gate_marker_storage()
            return

        # A far-away detection is not immediately treated as a second gate.
        # Require the current gate tracker to be established first and ignore
        # isolated far detections; this prevents the exact multi-X problem.
        self.get_logger().debug(
            f'[GateTrack] Ignored distant detection: '
            f'{jump:.2f}m from tracked gate'
        )

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _sync_gate_marker_storage(self):
        """Keep the legacy list/export path synchronized with the one gate."""
        if self._gate_track is None:
            self._gate_markers = []
            return

        gx, gy, gz, gyaw, stamp, update_count = self._gate_track

        # Existing CSV/export code expects 4-tuples.
        self._gate_markers = [(gx, gy, gz, stamp)]

    def _handle_tracked_detection(self, label: str, msg: Float32MultiArray, tracks: list):
        """
        Track each detected bin/green-mat sighting as one of a small set of
        persistent physical objects, the same strategy _handle_gate_detection
        uses for the gate. A detection within OBJECT_ASSOCIATION_DIST_M of an
        existing track smooth-updates that track in place; one further out
        (but still under OBJECT_MAX_JUMP_M) is treated as ambiguous noise and
        dropped; anything else starts a new track. Without this, odometry
        drift between sightings of the *same* stationary object would
        otherwise scatter it across many separate markers.

        Each track entry: (x, y, z, stamp, update_count).
        """
        if self.current_x is None or self.current_z is None:
            return   # no complete odom pose yet — can't transform body frame to odom

        if len(msg.data) < 4:
            return

        x_fwd, y_left, z_up, confidence = msg.data[0], msg.data[1], msg.data[2], msg.data[3]

        threshold = self.GREEN_CONFIDENCE_THRESHOLD if label == 'green' else self.CONFIDENCE_THRESHOLD
        if confidence < threshold:
            return

        if not all(math.isfinite(v) for v in (x_fwd, y_left, z_up)):
            return

        # Same correction gate_navigator_node applies: subtract camera
        # mounting offset so this reflects a base_link-relative measurement.
        x_body = x_fwd - self.CAM_OFFSET_FWD
        y_body = y_left - self.CAM_OFFSET_LEFT
        z_body = z_up - self.CAM_OFFSET_UP
        yaw = self.current_yaw

        # Rotate body-frame horizontal offset into odom frame and add robot
        # position. Z is the vertical body coordinate, so yaw does not affect
        # it; it must, however, be anchored to the AUV's current odom Z.
        odom_x = self.current_x + x_body * math.cos(yaw) - y_body * math.sin(yaw)
        odom_y = self.current_y + x_body * math.sin(yaw) + y_body * math.cos(yaw)
        odom_z = self.current_z + z_body

        if not all(math.isfinite(v) for v in (odom_x, odom_y, odom_z)):
            return

        stamp = self.get_clock().now().to_msg()

        # Associate with the nearest existing track of this class.
        nearest_idx, nearest_dist = None, None
        for idx, (tx, ty, tz, _stamp, _count) in enumerate(tracks):
            dist = math.sqrt((odom_x - tx) ** 2 + (odom_y - ty) ** 2 + (odom_z - tz) ** 2)
            if nearest_dist is None or dist < nearest_dist:
                nearest_idx, nearest_dist = idx, dist

        if nearest_dist is not None and nearest_dist <= self.OBJECT_MAX_JUMP_M:
            if nearest_dist > self.OBJECT_ASSOCIATION_DIST_M:
                return   # ambiguous distance — likely noise, not a new object

            a = self.OBJECT_SMOOTH_ALPHA
            tx, ty, tz, _old_stamp, count = tracks[nearest_idx]
            tx = (1.0 - a) * tx + a * odom_x
            ty = (1.0 - a) * ty + a * odom_y
            tz = (1.0 - a) * tz + a * odom_z
            count += 1
            tracks[nearest_idx] = (tx, ty, tz, stamp, count)
            self._log_marker_csv(label, tx, ty, tz, stamp)
            return

        # No track close enough — this is a newly discovered object.
        tracks.append((odom_x, odom_y, odom_z, stamp, 1))
        self._log_marker_csv(label, odom_x, odom_y, odom_z, stamp)

        self.get_logger().info(
            f'[Marker] New {label} at odom=({odom_x:.2f}, {odom_y:.2f}, {odom_z:.2f}) '
            f'| conf={confidence:.2f} | total {label}s so far: {len(tracks)}'
        )

    def _best_track(self, tracks: list):
        """The single most-confirmed track (highest update_count), or None.

        There is only ever one physical green mat, but the mat's contour-
        based confidence threshold is deliberately very low (0.001 — real
        readings only reach ~0.02-0.05, see GREEN_CONFIDENCE_THRESHOLD), so
        a noisy or momentarily-distant reading can start a second/third
        track alongside the real one. Rendering all of them makes the map
        show several green mats that don't exist; the one with the most
        confirming updates is overwhelmingly likely to be the real one.
        """
        if not tracks:
            return None
        return max(tracks, key=lambda t: t[4])

    def _log_marker_csv(self, label, x, y, z, stamp):
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        self._markers_csv_writer.writerow([
            label, f'{x:.4f}', f'{y:.4f}', f'{z:.4f}', f'{stamp_sec:.3f}'
        ])
        self._markers_csv_file.flush()

    # ── Path publisher ───────────────────────────────────────────────────

    def _publish_path(self):
        if not self._trail:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for (x, y, z, _yaw_deg, stamp, q) in self._trail:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = 'map'
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = z
            ps.pose.orientation = q
            path_msg.poses.append(ps)

        self._path_pub.publish(path_msg)

    # ── Marker publisher (for live viewing in RViz) ─────────────────────

    def _publish_markers(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        marker_id = 0

        # Gate -> one persistent thin cuboid representing the actual gate
        # region. It is oriented using the gate transverse-axis yaw.
        if self._gate_track is not None:
            x, y, z, gate_yaw_odom, _stamp, update_count = self._gate_track

            if gate_yaw_odom is not None:
                marker_array.markers.append(
                    self._make_gate_cuboid_marker(
                        marker_id, x, y, z, gate_yaw_odom, now
                    )
                )
            else:
                # Before orientation becomes valid, still show one stable
                # location rather than many crosses.
                marker_array.markers.append(
                    self._make_cross_marker(marker_id, x, y, z, now)
                )
            marker_id += 1

        # Bin markers -> blue circle/sphere
        for (x, y, z, _stamp, _count) in self._bin_markers:
            marker_array.markers.append(
                self._make_sphere_marker(marker_id, x, y, z, now)
            )
            marker_id += 1

        if marker_array.markers:
            self._marker_pub.publish(marker_array)

    def _make_cross_marker(self, marker_id, x, y, z, stamp, size=0.3):
        """Gate marker: red X, built from two crossed LINE_LIST segments."""
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = stamp
        m.ns = 'gate_markers'
        m.id = marker_id
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.05   # line thickness
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0

        half = size / 2.0
        p1, p2 = Point(x=x - half, y=y - half, z=z), Point(x=x + half, y=y + half, z=z)
        p3, p4 = Point(x=x - half, y=y + half, z=z), Point(x=x + half, y=y - half, z=z)
        m.points = [p1, p2, p3, p4]
        return m

    def _make_gate_cuboid_marker(
        self, marker_id, x, y, z, gate_yaw_odom, stamp
    ):
        """
        Thin CUBE representing the gate plane.

        CUBE local X = gate width/transverse axis
        CUBE local Y = gate thickness/normal direction
        CUBE local Z = gate height
        """
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = stamp
        m.ns = 'gate_region'
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z

        # Yaw of local X axis is the gate transverse-axis yaw.
        half = 0.5 * gate_yaw_odom
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = math.sin(half)
        m.pose.orientation.w = math.cos(half)

        m.scale.x = self.GATE_WIDTH_M
        m.scale.y = self.GATE_THICKNESS_M
        m.scale.z = self.GATE_HEIGHT_M

        # Keep the existing gate visual convention.
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 0.25

        return m

    def _make_sphere_marker(self, marker_id, x, y, z, stamp, size=None):
        """Bin marker: blue sphere/circle."""
        if size is None:
            size = 2.0 * self.BIN_RADIUS_M   # marker scale is diameter
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = stamp
        m.ns = 'bin_markers'
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = size
        m.color.r = 0.0
        m.color.g = 0.3
        m.color.b = 1.0
        m.color.a = 1.0
        return m

    # ── 2D map export (Bonus Mapping Challenge deliverable) ─────────────

    def _export_2d_map(self):
        """
        Saves a top-down 2D plot: AUV trail line (colored by mission
        progress), gate/green/bin as true-to-scale physical footprints
        using the pool's known, fixed geometry, with the pool bounds.
        """
        L, W = self.POOL_LENGTH_M, self.POOL_WIDTH_M
        fig, ax = plt.subplots(figsize=(11, 9.5))

        # Pool bounds + wall thickness (walls are ~0.2m thick in the real pool)
        wall_t = 0.2
        ax.add_patch(patches.Rectangle(
            (-wall_t, -wall_t), L + 2 * wall_t, W + 2 * wall_t,
            facecolor='#37474f', edgecolor='none', zorder=0))
        pool = patches.Rectangle((0, 0), L, W, linewidth=0, facecolor='#e0f7fa', zorder=1,
                                  label=f'Pool ({L:.0f}x{W:.0f}m)')
        ax.add_patch(pool)

        # Green target mat — fixed footprint, long axis parallel to the
        # 28m walls (perpendicular to the 25m east/west walls). Only the
        # single best-confirmed track is drawn — see _best_track().
        best_green = self._best_track(self._green_markers)
        if best_green is not None:
            gx, gy, gz, _stamp, _count = best_green
            rect = patches.Rectangle(
                (gx - self.GREEN_LENGTH_M / 2.0, gy - self.GREEN_WIDTH_M / 2.0),
                self.GREEN_LENGTH_M, self.GREEN_WIDTH_M,
                linewidth=2, edgecolor='#1b5e20', facecolor='#a5d6a7', zorder=2,
                label='Green Mat')
            ax.add_patch(rect)

        if self._trail:
            xs = np.array([p[0] for p in self._trail])
            ys = np.array([p[1] for p in self._trail])
            pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            progress = np.linspace(0, 1, max(len(segs), 1))
            lc = mcollections.LineCollection(
                segs, cmap='plasma', linewidths=3.0, zorder=3)
            lc.set_array(progress)
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label('Mission progress')
            ax.plot(xs[0], ys[0], 'o', color='#00c853', markersize=12,
                    markeredgecolor='black', label='Start', zorder=5)
            ax.plot(xs[-1], ys[-1], 's', color='#d50000', markersize=12,
                    markeredgecolor='black', label='End', zorder=5)

        # Gate — fixed-size footprint at its tracked position, oriented
        # perpendicular to the 25m east/west walls (known constant yaw).
        if self._gate_track is not None:
            gx, gy, gz, gyaw, _stamp, _update_count = self._gate_track
            width = self.GATE_WIDTH_M
            thick = self.GATE_THICKNESS_M
            dx, dy = width / 2.0, thick / 2.0
            corners = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
            rot = np.array([[np.cos(gyaw), -np.sin(gyaw)],
                             [np.sin(gyaw), np.cos(gyaw)]])
            rotated = corners @ rot.T + np.array([gx, gy])
            ax.add_patch(patches.Polygon(
                rotated, closed=True, linewidth=2, edgecolor='#b71c1c',
                facecolor='#ef9a9a', zorder=4, label='Gate'))
            # Post markers at the two ends for visual clarity.
            post_offset = np.array([np.cos(gyaw), np.sin(gyaw)]) * (width / 2.0)
            for sign in (1, -1):
                px, py = gx + sign * post_offset[0], gy + sign * post_offset[1]
                ax.add_patch(patches.Circle((px, py), thick, color='black', zorder=5))

        # Bins — fixed-radius circular footprints at their tracked positions.
        bin_added = False
        for (bx, by, bz, _stamp, _count) in self._bin_markers:
            circle = patches.Circle(
                (bx, by), self.BIN_RADIUS_M, linewidth=2, edgecolor='#0d47a1',
                facecolor='#90caf9', zorder=4, label='Bin' if not bin_added else "")
            ax.add_patch(circle)
            bin_added = True

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        title = 'AUV Mission Map — Physical Footprints'
        if self._trail:
            title += f'  |  distance={self._total_dist:.1f}m'
        ax.set_title(title)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linestyle='--', alpha=0.4, zorder=1, color='white')

        ax.set_xlim(-2, L + 2)
        ax.set_ylim(-2, W + 2)

        # Write to a temp path and atomically rename into place. savefig()
        # takes long enough (large figure, dpi=300) that a forceful kill
        # during shutdown (e.g. a second Ctrl+C) can land mid-write and
        # leave a truncated, unopenable PNG at the real path. Renaming only
        # after the write fully completes means the final path is always
        # either a complete file or simply absent — never a corrupt one.
        tmp_path = self._map_png_path + '.tmp'
        fig.savefig(tmp_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        os.replace(tmp_path, self._map_png_path)

        self.get_logger().info(f'[Map] 2D mission map saved -> {self._map_png_path}')

    # Fixed triangle indices for a closed 8-vertex box mesh (see _box_mesh).
    _BOX_I = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
    _BOX_J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    _BOX_K = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]

    def _box_mesh(self, cx, cy, cz, size_x, size_y, size_z, yaw=0.0):
        """
        8 vertices of a solid box (size_x/y/z full dimensions) centered at
        (cx, cy, cz), rotated by yaw about Z. Pair with _BOX_I/J/K for a
        closed go.Mesh3d. Used for the gate posts/crossbar and green mat so
        every fixed-size object renders as a real solid, not a flat quad.
        """
        hx, hy, hz = size_x / 2.0, size_y / 2.0, size_z / 2.0
        local = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        world = local @ rot.T + np.array([cx, cy, cz])
        return world[:, 0], world[:, 1], world[:, 2]

    def _add_box_trace(self, fig, cx, cy, cz, sx, sy, sz, yaw, color, name, opacity=0.95, showlegend=True):
        x, y, z = self._box_mesh(cx, cy, cz, sx, sy, sz, yaw)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=self._BOX_I, j=self._BOX_J, k=self._BOX_K,
            color=color, opacity=opacity, name=name, showlegend=showlegend,
            flatshading=True, lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
        ))

    def _export_3d_map_html(self):
        """
        Saves an interactive, rotatable 3-D map as a standalone HTML file
        that renders the pool itself (floor, lane markings, walls, water
        surface) plus every fixed-size object — gate, green mat, bins — as
        true-to-scale solids at their tracked positions, and the AUV path
        colored by mission progress.
        """
        if not PLOTLY_AVAILABLE:
            self.get_logger().warn(
                "[Map3D] plotly not installed — skipping interactive 3D map. "
                "Install with: pip install plotly"
            )
            return

        L, W = self.POOL_LENGTH_M, self.POOL_WIDTH_M
        floor_z, water_z = self.POOL_FLOOR_Z, self.WATER_SURFACE_Z
        fig = go.Figure()

        i_quad, j_quad, k_quad = [0, 0], [1, 2], [2, 3]

        # ── Pool floor + lane markings (real pools have lane lines running
        # parallel to the swim direction; the AUV travels along Y here) ───
        fig.add_trace(go.Mesh3d(
            x=[0, L, L, 0], y=[0, 0, W, W], z=[floor_z] * 4,
            i=i_quad, j=j_quad, k=k_quad,
            opacity=1.0, color='#b0bec5', name='Pool Floor'
        ))
        for lane_x in np.linspace(0, L, 6)[1:-1]:
            fig.add_trace(go.Scatter3d(
                x=[lane_x, lane_x], y=[0, W], z=[floor_z + 0.01, floor_z + 0.01],
                mode='lines', line=dict(color='white', width=3),
                showlegend=False, hoverinfo='skip'
            ))

        # Water surface — translucent cap over the whole pool.
        fig.add_trace(go.Mesh3d(
            x=[0, L, L, 0], y=[0, 0, W, W], z=[water_z] * 4,
            i=i_quad, j=j_quad, k=k_quad,
            opacity=0.15, color='#0288d1', name='Water Surface'
        ))

        # Pool walls (the two Y-spanning walls at X=0/X=L are the "25m"
        # walls the gate/green mat are built perpendicular to).
        wall_color = '#4fc3f7'
        fig.add_trace(go.Mesh3d(x=[0, L, L, 0], y=[0, 0, 0, 0], z=[floor_z, floor_z, water_z, water_z], i=i_quad, j=j_quad, k=k_quad, opacity=0.25, color=wall_color, name='Pool Walls'))
        fig.add_trace(go.Mesh3d(x=[0, L, L, 0], y=[W, W, W, W], z=[floor_z, floor_z, water_z, water_z], i=i_quad, j=j_quad, k=k_quad, opacity=0.25, color=wall_color, showlegend=False))
        fig.add_trace(go.Mesh3d(x=[0, 0, 0, 0], y=[0, W, W, 0], z=[floor_z, floor_z, water_z, water_z], i=i_quad, j=j_quad, k=k_quad, opacity=0.25, color=wall_color, showlegend=False))
        fig.add_trace(go.Mesh3d(x=[L, L, L, L], y=[0, W, W, 0], z=[floor_z, floor_z, water_z, water_z], i=i_quad, j=j_quad, k=k_quad, opacity=0.25, color=wall_color, showlegend=False))

        # ── AUV trail, colored by mission progress ─────────────────────
        if self._trail:
            xs = [p[0] for p in self._trail]
            ys = [p[1] for p in self._trail]
            zs = [p[2] for p in self._trail]
            progress = np.linspace(0, 1, len(xs))

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(color=progress, colorscale='Plasma', width=7,
                          colorbar=dict(title='Mission<br>progress', x=1.02)),
                name='AUV trail',
                hovertemplate='x=%{x:.2f}m<br>y=%{y:.2f}m<br>z=%{z:.2f}m<extra></extra>',
            ))
            fig.add_trace(go.Scatter3d(
                x=[xs[0]], y=[ys[0]], z=[zs[0]],
                mode='markers', marker=dict(color='#00c853', size=7, symbol='circle',
                                             line=dict(color='black', width=1)),
                name='Start'
            ))
            fig.add_trace(go.Scatter3d(
                x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
                mode='markers', marker=dict(color='#d50000', size=7, symbol='square',
                                             line=dict(color='black', width=1)),
                name='End'
            ))

        # ── Gate — fixed-size solid posts + crossbar, oriented perpendicular
        # to the 25m walls (known constant yaw, not measured per-frame) ────
        if self._gate_track is not None:
            gx, gy, gz, gyaw, _stamp, _update_count = self._gate_track
            width, height, thick = self.GATE_WIDTH_M, self.GATE_HEIGHT_M, self.GATE_THICKNESS_M
            offset = np.array([np.cos(gyaw), np.sin(gyaw)]) * (width / 2.0)
            top_z = gz + height / 2.0

            for sign, label in ((-1, 'Gate Post L'), (1, 'Gate Post R')):
                px, py = gx + sign * offset[0], gy + sign * offset[1]
                self._add_box_trace(fig, px, py, gz, thick, thick, height,
                                     0.0, 'black', label, showlegend=(sign == -1))
            self._add_box_trace(fig, gx, gy, top_z, width + thick, thick, thick,
                                 gyaw, 'black', 'Gate Crossbar', showlegend=False)
            # Faint red plane showing the swim-through opening.
            self._add_box_trace(fig, gx, gy, gz, width, thick, height,
                                 gyaw, '#ff5252', 'Gate Opening', opacity=0.18)

        # ── Bins — true-to-scale cylinders ──────────────────────────────
        theta = np.linspace(0, 2 * np.pi, 24)
        for i, (bx, by, bz, _stamp, _count) in enumerate(self._bin_markers):
            z_vals = np.array([bz - self.BIN_HEIGHT_M / 2.0, bz + self.BIN_HEIGHT_M / 2.0])
            X, Y, Z = [], [], []
            for zv in z_vals:
                for th in theta:
                    X.append(bx + self.BIN_RADIUS_M * np.cos(th))
                    Y.append(by + self.BIN_RADIUS_M * np.sin(th))
                    Z.append(zv)
            fig.add_trace(go.Mesh3d(
                x=X, y=Y, z=Z, alphahull=0, color='#1565c0', opacity=0.95,
                name=f'Blue Bin {i + 1}'
            ))

        # ── Green mat — fixed-size solid slab, long axis along X (already
        # perpendicular to the 25m walls, per the known pool geometry).
        # Only the single best-confirmed track is drawn — see _best_track().
        best_green = self._best_track(self._green_markers)
        if best_green is not None:
            gx, gy, gz, _stamp, _count = best_green
            self._add_box_trace(fig, gx, gy, gz, self.GREEN_LENGTH_M, self.GREEN_WIDTH_M, 0.05,
                                 0.0, '#2e7d32', 'Green Mat')

        fig.update_layout(
            title='AUV Mission Map — 3D Pool Reconstruction',
            scene=dict(
                xaxis=dict(title='X (m)', range=[-2, L + 2]),
                yaxis=dict(title='Y (m)', range=[-2, W + 2]),
                zaxis=dict(title='Z (m)', range=[floor_z - 0.5, 1.5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=W / L, z=(abs(floor_z) + 1.5) / L),
                camera=dict(eye=dict(x=1.35, y=-1.55, z=0.9)),
                bgcolor='#eceff1',
            ),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='white',
        )

        # Same atomic-write reasoning as _export_2d_map: this file is several
        # MB (embeds the full plotly.js bundle) and takes noticeably longer
        # to write than the PNG, widening the window for a forceful kill
        # during shutdown to leave a truncated HTML file.
        tmp_path = self._map_html_path + '.tmp'
        fig.write_html(tmp_path)
        os.replace(tmp_path, self._map_html_path)
        self.get_logger().info(f'[Map3D] Interactive 3D map saved -> {self._map_html_path}')

    # ── Periodic export (survives an unclean shutdown) ──────────────────

    def _periodic_map_export(self):
        if not self._trail:
            return
        self._export_2d_map()
        self._export_3d_map_html()

    # ── Shutdown ──────────────────────────────────────────────────────────

    def destroy_node(self):
        if not self._csv_file.closed:
            self._csv_file.close()
        if not self._markers_csv_file.closed:
            self._markers_csv_file.close()

        self._export_2d_map()
        self._export_3d_map_html()

        self.get_logger().info(
            f'Trail Mapper shut down. '
            f'{len(self._trail)} poses | '
            f'{1 if self._gate_track is not None else 0} gate | '
            f'{len(self._bin_markers)} bins | '
            f'total dist={self._total_dist:.2f}m | '
            f'trail csv={self._csv_path} | markers csv={self._markers_csv_path} | '
            f'map png={self._map_png_path} | map 3d html={self._map_html_path}'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrailMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
