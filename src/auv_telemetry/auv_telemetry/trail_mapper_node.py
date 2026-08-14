#!/usr/bin/env python3
"""
trail_mapper_node.py  —  AUV Trail Mapper

Subscribes to /auv/odom and builds a 3-D trajectory trail.
Publishes the trail as nav_msgs/Path on /auv/trail_map (1 Hz).
Saves the full trail to a CSV in ~/auv_ws/mission_logs/ on shutdown.

Downsampling:  A new pose is recorded only when the AUV has moved at
               least MIN_DIST_M metres from the last stored pose.
               This prevents flooding the path with nearly-identical points
               while the robot is stationary or turning in place.

ROS topics
----------
Sub: /auv/odom          nav_msgs/Odometry   — AUV pose in map frame
Pub: /auv/trail_map     nav_msgs/Path       — full recorded trail

Usage
-----
ros2 run auv_telemetry trail_mapper_node
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import math
import os
import csv
from datetime import datetime


class TrailMapperNode(Node):

    # ── Tunable parameters ────────────────────────────────────────────────
    # Minimum distance (m) the AUV must travel before a new point is stored.
    # Lower = more detail, higher memory use. 0.05–0.20 is a good range.
    MIN_DIST_M = 0.10

    # How often (Hz) to publish the accumulated path.
    PUBLISH_HZ = 1.0
    # ─────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('trail_mapper_node')

        # Internal trail storage: list of (x, y, z, yaw_deg, stamp)
        self._trail: list[tuple] = []

        # Last stored position — used for distance downsampling
        self._last_x: float | None = None
        self._last_y: float | None = None
        self._last_z: float | None = None

        # Total distance accumulated over the mission
        self._total_dist: float = 0.0

        # ── CSV file ─────────────────────────────────────────────────────
        log_dir = os.path.join(os.path.expanduser('~'), 'auv_ws', 'mission_logs')
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._csv_path = os.path.join(log_dir, f'trail_{ts}.csv')
        self._csv_file = open(self._csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ['index', 'x_m', 'y_m', 'z_m', 'yaw_deg', 'dist_from_prev_m', 'total_dist_m']
        )

        # ── ROS interfaces ────────────────────────────────────────────────
        self._odom_sub = self.create_subscription(
            Odometry, '/auv/odom', self._odom_callback, 10
        )
        self._path_pub = self.create_publisher(Path, '/auv/trail_map', 10)

        self.create_timer(1.0 / self.PUBLISH_HZ, self._publish_path)

        self.get_logger().info(
            f'Trail Mapper started | MIN_DIST={self.MIN_DIST_M}m | '
            f'publish={self.PUBLISH_HZ}Hz | csv={self._csv_path}'
        )

    # ── Odometry callback ─────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # ── Distance downsampling ─────────────────────────────────────────
        if self._last_x is not None:
            dist = math.sqrt(
                (x - self._last_x) ** 2 +
                (y - self._last_y) ** 2 +
                (z - self._last_z) ** 2
            )
            if dist < self.MIN_DIST_M:
                return          # Too close to last stored point — skip
            self._total_dist += dist
        else:
            dist = 0.0          # Very first point

        # ── Compute yaw from quaternion ───────────────────────────────────
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        yaw_deg = math.degrees(yaw)

        # ── Store the pose ────────────────────────────────────────────────
        self._trail.append((x, y, z, yaw_deg, msg.header.stamp, q))
        self._last_x, self._last_y, self._last_z = x, y, z

        idx = len(self._trail)
        self._csv_writer.writerow([
            idx,
            f'{x:.4f}', f'{y:.4f}', f'{z:.4f}',
            f'{yaw_deg:.2f}',
            f'{dist:.4f}',
            f'{self._total_dist:.4f}',
        ])
        self._csv_file.flush()

        if idx % 20 == 0:       # Log a summary every 20 stored points
            self.get_logger().info(
                f'[Trail] {idx} poses recorded | '
                f'total_dist={self._total_dist:.2f}m | '
                f'pos=({x:.2f}, {y:.2f}, {z:.2f})'
            )

    # ── Path publisher ────────────────────────────────────────────────────

    def _publish_path(self):
        if not self._trail:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for (x, y, z, _yaw_deg, stamp, q) in self._trail:
            ps = PoseStamped()
            ps.header.stamp    = stamp
            ps.header.frame_id = 'map'
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = z
            # Store full orientation (not just position)
            ps.pose.orientation = q
            path_msg.poses.append(ps)

        self._path_pub.publish(path_msg)

    # ── Shutdown ──────────────────────────────────────────────────────────

    def destroy_node(self):
        if not self._csv_file.closed:
            self._csv_file.close()
        self.get_logger().info(
            f'Trail Mapper shut down. '
            f'{len(self._trail)} poses | '
            f'total dist={self._total_dist:.2f}m | '
            f'saved → {self._csv_path}'
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
