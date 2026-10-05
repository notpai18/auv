#!/usr/bin/env python3
"""
cmd_vel_mixer_node.py  —  Merges navigation + depth cmd_vel streams.

Architecture
------------
  Navigator nodes (gate, green) publish HORIZONTAL motion:
    linear.x, linear.y, angular.z  →  /auv/nav_cmd_vel

  depth_hold_node publishes VERTICAL motion:
    linear.z                        →  /auv/depth_cmd_vel

  This node merges them into one Twist on /model/auv_box/cmd_vel
  which the Gazebo VelocityControl plugin consumes.

This separation means:
  - Horizontal navigation and depth control are tuned independently.
  - Matches the real hardware architecture (separate vertical thrusters).
  - Neither navigator needs to know about depth at all.

Safety timeouts
---------------
  If nav_cmd_vel is stale (> nav_timeout_sec), X/Y/yaw are zeroed.
  If depth_cmd_vel is stale (> depth_timeout_sec), Z is zeroed.
  Both are ROS 2 parameters.
"""

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelMixerNode(Node):

    def __init__(self):
        super().__init__('cmd_vel_mixer_node')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('nav_topic',          '/auv/nav_cmd_vel')
        self.declare_parameter('depth_topic',        '/auv/depth_cmd_vel')
        self.declare_parameter('output_topic',       '/model/auv_box/cmd_vel')
        self.declare_parameter('publish_hz',         20.0)  # merged output rate
        self.declare_parameter('nav_timeout_sec',    0.5)   # zero XY if stale
        self.declare_parameter('depth_timeout_sec',  1.0)   # zero Z if stale
        # ────────────────────────────────────────────────────────────────

        p = self.get_parameter
        self._nav_timeout   = p('nav_timeout_sec').value
        self._depth_timeout = p('depth_timeout_sec').value
        nav_topic    = p('nav_topic').value
        depth_topic  = p('depth_topic').value
        output_topic = p('output_topic').value

        self._nav_cmd:   Twist = Twist()
        self._depth_cmd: Twist = Twist()
        self._nav_last_t:   float = 0.0
        self._depth_last_t: float = 0.0

        # ── ROS interfaces ───────────────────────────────────────────────
        self._nav_sub = self.create_subscription(
            Twist, nav_topic, self._nav_cb, 10)
        self._depth_sub = self.create_subscription(
            Twist, depth_topic, self._depth_cb, 10)
        self._pub = self.create_publisher(Twist, output_topic, 10)

        period = 1.0 / max(p('publish_hz').value, 1.0)
        self.create_timer(period, self._publish)
        # ────────────────────────────────────────────────────────────────

        self.get_logger().info(
            f'CmdVel Mixer started: '
            f'({nav_topic} | {depth_topic}) → {output_topic} @ {p("publish_hz").value:.0f} Hz'
        )

    def _nav_cb(self, msg: Twist):
        self._nav_cmd    = msg
        self._nav_last_t = time.monotonic()

    def _depth_cb(self, msg: Twist):
        self._depth_cmd    = msg
        self._depth_last_t = time.monotonic()

    def _publish(self):
        now    = time.monotonic()
        merged = Twist()

        # Horizontal motion (X, Y, yaw) — zero if navigator is silent/stale
        if (now - self._nav_last_t) < self._nav_timeout:
            merged.linear.x  = self._nav_cmd.linear.x
            merged.linear.y  = self._nav_cmd.linear.y
            merged.angular.z = self._nav_cmd.angular.z

        # Vertical motion (Z) — zero if depth controller is silent/stale
        if (now - self._depth_last_t) < self._depth_timeout:
            merged.linear.z = self._depth_cmd.linear.z

        self._pub.publish(merged)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMixerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
