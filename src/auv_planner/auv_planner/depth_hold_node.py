#!/usr/bin/env python3
"""
depth_hold_node.py  —  Vertical thruster PID depth controller.

Replaces the previous "gravity=0 / fixed Z" approach with realistic
thruster-based depth holding:

  1. Reads AUV depth (Z) and vertical velocity from /auv/odom.
  2. Runs a PID loop: error = target_depth - current_z.
  3. Publishes linear.z velocity command to /auv/depth_cmd_vel.
     cmd_vel_mixer_node blends this with the horizontal nav commands.

Target depth
------------
  - If `target_depth` parameter is NaN (default), the node latches the first
    odom Z reading and holds that depth for the rest of the run.
  - Override at any time by publishing to /auv/set_target_depth (Float32).

Tuning
------
  All gains and limits are ROS 2 parameters in sim_params.yaml.
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class DepthHoldNode(Node):

    def __init__(self):
        super().__init__('depth_hold_node')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('target_depth',     float('nan'))  # NaN = latch first odom Z
        self.declare_parameter('kp',               1.5)
        self.declare_parameter('ki',               0.05)
        self.declare_parameter('kd',               0.8)
        self.declare_parameter('max_vertical_vel', 0.30)   # m/s hard cap
        self.declare_parameter('deadband_m',       0.02)   # m  stop chatter near target
        self.declare_parameter('odom_topic',       '/auv/odom')
        self.declare_parameter('output_topic',     '/auv/depth_cmd_vel')
        self.declare_parameter('setpoint_topic',   '/auv/set_target_depth')
        # ────────────────────────────────────────────────────────────────

        p = self.get_parameter
        self._target_z   = p('target_depth').value
        self._kp         = p('kp').value
        self._ki         = p('ki').value
        self._kd         = p('kd').value
        self._max_vel    = p('max_vertical_vel').value
        self._deadband   = p('deadband_m').value

        # Anti-windup: integral term can contribute at most max_vel
        self._int_limit  = self._max_vel / max(self._ki, 1e-9)
        self._integral   = 0.0
        self._prev_error = 0.0
        self._last_time  = None

        # ── ROS interfaces ───────────────────────────────────────────────
        self._odom_sub = self.create_subscription(
            Odometry, p('odom_topic').value, self._odom_cb, 10)
        self._setpoint_sub = self.create_subscription(
            Float32, p('setpoint_topic').value, self._setpoint_cb, 10)
        self._cmd_pub = self.create_publisher(
            Twist, p('output_topic').value, 10)
        # Monitoring: raw depth for telemetry / mission_monitor
        self._depth_pub = self.create_publisher(Float32, '/auv/current_depth', 10)
        # ────────────────────────────────────────────────────────────────

        target_str = (
            'latch-first-reading'
            if math.isnan(self._target_z)
            else f'{self._target_z:.3f} m'
        )
        self.get_logger().info(
            f'Depth Hold started | PID Kp={self._kp} Ki={self._ki} Kd={self._kd} | '
            f'max_vel={self._max_vel} m/s | deadband={self._deadband} m | '
            f'target={target_str}'
        )

    # ── Setpoint update ──────────────────────────────────────────────────

    def _setpoint_cb(self, msg: Float32):
        old = self._target_z
        self._target_z   = float(msg.data)
        self._integral   = 0.0
        self._prev_error = 0.0
        self.get_logger().info(
            f'[DepthHold] Target changed: {old:.3f} → {self._target_z:.3f} m'
        )

    # ── PID callback (runs at odom rate, ~50 Hz) ─────────────────────────

    def _odom_cb(self, msg: Odometry):
        current_z = msg.pose.pose.position.z

        # Publish depth for telemetry
        dm = Float32()
        dm.data = float(current_z)
        self._depth_pub.publish(dm)

        # Latch target on first valid odom message
        if math.isnan(self._target_z):
            self._target_z = current_z
            self.get_logger().info(
                f'[DepthHold] Latched initial depth: {self._target_z:.3f} m'
            )

        # ── Time delta ────────────────────────────────────────────────────
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_time is None:
            self._last_time = now
            return
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0.0 or dt > 1.0:
            return  # clock jump or first-tick anomaly

        # ── PID ──────────────────────────────────────────────────────────
        error = self._target_z - current_z

        if abs(error) < self._deadband:
            error = 0.0  # deadband: no output when essentially on-target

        self._integral += error * dt
        self._integral  = max(-self._int_limit,
                              min(self._int_limit, self._integral))

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = (self._kp * error
                  + self._ki * self._integral
                  + self._kd * derivative)
        output = max(-self._max_vel, min(self._max_vel, output))

        # ── Publish ───────────────────────────────────────────────────────
        cmd = Twist()
        cmd.linear.z = output
        self._cmd_pub.publish(cmd)

        if abs(error) > self._deadband:
            self.get_logger().debug(
                f'z={current_z:.3f} tgt={self._target_z:.3f} '
                f'err={error:+.3f} I={self._integral:.3f} out={output:+.3f} m/s'
            )


def main(args=None):
    rclpy.init(args=args)
    node = DepthHoldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
