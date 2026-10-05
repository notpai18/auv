#!/usr/bin/env python3
"""
velocity_controller_node.py — Closed-Loop 6-DOF Body Wrench Controller.

Consumes:
  - Commanded body velocities (geometry_msgs/Twist) on /model/auv_box/cmd_vel
  - Vehicle state & odometry (nav_msgs/Odometry) on /auv/odom
  - Optional target depth (std_msgs/Float32) on /auv/set_target_depth

Produces:
  - Desired 6-DOF net body wrench (geometry_msgs/Wrench) on /auv/wrench_cmd

Features:
  - Closed-loop PID tracking on surge (vx), sway (vy), and yaw rate (wz)
  - Active depth holding via heave (Fz) PID when vertical command is zero
  - Hydrodynamic drag feedforward compensation
  - Metacentric roll & pitch active damping to maintain level posture
  - Safety tipping damping if vehicle tilts excessively
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Wrench
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


def quat_to_euler(x, y, z, w):
    """Convert quaternion (x, y, z, w) to roll, pitch, yaw (radians)."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class VelocityControllerNode(Node):

    def __init__(self):
        super().__init__('velocity_controller_node')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('cmd_vel_topic',    '/model/auv_box/cmd_vel')
        self.declare_parameter('odom_topic',       '/auv/odom')
        self.declare_parameter('wrench_topic',     '/auv/wrench_cmd')
        self.declare_parameter('control_rate_hz',  50.0)
        self.declare_parameter('cmd_timeout_sec',  0.5)

        # Target depth (NaN = latch initial depth from odom)
        self.declare_parameter('target_depth',     float('nan'))

        # Surge PID gains
        self.declare_parameter('kp_surge', 30.0)
        self.declare_parameter('ki_surge', 2.0)
        self.declare_parameter('kd_surge', 2.0)

        # Sway PID gains
        self.declare_parameter('kp_sway',  30.0)
        self.declare_parameter('ki_sway',  2.0)
        self.declare_parameter('kd_sway',  2.0)

        # Heave PID gains (for depth holding)
        self.declare_parameter('kp_depth', 35.0)
        self.declare_parameter('ki_depth', 2.0)
        self.declare_parameter('kd_depth', 15.0)

        # Yaw rate PID gains
        self.declare_parameter('kp_yaw',   10.0)
        self.declare_parameter('ki_yaw',   0.5)
        self.declare_parameter('kd_yaw',   0.8)

        # Posture stabilization (roll & pitch damping)
        self.declare_parameter('kp_roll',  15.0)
        self.declare_parameter('kd_roll',  3.0)
        self.declare_parameter('kp_pitch', 15.0)
        self.declare_parameter('kd_pitch', 3.0)

        # Maximum forces & moments
        self.declare_parameter('max_force_x',  80.0)  # N
        self.declare_parameter('max_force_y',  80.0)  # N
        self.declare_parameter('max_force_z',  60.0)  # N
        self.declare_parameter('max_torque_z', 25.0)  # N*m
        self.declare_parameter('max_torque_xy',20.0)  # N*m

        p = self.get_parameter
        self._cmd_timeout = float(p('cmd_timeout_sec').value)
        self._target_depth = float(p('target_depth').value)

        self._kp_surge = float(p('kp_surge').value)
        self._ki_surge = float(p('ki_surge').value)
        self._kd_surge = float(p('kd_surge').value)

        self._kp_sway = float(p('kp_sway').value)
        self._ki_sway = float(p('ki_sway').value)
        self._kd_sway = float(p('kd_sway').value)

        self._kp_depth = float(p('kp_depth').value)
        self._ki_depth = float(p('ki_depth').value)
        self._kd_depth = float(p('kd_depth').value)

        self._kp_yaw = float(p('kp_yaw').value)
        self._ki_yaw = float(p('ki_yaw').value)
        self._kd_yaw = float(p('kd_yaw').value)

        self._kp_roll = float(p('kp_roll').value)
        self._kd_roll = float(p('kd_roll').value)
        self._kp_pitch = float(p('kp_pitch').value)
        self._kd_pitch = float(p('kd_pitch').value)

        self._max_fx = float(p('max_force_x').value)
        self._max_fy = float(p('max_force_y').value)
        self._max_fz = float(p('max_force_z').value)
        self._max_tz = float(p('max_torque_z').value)
        self._max_txy = float(p('max_torque_xy').value)

        # State storage
        self._cmd_vel = Twist()
        self._last_cmd_time = 0.0

        self._current_pos = [0.0, 0.0, 0.0]
        self._current_rpy = [0.0, 0.0, 0.0]
        self._body_lin_vel = [0.0, 0.0, 0.0]
        self._body_ang_vel = [0.0, 0.0, 0.0]
        self._has_odom = False

        # Integrators
        self._int_surge = 0.0
        self._int_sway = 0.0
        self._int_depth = 0.0
        self._int_yaw = 0.0

        # Previous errors for derivative
        self._prev_err_surge = 0.0
        self._prev_err_sway = 0.0
        self._prev_err_depth = 0.0
        self._prev_err_yaw = 0.0
        self._last_tick_time = None

        # ── ROS Interfaces ──────────────────────────────────────────────
        self._cmd_sub = self.create_subscription(
            Twist, p('cmd_vel_topic').value, self._cmd_cb, 10)
        self._odom_sub = self.create_subscription(
            Odometry, p('odom_topic').value, self._odom_cb, 10)
        self._depth_set_sub = self.create_subscription(
            Float32, '/auv/set_target_depth', self._set_depth_cb, 10)

        self._wrench_pub = self.create_publisher(
            Wrench, p('wrench_topic').value, 10)

        period = 1.0 / max(float(p('control_rate_hz').value), 1.0)
        self._timer = self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f'Velocity Controller Node running @ {p("control_rate_hz").value:.0f} Hz -> {p("wrench_topic").value}'
        )

    def _cmd_cb(self, msg: Twist):
        self._cmd_vel = msg
        self._last_cmd_time = time.monotonic()

    def _set_depth_cb(self, msg: Float32):
        self._target_depth = float(msg.data)
        self._int_depth = 0.0
        self.get_logger().info(f'[VelocityController] Target depth updated: {self._target_depth:.3f} m')

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._current_pos = [p.x, p.y, p.z]

        # Latch initial depth if NaN
        if math.isnan(self._target_depth):
            self._target_depth = p.z
            self.get_logger().info(f'[VelocityController] Latched initial depth: {self._target_depth:.3f} m')

        q = msg.pose.pose.orientation
        r, pt, y = quat_to_euler(q.x, q.y, q.z, q.w)
        self._current_rpy = [r, pt, y]

        # Odometry twist is in child_frame (base_link) or map frame.
        # Check frame_id: Gazebo OdometryPublisher publishes linear in child_frame or map.
        # We transform world linear velocity into body frame to be 100% robust.
        vx_raw = msg.twist.twist.linear.x
        vy_raw = msg.twist.twist.linear.y
        vz_raw = msg.twist.twist.linear.z

        # If odom frame is map and child_frame_id is base_link, twist is often body frame.
        # If in map frame, rotate by -yaw:
        # Since OdometryPublisher in Gazebo uses robot_base_frame=base_link,
        # msg.twist.twist is already expressed in the body frame.
        self._body_lin_vel = [vx_raw, vy_raw, vz_raw]
        self._body_ang_vel = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ]
        self._has_odom = True

    def _control_loop(self):
        if not self._has_odom:
            return

        now = time.monotonic()
        if self._last_tick_time is None:
            self._last_tick_time = now
            return
        dt = now - self._last_tick_time
        self._last_tick_time = now
        if dt <= 0.0 or dt > 0.5:
            return

        # Check command freshness
        if (now - self._last_cmd_time) < self._cmd_timeout:
            cmd_vx = self._cmd_vel.linear.x
            cmd_vy = self._cmd_vel.linear.y
            cmd_vz = self._cmd_vel.linear.z
            cmd_wz = self._cmd_vel.angular.z
        else:
            cmd_vx = 0.0
            cmd_vy = 0.0
            cmd_vz = 0.0
            cmd_wz = 0.0

        # Current body states
        vx, vy, vz = self._body_lin_vel
        roll, pitch, yaw = self._current_rpy
        wx, wy, wz = self._body_ang_vel
        cz = self._current_pos[2]

        # ── 1. Surge Control (Fx) ───────────────────────────────────────
        err_surge = cmd_vx - vx
        self._int_surge = max(-20.0, min(20.0, self._int_surge + err_surge * dt))
        der_surge = (err_surge - self._prev_err_surge) / dt
        self._prev_err_surge = err_surge

        # Feedforward drag compensation (BlueROV2: X_u|u| = 33.7)
        ff_surge = 33.732 * cmd_vx * abs(cmd_vx)
        fx = self._kp_surge * err_surge + self._ki_surge * self._int_surge + self._kd_surge * der_surge + ff_surge
        fx = max(-self._max_fx, min(self._max_fx, fx))

        # ── 2. Sway Control (Fy) ────────────────────────────────────────
        err_sway = cmd_vy - vy
        self._int_sway = max(-20.0, min(20.0, self._int_sway + err_sway * dt))
        der_sway = (err_sway - self._prev_err_sway) / dt
        self._prev_err_sway = err_sway

        ff_sway = 54.16 * cmd_vy * abs(cmd_vy)
        fy = self._kp_sway * err_sway + self._ki_sway * self._int_sway + self._kd_sway * der_sway + ff_sway
        fy = max(-self._max_fy, min(self._max_fy, fy))

        # ── 3. Heave / Depth Control (Fz) ───────────────────────────────
        if abs(cmd_vz) > 0.01:
            # Commanded vertical velocity
            err_heave = cmd_vz - vz
            fz = self._kp_surge * err_heave
            # Update target depth dynamically while moving vertically
            self._target_depth = cz
            self._int_depth = 0.0
        else:
            # Active depth hold
            err_depth = self._target_depth - cz
            self._int_depth = max(-15.0, min(15.0, self._int_depth + err_depth * dt))
            fz = self._kp_depth * err_depth - self._kd_depth * vz + self._ki_depth * self._int_depth

        fz = max(-self._max_fz, min(self._max_fz, fz))

        # ── 4. Roll & Pitch Attitude Stabilization (Mx, My) ─────────────
        # Active restoration torque to maintain zero roll and pitch
        mx = -self._kp_roll * roll - self._kd_roll * wx
        my = -self._kp_pitch * pitch - self._kd_pitch * wy

        # Safety tipping prevention: if tilt > 35 deg, damp horizontal thrust
        tilt = math.sqrt(roll * roll + pitch * pitch)
        if tilt > math.radians(35.0):
            fx *= 0.2
            fy *= 0.2
            mx *= 1.5
            my *= 1.5

        mx = max(-self._max_txy, min(self._max_txy, mx))
        my = max(-self._max_txy, min(self._max_txy, my))

        # ── 5. Yaw Control (Mz) ─────────────────────────────────────────
        err_yaw = cmd_wz - wz
        self._int_yaw = max(-10.0, min(10.0, self._int_yaw + err_yaw * dt))
        der_yaw = (err_yaw - self._prev_err_yaw) / dt
        self._prev_err_yaw = err_yaw

        ff_yaw = 3.992 * cmd_wz * abs(cmd_wz)
        mz = self._kp_yaw * err_yaw + self._ki_yaw * self._int_yaw + self._kd_yaw * der_yaw + ff_yaw
        mz = max(-self._max_tz, min(self._max_tz, mz))

        # ── Publish Wrench ──────────────────────────────────────────────
        wrench = Wrench()
        wrench.force.x = float(fx)
        wrench.force.y = float(fy)
        wrench.force.z = float(fz)
        wrench.torque.x = float(mx)
        wrench.torque.y = float(my)
        wrench.torque.z = float(mz)

        self._wrench_pub.publish(wrench)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
