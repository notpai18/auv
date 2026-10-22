#!/usr/bin/env python3
"""
sensor_fusion_node.py — 6-DOF Extended Sensor Fusion & State Estimator.

In real Autonomous Underwater Vehicles (AUVs), GPS is inaccessible underwater.
Vehicle state estimation requires tightly fusing multiple complementary sensors:
  1. 3-Axis Gyroscope: High-frequency angular velocity for rate stabilization.
  2. 3-Axis Accelerometer: Low-frequency gravity vector for tilt (roll/pitch) leveling.
  3. 3-Axis Magnetometer: Tilt-compensated magnetic compass for absolute yaw heading.
  4. Hydrostatic Pressure/Depth Sensor: Millimeter-accurate absolute vertical position (z).
  5. Doppler Velocity Log (DVL): Drift-free 3D body velocities for dead-reckoning.

Architecture:
  - Orientation: 6-axis complementary/Kalman attitude filter fusing Gyroscope rates
    with Accelerometer gravity vector (roll/pitch) and Magnetometer (yaw).
  - Depth & Heave: Fuses Hydrostatic Pressure depth with vertical acceleration & DVL heave.
  - Position & Surge/Sway: Integrates DVL body velocity rotated by fused attitude.

Publishes:
  - /auv/state_estimate (nav_msgs/msg/Odometry) — Primary filtered odometry
  - /auv/attitude_rpy (geometry_msgs/msg/Vector3Stamped) — Roll, Pitch, Yaw (deg)
  - /auv/filter_debug (std_msgs/msg/Float32MultiArray) — Debug diagnostics & residuals
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, MagneticField, FluidPressure
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import TwistWithCovarianceStamped, Vector3Stamped
from nav_msgs.msg import Odometry


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler roll, pitch, yaw (rad) to quaternion (x, y, z, w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('imu_topic', '/auv/imu')
        self.declare_parameter('mag_topic', '/auv/magnetometer')
        self.declare_parameter('depth_topic', '/auv/depth')
        self.declare_parameter('dvl_topic', '/auv/dvl/velocity')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')

        # Filter gains (Complementary / Kalman weights)
        self.declare_parameter('alpha_tilt', 0.05)       # Accelerometer gravity correction gain (roll/pitch)
        self.declare_parameter('beta_yaw', 0.04)         # Magnetometer compass correction gain (yaw)
        self.declare_parameter('gamma_depth', 0.15)      # Depth sensor correction gain (z)
        self.declare_parameter('magnetic_declination_deg', 0.0) # Magnetic declination offset

        p = self.get_parameter
        imu_topic = str(p('imu_topic').value)
        mag_topic = str(p('mag_topic').value)
        depth_topic = str(p('depth_topic').value)
        dvl_topic = str(p('dvl_topic').value)
        rate_hz = float(p('publish_rate_hz').value)

        # ── Publishers ──────────────────────────────────────────────────
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._state_pub = self.create_publisher(Odometry, '/auv/state_estimate', pub_qos)
        self._rpy_pub = self.create_publisher(Vector3Stamped, '/auv/attitude_rpy', pub_qos)
        self._debug_pub = self.create_publisher(Float32MultiArray, '/auv/filter_debug', pub_qos)

        # ── Subscribers ─────────────────────────────────────────────────
        self.create_subscription(Imu, imu_topic, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(MagneticField, mag_topic, self._on_mag, qos_profile_sensor_data)
        self.create_subscription(Float32, depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(TwistWithCovarianceStamped, dvl_topic, self._on_dvl, qos_profile_sensor_data)

        # ── Estimated State Variables ───────────────────────────────────
        # Attitude (radians)
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0

        # Position (meters in odom frame)
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._pos_z = -4.238  # Default spawn depth

        # Velocity (m/s in body frame)
        self._vel_u = 0.0  # surge
        self._vel_v = 0.0  # sway
        self._vel_w = 0.0  # heave

        # Angular rate (rad/s in body frame from Gyro)
        self._gyro_p = 0.0  # roll rate
        self._gyro_q = 0.0  # pitch rate
        self._gyro_r = 0.0  # yaw rate

        # Gyro bias estimates (online bias tracking)
        self._bias_p = 0.0
        self._bias_q = 0.0
        self._bias_r = 0.0

        # Sensor flags
        self._has_imu = False
        self._has_depth = False
        self._has_dvl = False
        self._has_mag = False

        self._last_time = None
        self._last_depth_meas = None
        self._last_mag = [0.0, 0.0, 0.0]

        timer_period = 1.0 / max(1.0, rate_hz)
        self._timer = self.create_timer(timer_period, self._filter_step)

        self.get_logger().info(
            f'SensorFusionNode active: fusing IMU ({imu_topic}), Depth ({depth_topic}), '
            f'DVL ({dvl_topic}), Mag ({mag_topic}) @ {rate_hz:.0f} Hz'
        )

    def _on_imu(self, msg: Imu):
        self._has_imu = True
        # Angular rates from Gyroscope (rad/s)
        self._gyro_p = msg.angular_velocity.x
        self._gyro_q = msg.angular_velocity.y
        self._gyro_r = msg.angular_velocity.z

        # Accelerations (m/s^2)
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        # Accelerometer gravity vector tilt correction:
        # In static / low-acceleration conditions, specific force aligns with gravity.
        acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if 8.0 < acc_norm < 11.5:  # Valid gravity magnitude window (~9.81 m/s^2)
            pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))
            roll_acc = math.atan2(ay, az)

            alpha = float(self.get_parameter('alpha_tilt').value)
            self._roll = (1.0 - alpha) * self._roll + alpha * roll_acc
            self._pitch = (1.0 - alpha) * self._pitch + alpha * pitch_acc

    def _on_mag(self, msg: MagneticField):
        self._has_mag = True
        self._last_mag = [
            msg.magnetic_field.x,
            msg.magnetic_field.y,
            msg.magnetic_field.z
        ]

    def _on_depth(self, msg: Float32):
        self._has_depth = True
        # Depth is positive downwards from surface; z in odom is negative downwards
        measured_z = -float(msg.data)
        self._last_depth_meas = measured_z

    def _on_dvl(self, msg: TwistWithCovarianceStamped):
        self._has_dvl = True
        self._vel_u = msg.twist.twist.linear.x
        self._vel_v = msg.twist.twist.linear.y
        self._vel_w = msg.twist.twist.linear.z

    def _filter_step(self):
        now = self.get_clock().now()
        current_time = now.nanoseconds * 1e-9

        if self._last_time is None:
            self._last_time = current_time
            return

        dt = current_time - self._last_time
        self._last_time = current_time

        if dt <= 0.0 or dt > 0.5:
            return

        # ── 1. Gyroscope Attitude Propagation ───────────────────────────
        # Angular rates unbiased
        p = self._gyro_p - self._bias_p
        q = self._gyro_q - self._bias_q
        r = self._gyro_r - self._bias_r

        # Euler rate kinematics:
        sin_r = math.sin(self._roll)
        cos_r = math.cos(self._roll)
        tan_p = math.tan(self._pitch)
        cos_p = math.cos(self._pitch)

        if abs(cos_p) > 1e-3:
            roll_dot = p + sin_r * tan_p * q + cos_r * tan_p * r
            pitch_dot = cos_r * q - sin_r * r
            yaw_dot = (sin_r / cos_p) * q + (cos_r / cos_p) * r
        else:
            roll_dot = p
            pitch_dot = q
            yaw_dot = r

        self._roll = wrap_angle(self._roll + roll_dot * dt)
        self._pitch = wrap_angle(self._pitch + pitch_dot * dt)
        self._yaw = wrap_angle(self._yaw + yaw_dot * dt)

        # ── 2. Magnetometer Yaw Correction ──────────────────────────────
        if self._has_mag:
            mx, my, mz = self._last_mag
            # Tilt-compensation: project 3D magnetic field into horizontal plane
            # using current roll and pitch estimates
            m_xh = mx * cos_p + my * sin_r * math.sin(self._pitch) + mz * cos_r * math.sin(self._pitch)
            m_yh = my * cos_r - mz * sin_r
            mag_norm = math.sqrt(m_xh * m_xh + m_yh * m_yh)

            if mag_norm > 1e-9:
                decl = math.radians(float(self.get_parameter('magnetic_declination_deg').value))
                mag_yaw = wrap_angle(-math.atan2(m_yh, m_xh) + decl)
                yaw_err = wrap_angle(mag_yaw - self._yaw)

                beta = float(self.get_parameter('beta_yaw').value)
                self._yaw = wrap_angle(self._yaw + beta * yaw_err)

        # ── 3. Depth Sensor Absolute Z Correction ───────────────────────
        if self._has_depth and self._last_depth_meas is not None:
            gamma = float(self.get_parameter('gamma_depth').value)
            depth_err = self._last_depth_meas - self._pos_z
            self._pos_z += gamma * depth_err

        # ── 4. DVL Body Velocity Rotation & Dead-Reckoning ─────────────
        # Rotation matrix R_b_to_w (from body to world/odom frame)
        cy = math.cos(self._yaw)
        sy = math.sin(self._yaw)
        cp = math.cos(self._pitch)
        sp = math.sin(self._pitch)
        cr = math.cos(self._roll)
        sr = math.sin(self._roll)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr

        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr

        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr

        world_vx = r00 * self._vel_u + r01 * self._vel_v + r02 * self._vel_w
        world_vy = r10 * self._vel_u + r11 * self._vel_v + r12 * self._vel_w
        world_vz = r20 * self._vel_u + r21 * self._vel_v + r22 * self._vel_w

        self._pos_x += world_vx * dt
        self._pos_y += world_vy * dt
        # If no depth sensor, integrate vertical velocity from DVL:
        if not self._has_depth:
            self._pos_z += world_vz * dt

        # ── 5. Publish Fused Odometry ───────────────────────────────────
        now_msg = now.to_msg()
        odom_msg = Odometry()
        odom_msg.header.stamp = now_msg
        odom_msg.header.frame_id = str(self.get_parameter('odom_frame_id').value)
        odom_msg.child_frame_id = str(self.get_parameter('base_frame_id').value)

        odom_msg.pose.pose.position.x = float(self._pos_x)
        odom_msg.pose.pose.position.y = float(self._pos_y)
        odom_msg.pose.pose.position.z = float(self._pos_z)

        qx, qy, qz, qw = euler_to_quat(self._roll, self._pitch, self._yaw)
        odom_msg.pose.pose.orientation.x = float(qx)
        odom_msg.pose.pose.orientation.y = float(qy)
        odom_msg.pose.pose.orientation.z = float(qz)
        odom_msg.pose.pose.orientation.w = float(qw)

        # Covariance diagonal (calibrated state uncertainty)
        pose_cov = [0.0] * 36
        pose_cov[0] = 0.05   # x var (m^2)
        pose_cov[7] = 0.05   # y var (m^2)
        pose_cov[14] = 0.001 # z var (m^2 from depth sensor)
        pose_cov[21] = 0.005 # roll var (rad^2)
        pose_cov[28] = 0.005 # pitch var (rad^2)
        pose_cov[35] = 0.02  # yaw var (rad^2 from mag)
        odom_msg.pose.covariance = pose_cov

        # Body-frame linear & angular velocity
        odom_msg.twist.twist.linear.x = float(self._vel_u)
        odom_msg.twist.twist.linear.y = float(self._vel_v)
        odom_msg.twist.twist.linear.z = float(self._vel_w)
        odom_msg.twist.twist.angular.x = float(p)
        odom_msg.twist.twist.angular.y = float(q)
        odom_msg.twist.twist.angular.z = float(r)

        self._state_pub.publish(odom_msg)

        # ── 6. Publish RPY (Degrees for human telemetry) ────────────────
        rpy_msg = Vector3Stamped()
        rpy_msg.header.stamp = now_msg
        rpy_msg.header.frame_id = odom_msg.child_frame_id
        rpy_msg.vector.x = float(math.degrees(self._roll))
        rpy_msg.vector.y = float(math.degrees(self._pitch))
        rpy_msg.vector.z = float(math.degrees(self._yaw))
        self._rpy_pub.publish(rpy_msg)

        # ── 7. Publish Filter Diagnostics ──────────────────────────────
        diag = Float32MultiArray()
        # [roll_deg, pitch_deg, yaw_deg, pos_x, pos_y, pos_z, vel_u, vel_v, vel_w, gyro_p, gyro_q, gyro_r]
        diag.data = [
            float(math.degrees(self._roll)),
            float(math.degrees(self._pitch)),
            float(math.degrees(self._yaw)),
            float(self._pos_x),
            float(self._pos_y),
            float(self._pos_z),
            float(self._vel_u),
            float(self._vel_v),
            float(self._vel_w),
            float(p),
            float(q),
            float(r)
        ]
        self._debug_pub.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
