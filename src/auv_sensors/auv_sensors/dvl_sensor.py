#!/usr/bin/env python3
"""
dvl_sensor.py — Doppler Velocity Log (DVL) & Acoustic Altimeter Node.

In a real Autonomous Underwater Vehicle (AUV), GPS radio signals cannot penetrate
conductive seawater. A DVL (e.g. Water Linked DVL A50, Teledyne RDI Pathfinder) is
the core sensor for underwater dead-reckoning navigation:

1. Theory of Operation:
   - Emits 4 acoustic ultrasound beams (typically 300 kHz - 1 MHz) in a "Janus"
     configuration tilted at ~22.5° or 30° toward the seabed.
   - Receives Doppler-shifted acoustic reflections from the bottom (bottom-track)
     or water layers (water-track).
   - Solves for 3D body-frame linear velocity [vx, vy, vz]^T with millimeter precision.
   - Measures two-way acoustic travel time to determine altitude above seabed.

2. Topics Published:
   - /auv/dvl/velocity (geometry_msgs/msg/TwistWithCovarianceStamped): body-frame linear velocities
   - /auv/dvl/altitude (sensor_msgs/msg/Range & std_msgs/msg/Float32): height above seabed
   - /auv/dvl/odom (nav_msgs/msg/Odometry): DVL dead-reckoned odometry
   - /auv/dvl/status (std_msgs/msg/String): bottom lock status and beam health

3. Subscribes:
   - odom_topic (nav_msgs/msg/Odometry, default: /auv/odom) for ground-truth velocity derivation in sim
   - /auv/imu (sensor_msgs/msg/Imu) for heading projection during dead-reckoning
"""

import math
import random
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistWithCovarianceStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range, Imu
from std_msgs.msg import Float32, String


class DvlSensorNode(Node):
    def __init__(self):
        super().__init__('dvl_sensor')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('odom_topic', '/auv/odom')
        self.declare_parameter('imu_topic', '/auv/imu')
        self.declare_parameter('publish_rate_hz', 20.0)           # Real DVLs typically tick at 10-25 Hz
        self.declare_parameter('velocity_noise_std', 0.005)       # 5 mm/s standard deviation (typical for DVL A50)
        self.declare_parameter('altitude_noise_std', 0.01)        # 1 cm standard deviation
        self.declare_parameter('pool_floor_z', -5.0)              # World z position of pool floor (m)
        self.declare_parameter('max_range_m', 50.0)               # Acoustic bottom lock range limit
        self.declare_parameter('min_range_m', 0.1)                # Minimum acoustic blanking distance
        self.declare_parameter('beam_angle_deg', 25.0)            # Janus beam tilt angle
        self.declare_parameter('frame_id', 'dvl_link')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('odom_frame_id', 'odom')

        odom_topic = str(self.get_parameter('odom_topic').value)
        imu_topic = str(self.get_parameter('imu_topic').value)
        rate_hz = float(self.get_parameter('publish_rate_hz').value)

        # ── Publishers ──────────────────────────────────────────────────
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._vel_pub = self.create_publisher(TwistWithCovarianceStamped, '/auv/dvl/velocity', pub_qos)
        self._alt_range_pub = self.create_publisher(Range, '/auv/dvl/altitude_range', pub_qos)
        self._alt_float_pub = self.create_publisher(Float32, '/auv/dvl/altitude', pub_qos)
        self._dvl_odom_pub = self.create_publisher(Odometry, '/auv/dvl/odom', pub_qos)
        self._status_pub = self.create_publisher(String, '/auv/dvl/status', pub_qos)

        # ── Subscribers ─────────────────────────────────────────────────
        self.create_subscription(Odometry, odom_topic, self._on_odom, qos_profile_sensor_data)
        self.create_subscription(Imu, imu_topic, self._on_imu, qos_profile_sensor_data)

        # ── Internal State ──────────────────────────────────────────────
        self._last_body_vel = [0.0, 0.0, 0.0]
        self._current_pos = [0.0, 0.0, 0.0]
        self._current_yaw = 0.0
        self._has_odom = False
        self._has_imu = False

        # Dead-reckoned integration state
        self._dr_x = 0.0
        self._dr_y = 0.0
        self._dr_z = 0.0
        self._last_dr_time = None

        timer_period = 1.0 / max(1.0, rate_hz)
        self._timer = self.create_timer(timer_period, self._on_timer)

        self.get_logger().info(
            f'DvlSensorNode initialized: rate={rate_hz:.1f}Hz, noise_std={self.get_parameter("velocity_noise_std").value} m/s'
        )

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self._current_pos = [p.x, p.y, p.z]

        # Twist in Odometry is in body frame (base_link)
        self._last_body_vel = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ]
        self._has_odom = True

    def _on_imu(self, msg: Imu):
        # Extract yaw from IMU orientation quaternion
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._has_imu = True

    def _on_timer(self):
        now = self.get_clock().now()
        now_msg = now.to_msg()
        frame_id = str(self.get_parameter('frame_id').value)
        vel_noise_std = float(self.get_parameter('velocity_noise_std').value)
        alt_noise_std = float(self.get_parameter('altitude_noise_std').value)
        pool_floor_z = float(self.get_parameter('pool_floor_z').value)
        max_range = float(self.get_parameter('max_range_m').value)
        min_range = float(self.get_parameter('min_range_m').value)

        # 1. Compute Body Velocity with Doppler Acoustic Noise
        noise_x = random.gauss(0.0, vel_noise_std) if vel_noise_std > 0 else 0.0
        noise_y = random.gauss(0.0, vel_noise_std) if vel_noise_std > 0 else 0.0
        noise_z = random.gauss(0.0, vel_noise_std) if vel_noise_std > 0 else 0.0

        meas_vx = self._last_body_vel[0] + noise_x
        meas_vy = self._last_body_vel[1] + noise_y
        meas_vz = self._last_body_vel[2] + noise_z

        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = now_msg
        twist_msg.header.frame_id = frame_id
        twist_msg.twist.twist.linear.x = float(meas_vx)
        twist_msg.twist.twist.linear.y = float(meas_vy)
        twist_msg.twist.twist.linear.z = float(meas_vz)

        # Covariance diagonal: sigma^2
        var_v = vel_noise_std ** 2
        cov = [0.0] * 36
        cov[0] = var_v    # vx
        cov[7] = var_v    # vy
        cov[14] = var_v   # vz
        cov[21] = 1e6     # wx unknown by DVL
        cov[28] = 1e6     # wy unknown by DVL
        cov[35] = 1e6     # wz unknown by DVL
        twist_msg.twist.covariance = cov
        self._vel_pub.publish(twist_msg)

        # 2. Compute Acoustic Altitude to Seabed
        # current_pos[2] is negative underwater (e.g. -4.2m). Floor is at pool_floor_z (e.g. -5.0m).
        true_altitude = max(0.0, self._current_pos[2] - pool_floor_z)
        alt_noise = random.gauss(0.0, alt_noise_std) if alt_noise_std > 0 else 0.0
        measured_altitude = max(0.0, true_altitude + alt_noise)
        has_bottom_lock = (measured_altitude <= max_range and measured_altitude >= min_range)

        # Range msg
        range_msg = Range()
        range_msg.header.stamp = now_msg
        range_msg.header.frame_id = frame_id
        range_msg.radiation_type = Range.ULTRASOUND
        range_msg.field_of_view = math.radians(float(self.get_parameter('beam_angle_deg').value))
        range_msg.min_range = float(min_range)
        range_msg.max_range = float(max_range)
        range_msg.range = float(measured_altitude if has_bottom_lock else float('inf'))
        self._alt_range_pub.publish(range_msg)

        float_msg = Float32()
        float_msg.data = float(measured_altitude)
        self._alt_float_pub.publish(float_msg)

        # Status msg
        status_msg = String()
        status_msg.data = f'BOTTOM_LOCK={has_bottom_lock}|BEAMS=4/4|ALT={measured_altitude:.2f}m'
        self._status_pub.publish(status_msg)

        # 3. Dead-Reckoning Position Integration (DVL + Yaw)
        current_time = now.nanoseconds * 1e-9
        if self._last_dr_time is not None:
            dt = current_time - self._last_dr_time
            if 0.0 < dt < 0.5:
                # Rotate body velocity into world horizontal frame using current yaw
                cos_y = math.cos(self._current_yaw)
                sin_y = math.sin(self._current_yaw)
                world_vx = cos_y * meas_vx - sin_y * meas_vy
                world_vy = sin_y * meas_vx + cos_y * meas_vy
                world_vz = meas_vz

                self._dr_x += world_vx * dt
                self._dr_y += world_vy * dt
                self._dr_z += world_vz * dt

                # Publish DVL Odometry
                dr_odom = Odometry()
                dr_odom.header.stamp = now_msg
                dr_odom.header.frame_id = str(self.get_parameter('odom_frame_id').value)
                dr_odom.child_frame_id = str(self.get_parameter('base_frame_id').value)
                dr_odom.pose.pose.position.x = float(self._dr_x)
                dr_odom.pose.pose.position.y = float(self._dr_y)
                dr_odom.pose.pose.position.z = float(self._dr_z)
                # Orientation quaternion from yaw
                dr_odom.pose.pose.orientation.z = math.sin(self._current_yaw / 2.0)
                dr_odom.pose.pose.orientation.w = math.cos(self._current_yaw / 2.0)
                dr_odom.twist.twist.linear.x = float(meas_vx)
                dr_odom.twist.twist.linear.y = float(meas_vy)
                dr_odom.twist.twist.linear.z = float(meas_vz)
                self._dvl_odom_pub.publish(dr_odom)

        self._last_dr_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = DvlSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
