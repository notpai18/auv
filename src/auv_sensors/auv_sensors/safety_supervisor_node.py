#!/usr/bin/env python3
"""
safety_supervisor_node.py — Real-Time AUV Sensor Safety Supervisor & Failsafe Node.

In real Autonomous Underwater Vehicles (AUVs), vehicle survivability depends on
failsafe supervisory logic monitoring environmental, structural, and electrical sensors.
Without immediate autonomous intervention:
  - Hull leaks cause catastrophic salt-water electronics shorting within seconds.
  - Undetected battery depletion strands the AUV submerged at the bottom.
  - Excessive tilt / rollover prevents camera tracking and burns out thrusters.
  - Depth exceedance can exceed enclosure pressure rating (implosion).

Monitored Sensor Streams:
  1. /auv/leak (std_msgs/Bool): Hull water ingress sensors.
  2. /auv/battery (sensor_msgs/BatteryState): Voltage, current, state of charge (SoC).
  3. /auv/imu (sensor_msgs/Imu): Gyroscope & Accelerometer tilt monitoring.
  4. /auv/depth (std_msgs/Float32): Hydrostatic depth ceiling/floor limits.

Emergency Actions:
  - Publishes /auv/emergency_stop (std_msgs/Bool).
  - Publishes /auv/safety_status (std_msgs/String).
  - Can command emergency surface ascent on /model/auv_box/cmd_vel (stopping horizontal thrust
    and driving maximum safe positive heave thrust).
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, String, Float32
from sensor_msgs.msg import BatteryState, Imu
from geometry_msgs.msg import Twist


def quat_to_tilt_angles(x, y, z, w):
    """Compute roll and pitch tilt angles (radians) from quaternion."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    return roll, pitch


class SafetySupervisorNode(Node):
    def __init__(self):
        super().__init__('safety_supervisor_node')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('leak_topic', '/auv/leak')
        self.declare_parameter('battery_topic', '/auv/battery')
        self.declare_parameter('imu_topic', '/auv/imu')
        self.declare_parameter('depth_topic', '/auv/depth')
        self.declare_parameter('cmd_vel_topic', '/model/auv_box/cmd_vel')
        self.declare_parameter('monitor_rate_hz', 10.0)

        # Safety thresholds
        self.declare_parameter('min_battery_voltage', 13.6)       # 4S LiPo cutoff (3.4V / cell)
        self.declare_parameter('min_battery_percentage', 0.15)    # 15% reserve for emergency surfacing
        self.declare_parameter('max_tilt_deg', 50.0)              # Tumbling limit
        self.declare_parameter('max_depth_m', 8.0)                # Maximum safe operational depth
        self.declare_parameter('emergency_surface_vel', 0.3)      # m/s upward thrust during emergency
        self.declare_parameter('auto_override_cmd_vel', False)    # True = publish emergency ascent to cmd_vel

        p = self.get_parameter
        leak_topic = str(p('leak_topic').value)
        battery_topic = str(p('battery_topic').value)
        imu_topic = str(p('imu_topic').value)
        depth_topic = str(p('depth_topic').value)
        rate_hz = float(p('monitor_rate_hz').value)

        # ── Publishers ──────────────────────────────────────────────────
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._estop_pub = self.create_publisher(Bool, '/auv/emergency_stop', pub_qos)
        self._status_pub = self.create_publisher(String, '/auv/safety_status', pub_qos)
        self._cmd_pub = self.create_publisher(Twist, p('cmd_vel_topic').value, 10)

        # ── Subscribers ─────────────────────────────────────────────────
        self.create_subscription(Bool, leak_topic, self._on_leak, qos_profile_sensor_data)
        self.create_subscription(BatteryState, battery_topic, self._on_battery, qos_profile_sensor_data)
        self.create_subscription(Imu, imu_topic, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Float32, depth_topic, self._on_depth, qos_profile_sensor_data)

        # ── Internal State ──────────────────────────────────────────────
        self._leak_detected = False
        self._battery_voltage = 16.8
        self._battery_percentage = 1.0
        self._current_depth = 0.0
        self._current_tilt_deg = 0.0

        self._emergency_active = False
        self._alarm_reason = 'NOMINAL'

        timer_period = 1.0 / max(1.0, rate_hz)
        self._timer = self.create_timer(timer_period, self._monitor_loop)

        self.get_logger().info(
            f'SafetySupervisorNode active: monitoring Leak, Battery (<{p("min_battery_voltage").value}V), '
            f'Tilt (>{p("max_tilt_deg").value}°), Depth (>{p("max_depth_m").value}m) @ {rate_hz:.0f} Hz'
        )

    def _on_leak(self, msg: Bool):
        if msg.data and not self._leak_detected:
            self.get_logger().error('🚨 CRITICAL SAFETY ALARM: WATER INGRESS DETECTED IN HULL! 🚨')
            self._emergency_active = True
            estop_msg = Bool()
            estop_msg.data = True
            self._estop_pub.publish(estop_msg)
        self._leak_detected = msg.data

    def _on_battery(self, msg: BatteryState):
        self._battery_voltage = float(msg.voltage)
        self._battery_percentage = float(msg.percentage)

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        if abs(q.w) > 1e-4 or abs(q.x) > 1e-4 or abs(q.y) > 1e-4 or abs(q.z) > 1e-4:
            roll, pitch = quat_to_tilt_angles(q.x, q.y, q.z, q.w)
            self._current_tilt_deg = math.degrees(math.sqrt(roll * roll + pitch * pitch))

    def _on_depth(self, msg: Float32):
        self._current_depth = float(msg.data)

    def _monitor_loop(self):
        min_v = float(self.get_parameter('min_battery_voltage').value)
        min_pct = float(self.get_parameter('min_battery_percentage').value)
        max_tilt = float(self.get_parameter('max_tilt_deg').value)
        max_depth = float(self.get_parameter('max_depth_m').value)

        # 1. Evaluate safety constraints
        emergency = False
        reasons = []

        if self._leak_detected:
            emergency = True
            reasons.append('LEAK_INGRESS')

        if self._battery_voltage < min_v or self._battery_percentage < min_pct:
            emergency = True
            reasons.append(f'LOW_BATTERY({self._battery_voltage:.1f}V/{self._battery_percentage*100:.0f}%)')

        if self._current_tilt_deg > max_tilt:
            emergency = True
            reasons.append(f'EXCESSIVE_TILT({self._current_tilt_deg:.1f}deg)')

        if self._current_depth > max_depth:
            emergency = True
            reasons.append(f'MAX_DEPTH_EXCEEDED({self._current_depth:.1f}m)')

        self._emergency_active = emergency
        status_text = 'NOMINAL' if not emergency else f'EMERGENCY: {" | ".join(reasons)}'
        self._alarm_reason = status_text

        # 2. Publish safety topics
        estop_msg = Bool()
        estop_msg.data = bool(emergency)
        self._estop_pub.publish(estop_msg)

        status_msg = String()
        status_msg.data = f'{status_text} | DEPTH={self._current_depth:.2f}m | BATT={self._battery_voltage:.2f}V | TILT={self._current_tilt_deg:.1f}°'
        self._status_pub.publish(status_msg)

        # 3. Autonomous emergency ascent override
        if emergency and bool(self.get_parameter('auto_override_cmd_vel').value):
            ascent_cmd = Twist()
            # In ROS Z is positive upwards: command positive vertical velocity to reach surface
            ascent_cmd.linear.z = float(self.get_parameter('emergency_surface_vel').value)
            ascent_cmd.linear.x = 0.0
            ascent_cmd.linear.y = 0.0
            ascent_cmd.angular.z = 0.0
            self._cmd_pub.publish(ascent_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SafetySupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
