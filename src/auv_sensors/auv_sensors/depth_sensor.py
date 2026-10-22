#!/usr/bin/env python3
"""Simulated Bar02 Depth and Pressure Sensor Node.

Simulates an underwater pressure transducer (e.g., BlueRobotics Bar02/Bar30):
Reads vehicle vertical position z from odometry (where z <= 0 is submerged),
computes water depth relative to the surface (z=0), adds realistic Gaussian noise,
and calculates hydrostatic absolute pressure:
    P = P_atm + rho * g * depth

Subscribes:
  - odom_topic (nav_msgs/msg/Odometry, default: /auv/odom)

Publishes:
  - /auv/depth (std_msgs/msg/Float32, metres, positive downwards)
  - /auv/depth/pressure (sensor_msgs/msg/FluidPressure, Pascals)
"""

import math
import random
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Float32


class DepthSensorNode(Node):
    def __init__(self):
        super().__init__('depth_sensor')

        # Declare parameters
        self.declare_parameter('odom_topic', '/auv/odom')
        self.declare_parameter('surface_z', 0.0)                 # World z corresponding to water surface (m)
        self.declare_parameter('noise_std', 0.01)                # Gaussian noise standard deviation (m)
        self.declare_parameter('water_density', 1025.0)          # kg/m^3 (salt water ~1025, fresh ~1000)
        self.declare_parameter('atmospheric_pressure', 101325.0) # Pa (1 atm at sea level)
        self.declare_parameter('gravity', 9.81)                  # m/s^2
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('publish_rate_hz', 20.0)          # Continuous publish rate (Hz)

        odom_topic = self.get_parameter('odom_topic').value
        rate_hz = float(self.get_parameter('publish_rate_hz').value)

        # Reliable QoS for depth and pressure publishers so all subscribers connect
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.depth_pub = self.create_publisher(Float32, '/auv/depth', pub_qos)
        self.pressure_pub = self.create_publisher(FluidPressure, '/auv/depth/pressure', pub_qos)

        # Best-effort subscription: compatible with both reliable and best-effort publishers
        self.create_subscription(Odometry, odom_topic, self.on_odom, qos_profile_sensor_data)

        self.last_z = 0.0
        self.last_header = None
        self.has_odom = False

        # Periodic timer ensures data continues publishing at steady rate
        timer_period = 1.0 / max(1.0, rate_hz)
        self.timer = self.create_timer(timer_period, self.on_timer)

        self.get_logger().info(
            f'DepthSensorNode initialized: reading z from "{odom_topic}", '
            f'publishing /auv/depth and /auv/depth/pressure at {rate_hz:.1f} Hz'
        )

    def on_odom(self, msg: Odometry):
        self.last_z = msg.pose.pose.position.z
        self.last_header = msg.header
        self.has_odom = True

    def on_timer(self):
        surface_z = float(self.get_parameter('surface_z').value)
        noise_std = float(self.get_parameter('noise_std').value)
        rho = float(self.get_parameter('water_density').value)
        p_atm = float(self.get_parameter('atmospheric_pressure').value)
        g = float(self.get_parameter('gravity').value)
        frame_id = str(self.get_parameter('frame_id').value)

        # Depth is positive when submerged (z < surface_z)
        raw_depth = surface_z - self.last_z
        noise = random.gauss(0.0, noise_std) if noise_std > 0.0 else 0.0
        measured_depth = max(0.0, raw_depth + noise)

        # Absolute hydrostatic pressure: P = P_atm + rho * g * depth
        abs_pressure = p_atm + (rho * g * measured_depth)
        pressure_variance = (noise_std * rho * g) ** 2 if noise_std > 0.0 else 0.0

        # Construct FluidPressure message
        p_msg = FluidPressure()
        if self.last_header and self.last_header.stamp.sec > 0:
            p_msg.header.stamp = self.last_header.stamp
        else:
            p_msg.header.stamp = self.get_clock().now().to_msg()
        p_msg.header.frame_id = frame_id
        p_msg.fluid_pressure = float(abs_pressure)
        p_msg.variance = float(pressure_variance)

        self.pressure_pub.publish(p_msg)
        self.depth_pub.publish(Float32(data=float(measured_depth)))


def main(args=None):
    rclpy.init(args=args)
    node = DepthSensorNode()
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
