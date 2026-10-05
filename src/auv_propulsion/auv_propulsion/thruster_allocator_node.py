#!/usr/bin/env python3
"""
thruster_allocator_node.py — 8-Thruster Moore-Penrose Pseudo-Inverse Allocation Node.

Translates desired 6-DOF body wrench tau = [Fx, Fy, Fz, Mx, My, Mz]^T into
individual thrusts [u1, ..., u8]^T for the 8 physical thrusters on the AUV hull.

Geometry:
  T1..T4: Horizontal thrusters at +/-45 deg for surge, sway, yaw
  T5..T8: Vertical thrusters at corners for heave, roll, pitch
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64, Float32MultiArray


class ThrusterAllocatorNode(Node):

    def __init__(self):
        super().__init__('thruster_allocator_node')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('wrench_topic', '/auv/wrench_cmd')
        self.declare_parameter('max_thrust_n', 40.0)      # Saturation limit per thruster (+N)
        self.declare_parameter('min_thrust_n', -40.0)     # Saturation limit per thruster (-N)
        self.declare_parameter('publish_rate_hz', 50.0)   # Control loop rate
        self.declare_parameter('wrench_timeout_sec', 0.5) # Zero thrusters if wrench stale

        p = self.get_parameter
        self._max_thrust = float(p('max_thrust_n').value)
        self._min_thrust = float(p('min_thrust_n').value)
        self._wrench_timeout = float(p('wrench_timeout_sec').value)
        wrench_topic = p('wrench_topic').value
        publish_hz = float(p('publish_rate_hz').value)

        # ── Thruster Allocation Matrix B (6x8) ─────────────────────────
        # Rows: Fx, Fy, Fz, Mx, My, Mz in base_link frame
        # Cols: T1, T2, T3, T4, T5, T6, T7, T8
        self._B = np.array([
            [ 0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.0       ,  0.0       ,  0.0       ,  0.0        ],
            [ 0.70710678, -0.70710678, -0.70710678,  0.70710678,  0.0       ,  0.0       ,  0.0       ,  0.0        ],
            [ 0.0       ,  0.0       ,  0.0       ,  0.0       , -1.0       ,  1.0       ,  1.0       , -1.0        ],
            [-0.00898025,  0.00898025,  0.00898025, -0.00898025,  0.3647    ,  0.3647    , -0.3647    , -0.3647    ],
            [ 0.00898025,  0.00898025,  0.00898025,  0.00898025,  0.125     , -0.125     ,  0.125     , -0.125     ],
            [ 0.43593121, -0.43593121,  0.43593121, -0.43593121,  0.0       ,  0.0       ,  0.0       ,  0.0        ]
        ], dtype=np.float64)

        # Moore-Penrose pseudo-inverse: u = B^+ * tau
        self._B_pinv = np.linalg.pinv(self._B)

        self._latest_wrench = np.zeros(6, dtype=np.float64)
        self._last_wrench_time = 0.0

        # ── ROS Subscribers & Publishers ────────────────────────────────
        self._wrench_sub = self.create_subscription(
            Wrench, wrench_topic, self._wrench_cb, 10)

        # Publishers for each of the 8 thrusters in Gazebo
        self._thruster_pubs = []
        for i in range(1, 9):
            topic = f'/model/auv_box/joint/T_{i}/cmd_thrust'
            pub = self.create_publisher(Float64, topic, 10)
            self._thruster_pubs.append(pub)

        # Combined debug/telemetry publisher
        self._debug_pub = self.create_publisher(Float32MultiArray, '/auv/thruster_forces', 10)

        dt = 1.0 / max(publish_hz, 1.0)
        self._timer = self.create_timer(dt, self._control_loop)

        self.get_logger().info(
            f'Thruster Allocator running: {wrench_topic} -> 8 thrusters @ {publish_hz:.0f} Hz '
            f'[limit: {self._min_thrust:.1f} to {self._max_thrust:.1f} N]'
        )

    def _wrench_cb(self, msg: Wrench):
        self._latest_wrench[0] = msg.force.x
        self._latest_wrench[1] = msg.force.y
        self._latest_wrench[2] = msg.force.z
        self._latest_wrench[3] = msg.torque.x
        self._latest_wrench[4] = msg.torque.y
        self._latest_wrench[5] = msg.torque.z
        self._last_wrench_time = time.monotonic()

    def _control_loop(self):
        now = time.monotonic()
        if (now - self._last_wrench_time) > self._wrench_timeout:
            # Stale timeout: zero wrench to stop propulsion safely
            tau = np.zeros(6, dtype=np.float64)
        else:
            tau = self._latest_wrench

        # Compute optimal thrust vector via pseudo-inverse: u = B^+ * tau
        u = self._B_pinv @ tau

        # Clamp to thruster physical limits
        u_clamped = np.clip(u, self._min_thrust, self._max_thrust)

        # Publish to individual Gazebo thruster topics
        debug_msg = Float32MultiArray()
        for i, val in enumerate(u_clamped):
            f_msg = Float64()
            f_msg.data = float(val)
            self._thruster_pubs[i].publish(f_msg)
            debug_msg.data.append(float(val))

        self._debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ThrusterAllocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
