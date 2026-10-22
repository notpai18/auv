#!/usr/bin/env python3
"""Simulated Hull Leak Sensor Node.

Simulates an optical/conductivity bilge water leak sensor (e.g. SOS Leak Sensor):
Publishes std_msgs/msg/Bool on /auv/leak at a configurable rate.
Defaults to False (hull is dry). Can be triggered dynamically via ROS 2 parameters
to simulate water ingress and test autonomous emergency surfacing logic:

    ros2 param set /leak_mock leak true
    ros2 param set /leak_mock leak false
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool


class LeakMockNode(Node):
    def __init__(self):
        super().__init__('leak_mock')

        self.declare_parameter('leak', False)
        self.declare_parameter('rate_hz', 2.0)

        rate_hz = float(self.get_parameter('rate_hz').value)
        self.pub = self.create_publisher(Bool, '/auv/leak', 10)

        self._last_leak_state = bool(self.get_parameter('leak').value)
        self.add_on_set_parameters_callback(self.on_param_change)

        timer_period = 1.0 / max(0.1, rate_hz)
        self.timer = self.create_timer(timer_period, self.publish_state)

        self.get_logger().info(
            f'LeakMockNode initialized: publishing /auv/leak at {rate_hz:.1f} Hz '
            f'(current state: {"LEAK DETECTED" if self._last_leak_state else "DRY"}). '
            'Override anytime with: ros2 param set /leak_mock leak true'
        )

    def on_param_change(self, params):
        for param in params:
            if param.name == 'leak' and param.type_ == Parameter.Type.BOOL:
                new_state = param.value
                if new_state != self._last_leak_state:
                    if new_state:
                        self.get_logger().error(
                            '*** CRITICAL ALERT: WATER LEAK DETECTED IN HULL! *** '
                            'Triggering emergency leak status on /auv/leak!'
                        )
                    else:
                        self.get_logger().info(
                            'Leak status cleared: hull dry.'
                        )
                    self._last_leak_state = new_state
        return SetParametersResult(successful=True)

    def publish_state(self):
        current_state = bool(self.get_parameter('leak').value)
        self.pub.publish(Bool(data=current_state))


def main(args=None):
    rclpy.init(args=args)
    node = LeakMockNode()
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
