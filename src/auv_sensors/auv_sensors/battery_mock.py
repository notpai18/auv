#!/usr/bin/env python3
"""Simulated Battery Monitor Node.

Simulates an on-board Power Sense Module / Smart Battery (e.g., 4S LiPo battery pack):
Publishes sensor_msgs/msg/BatteryState on /auv/battery.
Supports gradual discharge over time based on simulated average current draw,
as well as manual override of battery percentage via parameter for mission abort / low-battery testing:

    ros2 param set /battery_mock percentage_override 0.15  # test low battery warning
    ros2 param set /battery_mock percentage_override -1.0  # resume discharge model
"""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState


class BatteryMockNode(Node):
    def __init__(self):
        super().__init__('battery_mock')

        # Parameters
        self.declare_parameter('voltage_full', 16.8)         # 4S LiPo fully charged (4.2V/cell)
        self.declare_parameter('voltage_empty', 13.2)        # 4S LiPo empty cutoff (3.3V/cell)
        self.declare_parameter('capacity_ah', 10.0)          # 10,000 mAh
        self.declare_parameter('current_draw_a', 4.5)        # Average continuous load in Amperes
        self.declare_parameter('simulate_discharge', True)   # Decrease charge over time
        self.declare_parameter('rate_hz', 1.0)               # Publish frequency
        self.declare_parameter('percentage_override', -1.0)  # [0.0 - 1.0] or -1 to disable
        self.declare_parameter('frame_id', 'base_link')

        rate_hz = float(self.get_parameter('rate_hz').value)
        self.pub = self.create_publisher(BatteryState, '/auv/battery', 10)

        self._capacity_ah = float(self.get_parameter('capacity_ah').value)
        self._current_charge_ah = self._capacity_ah
        self._last_time = time.time()

        timer_period = 1.0 / max(0.1, rate_hz)
        self.timer = self.create_timer(timer_period, self.publish_battery)

        self.get_logger().info(
            f'BatteryMockNode initialized: publishing /auv/battery at {rate_hz:.1f} Hz '
            f'({self._capacity_ah:.1f} Ah, 4S LiPo).'
        )

    def publish_battery(self):
        now = time.time()
        dt_hours = (now - self._last_time) / 3600.0
        self._last_time = now

        v_full = float(self.get_parameter('voltage_full').value)
        v_empty = float(self.get_parameter('voltage_empty').value)
        cap = float(self.get_parameter('capacity_ah').value)
        current_draw = float(self.get_parameter('current_draw_a').value)
        sim_discharge = bool(self.get_parameter('simulate_discharge').value)
        override = float(self.get_parameter('percentage_override').value)
        frame_id = str(self.get_parameter('frame_id').value)

        # Update simulated charge
        if sim_discharge and dt_hours > 0:
            self._current_charge_ah = max(0.0, self._current_charge_ah - (current_draw * dt_hours))

        if 0.0 <= override <= 1.0:
            percentage = override
            charge = cap * percentage
        else:
            percentage = max(0.0, min(1.0, self._current_charge_ah / max(0.001, cap)))
            charge = self._current_charge_ah

        # Simple linear approximation of voltage across LiPo discharge curve
        voltage = v_empty + (v_full - v_empty) * percentage

        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.voltage = float(voltage)
        msg.temperature = 24.5  # Nominal internal temperature in Celsius
        msg.current = float(-abs(current_draw))  # Negative indicates discharge
        msg.charge = float(charge)
        msg.capacity = float(cap)
        msg.design_capacity = float(cap)
        msg.percentage = float(percentage)

        if percentage <= 0.05:
            msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_NOT_CHARGING
            msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_DEAD
        else:
            msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD

        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        msg.present = True

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BatteryMockNode()
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
