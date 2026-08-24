#!/usr/bin/env python3
"""
gate_navigator_v2_node.py
=========================
Simplified one-pass gate navigator (no U-turn / return leg).

State machine
-------------
  FORWARD  → Drive straight forward for `forward_distance` metres (default 3.5 m)
               from the spawn position, using odometry dead-reckoning.
  SEARCH   → Sweep ±search_sweep_deg (default ±90°) from the forward heading,
               reversing at each limit, until the gate is detected.
  TRACK    → Drive toward the gate (YOLO live + odom dead-reckoning fallback).
  ALIGN    → Rotate to the locked gate-normal heading (gate perpendicular approach).
  CROSS    → Drive straight through the gate (gate_dist + cross_overshoot).
  DONE     → Stop. Mission complete.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray, Float32


class GateNavigatorV2Node(Node):

    def __init__(self):
        super().__init__('gate_navigator_v2_node')

        # ------------------------------------------------------------------ #
        #  Tunable parameters (overridable via YAML or CLI)                   #
        # ------------------------------------------------------------------ #
        self.declare_parameter('forward_distance',      3.5)
        self.declare_parameter('forward_speed',         0.20)
        self.declare_parameter('confidence_threshold',  0.10)
        self.declare_parameter('commit_distance',       5.5)
        self.declare_parameter('commit_ticks_required', 5)
        self.declare_parameter('align_threshold',       0.05)
        self.declare_parameter('align_ticks_required',  8)
        self.declare_parameter('cross_overshoot',       2.5)
        self.declare_parameter('gate_odom_freeze_dist', 6.0)
        self.declare_parameter('max_gate_odom_jump',    1.0)
        self.declare_parameter('cam_offset_fwd',        0.30)
        self.declare_parameter('cam_offset_left',       0.06)
        self.declare_parameter('yolo_freshness_sec',    1.5)
        self.declare_parameter('search_strafe_speed',   0.15)
        self.declare_parameter('search_sweep_dist',     6.0)
        self.declare_parameter('track_speed',           0.15)
        self.declare_parameter('cross_speed',           0.20)
        self.declare_parameter('track_p_gain',          1.5)
        self.declare_parameter('align_p_gain',          4.0)
        self.declare_parameter('angular_clamp',         0.30)

        p = self.get_parameter
        self.p_forward_distance      = p('forward_distance').value
        self.p_forward_speed         = p('forward_speed').value
        self.p_confidence_threshold  = p('confidence_threshold').value
        self.p_commit_distance       = p('commit_distance').value
        self.p_commit_ticks_required = p('commit_ticks_required').value
        self.p_align_threshold       = p('align_threshold').value
        self.p_align_ticks_required  = p('align_ticks_required').value
        self.p_cross_overshoot       = p('cross_overshoot').value
        self.p_gate_odom_freeze_dist = p('gate_odom_freeze_dist').value
        self.p_max_gate_odom_jump    = p('max_gate_odom_jump').value
        self.p_cam_offset_fwd        = p('cam_offset_fwd').value
        self.p_cam_offset_left       = p('cam_offset_left').value
        self.p_yolo_freshness_sec    = p('yolo_freshness_sec').value
        self.p_search_strafe_speed   = p('search_strafe_speed').value
        self.p_search_sweep_dist     = p('search_sweep_dist').value
        self.p_track_speed           = p('track_speed').value
        self.p_cross_speed           = p('cross_speed').value
        self.p_track_p_gain          = p('track_p_gain').value
        self.p_align_p_gain          = p('align_p_gain').value
        self.p_angular_clamp         = p('angular_clamp').value

        # ------------------------------------------------------------------ #
        #  State machine                                                       #
        # ------------------------------------------------------------------ #
        self.state      = 'FORWARD'
        self.prev_state = 'FORWARD'

        # FORWARD phase bookkeeping
        self.forward_start_x   = None
        self.forward_start_y   = None
        self.forward_start_yaw = None   # published for green_navigator_node
                                         # to reset its heading to after DONE

        # SEARCH phase bookkeeping
        # Position latched at the moment FORWARD ends; sweep is laterally ±p_search_sweep_dist.
        self.search_start_x    = None
        self.search_start_y    = None
        self.search_start_yaw  = None
        self.search_direction  = -1    # -1 = right (sway), 1 = left

        # Live YOLO detection
        self.latest_gate_pos  = None
        self.latest_gate_time = None

        # Gate position in odom frame
        self.gate_odom_x = None
        self.gate_odom_y = None

        # Odometry
        self.current_x   = None
        self.current_y   = None
        self.current_yaw = None

        # ALIGN / CROSS bookkeeping
        self.approach_yaw  = None
        self.cross_start_x = None
        self.cross_start_y = None

        # Debounce counters
        self.aligned_tick_count = 0
        self.commit_tick_count  = 0

        # Logging helpers
        self.log_tick         = 0
        self.yolo_update_time = None
        self.last_coord_conf  = 0.0
        self.done_logged      = False

        # ------------------------------------------------------------------ #
        #  ROS interfaces                                                      #
        # ------------------------------------------------------------------ #
        self.pos_sub = self.create_subscription(
            Float32MultiArray, '/auv/gate_position_3d',
            self.gate_position_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/auv/odom',
            self.odom_callback, 10)
        self.cmd_pub     = self.create_publisher(Twist,             '/model/auv_box/cmd_vel', 10)
        self.mission_pub = self.create_publisher(String,            '/auv/mission_state',     10)
        self.debug_pub   = self.create_publisher(Float32MultiArray, '/auv/navigator_debug',   10)
        # The heading the AUV was facing at mission start, before any
        # rotation — lets green_navigator_node reset to "facing forward"
        # after crossing, instead of continuing on whatever heading ALIGN
        # left it at (which can be slightly off from dead-straight).
        self.forward_yaw_pub = self.create_publisher(Float32, '/auv/forward_reference_yaw', 10)

        self._dbg_distance    = 0.0
        self._dbg_bearing_err = 0.0
        self._dbg_nav_source  = 0

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            f'GateNavigatorV2 started | '
            f'FORWARD={self.p_forward_distance}m  '
            f'COMMIT_DIST={self.p_commit_distance}m  '
            f'CONF_THRESH={self.p_confidence_threshold}'
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def normalize_angle(self, angle):
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        if self.forward_start_x is None:
            self.forward_start_x   = self.current_x
            self.forward_start_y   = self.current_y
            self.forward_start_yaw = self.current_yaw
            self.get_logger().info(
                f'[FORWARD] Start position latched: '
                f'({self.forward_start_x:.2f}, {self.forward_start_y:.2f}) '
                f'yaw={math.degrees(self.forward_start_yaw):.1f}deg'
            )

    def gate_position_callback(self, msg):
        confidence = msg.data[3]
        if confidence < self.p_confidence_threshold:
            return

        self.latest_gate_pos  = msg.data
        self.latest_gate_time = self.get_clock().now()

        if self.current_x is None or self.state not in ('FORWARD', 'SEARCH', 'TRACK'):
            return

        x_body = msg.data[0] - self.p_cam_offset_fwd
        y_body = msg.data[1] - self.p_cam_offset_left
        yaw    = self.current_yaw

        if abs(x_body) < self.p_gate_odom_freeze_dist and self.gate_odom_x is not None:
            self.get_logger().info(
                f'[YOLO SKIP] Gate frozen (x_body={x_body:.2f}m < '
                f'{self.p_gate_odom_freeze_dist}m) '
                f'-- keeping gate_odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})'
            )
            self.yolo_update_time = self.get_clock().now()
            self.last_coord_conf  = confidence
            return

        new_gate_odom_x = (
            self.current_x
            + x_body * math.cos(yaw)
            - y_body * math.sin(yaw)
        )
        new_gate_odom_y = (
            self.current_y
            + x_body * math.sin(yaw)
            + y_body * math.cos(yaw)
        )

        if self.gate_odom_x is not None:
            jump = math.sqrt(
                (new_gate_odom_x - self.gate_odom_x) ** 2 +
                (new_gate_odom_y - self.gate_odom_y) ** 2
            )
            if jump > self.p_max_gate_odom_jump:
                self.get_logger().warn(
                    f'[YOLO OUTLIER] gate_odom jump {jump:.2f}m > '
                    f'{self.p_max_gate_odom_jump}m -- discarding frame '
                    f'(body: fwd={x_body:.2f}m left={y_body:.2f}m conf={confidence:.2f})'
                )
                return

        self.gate_odom_x = new_gate_odom_x
        self.gate_odom_y = new_gate_odom_y
        self.yolo_update_time = self.get_clock().now()
        self.last_coord_conf  = confidence
        self.get_logger().info(
            f'[YOLO UPDATE] Gate odom -> ({self.gate_odom_x:.2f}, {self.gate_odom_y:.2f}) '
            f'| body=({x_body:.2f}m fwd, {y_body:.2f}m left) '
            f'| conf={confidence:.2f}'
        )

    # ------------------------------------------------------------------ #
    #  Main control loop (10 Hz)                                           #
    # ------------------------------------------------------------------ #

    def control_loop(self):
        if self.current_x is None:
            return

        self.log_tick += 1
        LOG_EVERY = 10

        is_detection_fresh = False
        yolo_age_sec = None
        if self.latest_gate_time is not None:
            yolo_age_sec = (self.get_clock().now() - self.latest_gate_time).nanoseconds / 1e9
            is_detection_fresh = (yolo_age_sec <= self.p_yolo_freshness_sec)

        coord_age_str = 'never'
        if self.yolo_update_time is not None:
            coord_age = (self.get_clock().now() - self.yolo_update_time).nanoseconds / 1e9
            coord_age_str = f'{coord_age:.1f}s ago (conf={self.last_coord_conf:.2f})'

        cmd        = Twist()
        next_state = self.state

        if self.state != 'ALIGN':
            self.aligned_tick_count = 0

        # ============================================================ #
        # FORWARD -- drive straight for forward_distance metres        #
        # ============================================================ #
        if self.state == 'FORWARD':
            if self.forward_start_x is None:
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
            else:
                dist_traveled = math.sqrt(
                    (self.current_x - self.forward_start_x) ** 2 +
                    (self.current_y - self.forward_start_y) ** 2
                )
                self._dbg_distance    = max(0.0, self.p_forward_distance - dist_traveled)
                self._dbg_bearing_err = 0.0
                self._dbg_nav_source  = 0

                if self.log_tick % LOG_EVERY == 0:
                    self.get_logger().info(
                        f'[FORWARD] traveled={dist_traveled:.2f}m / '
                        f'{self.p_forward_distance:.1f}m  '
                        f'remaining={max(0.0, self.p_forward_distance - dist_traveled):.2f}m'
                    )

                # ── Gate spotted during FORWARD? Skip SEARCH entirely ───────
                if is_detection_fresh and self.gate_odom_x is not None:
                    cmd.linear.x  = 0.0
                    cmd.angular.z = 0.0
                    next_state    = 'TRACK'
                    self.get_logger().info(
                        f'[FORWARD -> TRACK] Gate detected at {dist_traveled:.2f}m in! '
                        f'Skipping SEARCH. Gate odom: '
                        f'({self.gate_odom_x:.2f}, {self.gate_odom_y:.2f})'
                    )
                elif dist_traveled >= self.p_forward_distance:
                    cmd.linear.x  = 0.0
                    cmd.angular.z = 0.0
                    # Latch the state we've been driving on as the search centre
                    self.search_start_x    = self.current_x
                    self.search_start_y    = self.current_y
                    self.search_start_yaw  = self.current_yaw
                    self.search_direction  = -1    # start sweeping right
                    next_state    = 'SEARCH'
                    self.get_logger().info(
                        f'[FORWARD -> SEARCH] Reached {dist_traveled:.2f}m. '
                        f'Strafing laterally ±{self.p_search_sweep_dist:.1f}m'
                    )
                else:
                    cmd.linear.x  = self.p_forward_speed
                    cmd.angular.z = 0.0

        # ============================================================ #
        # SEARCH -- strafe laterally ±search_sweep_dist                #
        # ============================================================ #
        elif self.state == 'SEARCH':
            self.commit_tick_count = 0
            self._dbg_distance    = 0.0
            self._dbg_nav_source  = 0

            if self.search_start_yaw is None:
                # Safety
                self.search_start_x    = self.current_x
                self.search_start_y    = self.current_y
                self.search_start_yaw  = self.current_yaw
                self.search_direction  = -1

            # Calculate lateral distance from start point
            dx = self.current_x - self.search_start_x
            dy = self.current_y - self.search_start_y
            
            # Project onto local Y axis (lateral axis)
            lateral_dist = -dx * math.sin(self.search_start_yaw) + dy * math.cos(self.search_start_yaw)
            
            # Correct yaw drift while strafing
            yaw_err = self.normalize_angle(self.search_start_yaw - self.current_yaw)
            self._dbg_bearing_err = yaw_err

            # Reverse direction when a limit is reached
            if lateral_dist >= self.p_search_sweep_dist:
                self.search_direction = -1   # hit left limit, sweep right
                self.get_logger().info(
                    f'[SEARCH] Hit left limit (+{self.p_search_sweep_dist:.1f}m) '
                    f'— reversing to right'
                )
            elif lateral_dist <= -self.p_search_sweep_dist:
                self.search_direction = 1    # hit right limit, sweep left
                self.get_logger().info(
                    f'[SEARCH] Hit right limit (-{self.p_search_sweep_dist:.1f}m) '
                    f'— reversing to left'
                )

            cmd.linear.y = self.search_direction * self.p_search_strafe_speed
            
            # P-control to maintain heading
            cmd.angular.z = max(-self.p_angular_clamp, min(self.p_angular_clamp, self.p_align_p_gain * yaw_err))

            if self.log_tick % LOG_EVERY == 0:
                self.get_logger().info(
                    f'[SEARCH] lateral_dist={lateral_dist:.2f}m '
                    f'limit=±{self.p_search_sweep_dist:.1f}m '
                    f'dir={"LEFT" if self.search_direction > 0 else "RIGHT"}'
                )

            if is_detection_fresh and self.gate_odom_x is not None:
                next_state = 'TRACK'
                self.get_logger().info(
                    f'[SEARCH -> TRACK] Gate spotted! '
                    f'Odom estimate: ({self.gate_odom_x:.2f}, {self.gate_odom_y:.2f})'
                )

        # ============================================================ #
        # TRACK -- drive toward gate (YOLO + dead-reckoning)          #
        # ============================================================ #
        elif self.state == 'TRACK':
            if self.gate_odom_x is None:
                next_state = 'SEARCH'
            else:
                if is_detection_fresh:
                    x_body        = self.latest_gate_pos[0]
                    y_body        = self.latest_gate_pos[1]
                    bearing_error = math.atan2(y_body, x_body)
                    distance      = x_body
                    src_label     = 'YOLO-live'
                else:
                    target_bearing = math.atan2(
                        self.gate_odom_y - self.current_y,
                        self.gate_odom_x - self.current_x
                    )
                    bearing_error = self.normalize_angle(target_bearing - self.current_yaw)
                    distance      = math.sqrt(
                        (self.gate_odom_x - self.current_x) ** 2 +
                        (self.gate_odom_y - self.current_y) ** 2
                    )
                    src_label = 'odom-deadreckon'

                self._dbg_distance    = distance
                self._dbg_bearing_err = bearing_error
                self._dbg_nav_source  = 1 if is_detection_fresh else 2

                if self.log_tick % LOG_EVERY == 0:
                    self.get_logger().info(
                        f'[TRACK | {src_label}] '
                        f'dist={distance:.2f}m  bearing={math.degrees(bearing_error):.1f}deg  '
                        f'gate_odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})  '
                        f'last_YOLO={coord_age_str}'
                    )

                cmd.linear.x  = self.p_track_speed
                p_term        = self.p_track_p_gain * bearing_error
                cmd.angular.z = max(-self.p_angular_clamp, min(self.p_angular_clamp, p_term))

                if distance <= self.p_commit_distance:
                    self.commit_tick_count += 1
                    if self.log_tick % LOG_EVERY == 0:
                        self.get_logger().info(
                            f'[TRACK] In commit zone: {distance:.2f}m '
                            f'({self.commit_tick_count}/{self.p_commit_ticks_required} ticks)'
                        )
                    if self.commit_tick_count >= self.p_commit_ticks_required:
                        cmd.linear.x  = 0.0
                        cmd.angular.z = 0.0
                        self.commit_tick_count = 0
                        self.approach_yaw = math.atan2(
                            self.gate_odom_y - self.current_y,
                            self.gate_odom_x - self.current_x
                        )
                        next_state = 'ALIGN'
                        self.get_logger().info(
                            f'[TRACK -> ALIGN] Committing at {distance:.2f}m. '
                            f'approach_yaw={math.degrees(self.approach_yaw):.1f}deg '
                            f'| last YOLO={coord_age_str}'
                        )
                else:
                    self.commit_tick_count = 0

        # ============================================================ #
        # ALIGN -- rotate to gate-normal heading                       #
        # ============================================================ #
        elif self.state == 'ALIGN':
            if self.approach_yaw is None:
                self.get_logger().error('No approach_yaw locked! -> SEARCH')
                next_state = 'SEARCH'
            else:
                yaw_error = self.normalize_angle(self.approach_yaw - self.current_yaw)

                _dist_dbg = 0.0
                if self.gate_odom_x is not None:
                    _dist_dbg = math.sqrt(
                        (self.gate_odom_x - self.current_x) ** 2 +
                        (self.gate_odom_y - self.current_y) ** 2
                    )
                self._dbg_distance    = _dist_dbg
                self._dbg_bearing_err = yaw_error
                self._dbg_nav_source  = 0

                if self.log_tick % LOG_EVERY == 0:
                    status = 'HOLDING' if abs(yaw_error) < self.p_align_threshold else 'CORRECTING'
                    self.get_logger().info(
                        f'[ALIGN {status}] yaw_err={math.degrees(yaw_error):.2f}deg  '
                        f'tick={self.aligned_tick_count}/{self.p_align_ticks_required}  '
                        f'gate_odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})'
                    )

                if abs(yaw_error) < self.p_align_threshold:
                    self.aligned_tick_count += 1
                    if self.aligned_tick_count >= self.p_align_ticks_required:
                        self.cross_start_x = self.current_x
                        self.cross_start_y = self.current_y
                        next_state = 'CROSS'
                        self.get_logger().info(
                            f'[ALIGN -> CROSS] Aligned! '
                            f'Final yaw_err={math.degrees(yaw_error):.2f}deg'
                        )
                else:
                    self.aligned_tick_count = 0
                    p_term = self.p_align_p_gain * yaw_error
                    cmd.angular.z = max(-self.p_angular_clamp, min(self.p_angular_clamp, p_term))

        # ============================================================ #
        # CROSS -- drive through the gate and past it                 #
        # ============================================================ #
        elif self.state == 'CROSS':
            dist_to_gate = math.sqrt(
                (self.gate_odom_x - self.cross_start_x) ** 2 +
                (self.gate_odom_y - self.cross_start_y) ** 2
            )
            total_target = dist_to_gate + self.p_cross_overshoot

            dist_traveled = math.sqrt(
                (self.current_x - self.cross_start_x) ** 2 +
                (self.current_y - self.cross_start_y) ** 2
            )

            self._dbg_distance    = max(0.0, total_target - dist_traveled)
            self._dbg_bearing_err = 0.0
            self._dbg_nav_source  = 0

            if self.log_tick % LOG_EVERY == 0:
                self.get_logger().info(
                    f'[CROSS] traveled={dist_traveled:.2f}m / {total_target:.2f}m  '
                    f'(gate at {dist_to_gate:.2f}m + {self.p_cross_overshoot}m overshoot)  '
                    f'remaining={max(0.0, total_target - dist_traveled):.2f}m'
                )

            if dist_traveled >= total_target:
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
                next_state    = 'DONE'
                self.get_logger().info(
                    f'[CROSS -> DONE] Gate crossed! '
                    f'Traveled {dist_traveled:.2f}m. MISSION COMPLETE.'
                )
            else:
                cmd.linear.x  = self.p_cross_speed
                cmd.angular.z = 0.0

        # ============================================================ #
        # DONE -- hold position                                        #
        # ============================================================ #
        elif self.state == 'DONE':
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            if not self.done_logged:
                self.get_logger().info('*** MISSION COMPLETE -- Gate crossed! ***')
                self.done_logged = True

        # ---- State transition log ------------------------------------
        # Update state first so all checks below use the new state
        self.state = next_state

        if self.state != self.prev_state:
            self.get_logger().info(f'[STATE] {self.prev_state} -> {self.state}')
            self.prev_state = self.state

        state_msg      = String()
        state_msg.data = self.state
        self.mission_pub.publish(state_msg)
        if self.state != 'DONE':
            self.cmd_pub.publish(cmd)

        if self.forward_start_yaw is not None:
            yaw_msg = Float32()
            yaw_msg.data = float(self.forward_start_yaw)
            self.forward_yaw_pub.publish(yaw_msg)

        _nan = float('nan')
        dbg_msg = Float32MultiArray()
        dbg_msg.data = [
            float(self.gate_odom_x)         if self.gate_odom_x is not None  else _nan,
            float(self.gate_odom_y)         if self.gate_odom_y is not None  else _nan,
            math.degrees(self.approach_yaw) if self.approach_yaw is not None else _nan,
            float(self._dbg_distance),
            math.degrees(self._dbg_bearing_err),
            float(self.commit_tick_count),
            float(self.aligned_tick_count),
            1.0 if is_detection_fresh else 0.0,
            float(yolo_age_sec) if yolo_age_sec is not None else -1.0,
            float(self._dbg_nav_source),
        ]
        self.debug_pub.publish(dbg_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GateNavigatorV2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
