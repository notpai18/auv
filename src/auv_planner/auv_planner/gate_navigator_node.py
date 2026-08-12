#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
import math

class GateNavigatorNode(Node):
    # ------------------------------------------------------------------ #
    #  Tunable parameters                                                  #
    # ------------------------------------------------------------------ #

    # Minimum detection confidence to update the stored gate coordinate.
    # Should match or be slightly higher than gate_localizer's threshold.
    CONFIDENCE_THRESHOLD = 0.10

    # Distance (m) at which we stop approaching and commit to alignment.
    # Gate should be clearly visible to the camera at this range.
    # Distance (m) at which the AUV commits to ALIGN.
    # Raised to 5.5 m so we commit before the gate fills too much of the camera
    # frame and YOLO starts returning unreliable bbox centers.
    COMMIT_DISTANCE = 5.5

    # How many consecutive ticks the gate must be within COMMIT_DISTANCE
    # before we actually commit (debounce against noisy depth readings).
    COMMIT_TICKS_REQUIRED = 5

    # Yaw error (rad) below which we consider ourselves "aligned".
    # Set to 0.05 rad (2.86°). With P-gain=4.0, command at threshold =
    # 4.0 × 0.05 = 0.20 rad/s — above Gazebo's ~0.11 rad/s actuation minimum,
    # so no deadlock. Tighter than 0.08 rad to reduce lateral crossing offset.
    ALIGN_THRESHOLD = 0.05

    # Number of consecutive ticks we must hold alignment before moving.
    ALIGN_TICKS_REQUIRED = 8

    # Extra distance (m) to drive PAST the stored gate coordinate during CROSS.
    CROSS_OVERSHOOT = 2.5

    # Stop updating gate_odom once the gate is closer than this (m).
    # Below ~6 m the gate fills too much of the camera frame and YOLO
    # starts misdetecting individual poles instead of the full gate centre.
    # At this range we have a good enough odom estimate already.
    GATE_ODOM_FREEZE_DIST = 6.0

    # Maximum allowed shift in gate_odom between two consecutive YOLO updates.
    # A jump larger than this is treated as a bad detection and discarded.
    # (The bad frame in our mission analysis moved gate_odom_y by 2.33 m.)
    MAX_GATE_ODOM_JUMP = 1.0   # metres

    # Camera mounting offset relative to base_link (from model.sdf).
    # The localizer measures depth from the CAMERA origin, but the navigator
    # adds the result as if it were measured from base_link.
    # Subtracting these offsets corrects the stored gate odom coordinate.
    #   Left camera pose: (0.30, +0.06, 0) relative to base_link
    CAM_OFFSET_FWD  = 0.30   # metres forward
    CAM_OFFSET_LEFT = 0.06   # metres left

    # Number of control-loop ticks to hold STOP before starting the U-turn.
    # At 10 Hz this equals 2 seconds — enough time for the AUV to settle.
    STOP_PAUSE_TICKS = 20

    # ------------------------------------------------------------------ #

    def __init__(self):
        super().__init__('gate_navigator_node')

        # --- State machine ---
        self.state      = "SEARCH"
        self.prev_state = "SEARCH"

        # --- Live detection data (updated by gate_position_callback) ---
        self.latest_gate_pos  = None   # [x_fwd, y_left, z_up, conf]
        self.latest_gate_time = None

        # --- Gate coordinate stored in the odom frame ---
        # Continuously updated while SEARCH/TRACK; frozen once we commit.
        self.gate_odom_x = None
        self.gate_odom_y = None

        # --- Odometry ---
        self.current_x   = None
        self.current_y   = None
        self.current_yaw = None

        # --- CROSS state ---
        self.cross_start_x = None
        self.cross_start_y = None

        # Perpendicular approach direction locked at TRACK→ALIGN commit.
        # This is the bearing robot→gate at commit time = gate's normal direction.
        # Frozen so ALIGN always rotates to the same fixed heading.
        self.approach_yaw  = None

        # --- Return-run state ---
        # The reciprocal heading: approach_yaw + π, locked when STOP fires.
        self.return_yaw = None
        # Start position for the return cross leg.
        self.return_cross_start_x = None
        self.return_cross_start_y = None
        # Counter for the STOP pause before the U-turn begins.
        self.stop_pause_tick = 0
        # Guards the DONE log so it only prints once.
        self.done_logged = False

        # --- Debounce counters ---
        self.aligned_tick_count = 0
        self.commit_tick_count  = 0

        # --- Logging ---
        self.log_tick         = 0          # increments every control loop tick
        self.yolo_update_time = None       # wall-clock time of last coord update
        self.last_coord_conf  = 0.0        # confidence of last YOLO update

        self.mission_complete_logged = False

        # --- ROS interfaces ---
        self.pos_sub = self.create_subscription(
            Float32MultiArray, '/auv/gate_position_3d',
            self.gate_position_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/auv/odom',
            self.odom_callback, 10)
        self.cmd_pub     = self.create_publisher(Twist,  '/model/auv_box/cmd_vel', 10)
        self.mission_pub = self.create_publisher(String, '/auv/mission_state',     10)
        self.debug_pub   = self.create_publisher(Float32MultiArray, '/auv/navigator_debug', 10)

        # Per-tick debug values — updated inside each state handler
        self._dbg_distance    = 0.0  # distance to gate this tick (m)
        self._dbg_bearing_err = 0.0  # bearing / yaw error this tick (rad)
        self._dbg_nav_source  = 0    # 0=none  1=yolo-live  2=odom-deadreckon

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            f"Gate Navigator started | COMMIT_DIST={self.COMMIT_DISTANCE}m  "
            f"CONF_THRESH={self.CONFIDENCE_THRESHOLD}  "
            f"STOP_PAUSE={self.STOP_PAUSE_TICKS} ticks"
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def normalize_angle(self, angle):
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def _dist_to_gate_odom(self, ref_x, ref_y):
        """Euclidean distance from (ref_x, ref_y) to stored gate_odom."""
        return math.sqrt(
            (self.gate_odom_x - ref_x) ** 2 +
            (self.gate_odom_y - ref_y) ** 2
        )

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

    def gate_position_callback(self, msg):
        # msg.data = [x_forward, y_left, z_up, confidence]
        confidence = msg.data[3]

        if confidence < self.CONFIDENCE_THRESHOLD:
            return

        self.latest_gate_pos  = msg.data
        self.latest_gate_time = self.get_clock().now()

        # --- Update stored odom-frame gate coordinate ---
        # On the forward leg: update during SEARCH / TRACK only.
        # On the return leg (RETURN_TRACK): allow YOLO to refine the
        # stored coordinate only if it passes the outlier filter —
        # dead-reckoning is the primary source so we don't corrupt the
        # good forward-leg estimate if YOLO behaves unexpectedly from
        # the opposite side.
        update_states = ("SEARCH", "TRACK", "RETURN_TRACK")
        if self.current_x is not None and self.state in update_states:
            # Subtract camera-to-base_link mounting offset so the odom
            # coordinate reflects the gate position relative to base_link,
            # not relative to the camera lens.
            x_body = msg.data[0] - self.CAM_OFFSET_FWD    # forward (body frame)
            y_body = msg.data[1] - self.CAM_OFFSET_LEFT   # left    (body frame)
            yaw    = self.current_yaw

            # ── Fix 1: Freeze once gate is too close ─────────────────────
            # Below GATE_ODOM_FREEZE_DIST the gate fills most of the camera
            # frame and YOLO bbox centres become unreliable. Stop updating.
            # Use abs() so the freeze also triggers on the return approach
            # (where x_body measured from the other side may still be < 6m).
            if abs(x_body) < self.GATE_ODOM_FREEZE_DIST:
                if self.gate_odom_x is not None:
                    self.get_logger().info(
                        f"[YOLO SKIP] Gate frozen (x_body={x_body:.2f}m "
                        f"< freeze dist {self.GATE_ODOM_FREEZE_DIST}m) "
                        f"— keeping gate_odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})"
                    )
                # Still update latest_gate_pos/time so TRACK bearing still works
                self.yolo_update_time = self.get_clock().now()
                self.last_coord_conf  = confidence
                return

            # Rotate body-frame offset into odom frame and add robot position
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

            # ── Fix 2: Outlier rejection ──────────────────────────────────
            # If the new estimate jumps more than MAX_GATE_ODOM_JUMP metres
            # from the current stored estimate, it is almost certainly a bad
            # YOLO detection (single pole, partial frame, etc.). Discard it.
            if self.gate_odom_x is not None:
                jump = math.sqrt(
                    (new_gate_odom_x - self.gate_odom_x) ** 2 +
                    (new_gate_odom_y - self.gate_odom_y) ** 2
                )
                if jump > self.MAX_GATE_ODOM_JUMP:
                    self.get_logger().warn(
                        f"[YOLO OUTLIER] gate_odom jump {jump:.2f}m > "
                        f"{self.MAX_GATE_ODOM_JUMP}m — discarding frame "
                        f"(body: fwd={x_body:.2f}m left={y_body:.2f}m conf={confidence:.2f})"
                    )
                    return

            self.gate_odom_x = new_gate_odom_x
            self.gate_odom_y = new_gate_odom_y

            # Track when and how confidently the coordinate was last updated
            self.yolo_update_time = self.get_clock().now()
            self.last_coord_conf  = confidence
            self.get_logger().info(
                f"[YOLO UPDATE] Gate odom → ({self.gate_odom_x:.2f}, {self.gate_odom_y:.2f}) "
                f"| body=({x_body:.2f}m fwd, {y_body:.2f}m left) "
                f"| conf={confidence:.2f}"
            )

    # ------------------------------------------------------------------ #
    #  Main control loop (10 Hz)                                           #
    # ------------------------------------------------------------------ #

    def control_loop(self):
        if self.current_x is None:
            return

        self.log_tick += 1
        LOG_EVERY = 10   # print status every 10 ticks = ~1 second

        # Freshness check on YOLO detection
        is_detection_fresh = False
        yolo_age_sec = None
        if self.latest_gate_time is not None:
            yolo_age_sec = (self.get_clock().now() - self.latest_gate_time).nanoseconds / 1e9
            if yolo_age_sec <= 1.5:
                is_detection_fresh = True

        # Time since last odom coordinate update
        coord_age_str = "never"
        if self.yolo_update_time is not None:
            coord_age = (self.get_clock().now() - self.yolo_update_time).nanoseconds / 1e9
            coord_age_str = f"{coord_age:.1f}s ago (conf={self.last_coord_conf:.2f})"

        cmd        = Twist()
        next_state = self.state

        # Reset alignment counter whenever we leave any ALIGN state
        if self.state not in ("ALIGN", "RETURN_ALIGN", "RETURN_ALIGN2"):
            self.aligned_tick_count = 0

        # ---- SEARCH --------------------------------------------------
        if self.state == "SEARCH":
            self.commit_tick_count = 0
            cmd.angular.z = 0.15   # slow spin
            self._dbg_distance    = 0.0
            self._dbg_bearing_err = 0.0
            self._dbg_nav_source  = 0

            if is_detection_fresh and self.gate_odom_x is not None:
                next_state = "TRACK"
                self.get_logger().info(
                    f"Gate spotted! Odom estimate: "
                    f"({self.gate_odom_x:.2f}, {self.gate_odom_y:.2f}). -> TRACK"
                )

        # ---- TRACK ---------------------------------------------------
        # Once we have a stored gate odom coordinate, we ALWAYS drive toward it.
        # YOLO is used to REFINE the coordinate when it's fresh, not to gate movement.
        # We only return to SEARCH if we have no stored coordinate at all.
        elif self.state == "TRACK":
            if self.gate_odom_x is None:
                # No coordinate at all — should not happen, but safety fallback
                next_state = "SEARCH"
            else:
                if is_detection_fresh:
                    # YOLO sees the gate — use live body-frame bearing for steering
                    x_body = self.latest_gate_pos[0]
                    y_body = self.latest_gate_pos[1]
                    bearing_error = math.atan2(y_body, x_body)
                    distance      = x_body
                    yolo_src      = "YOLO-live"
                else:
                    # YOLO lost the gate — navigate using stored odom coordinate
                    target_bearing = math.atan2(
                        self.gate_odom_y - self.current_y,
                        self.gate_odom_x - self.current_x
                    )
                    bearing_error = self.normalize_angle(target_bearing - self.current_yaw)
                    distance      = math.sqrt(
                        (self.gate_odom_x - self.current_x) ** 2 +
                        (self.gate_odom_y - self.current_y) ** 2
                    )
                    yolo_src      = "odom-deadreckon"

                # Update per-tick debug values
                self._dbg_distance    = distance
                self._dbg_bearing_err = bearing_error
                self._dbg_nav_source  = 1 if is_detection_fresh else 2

                # Periodic status log
                if self.log_tick % LOG_EVERY == 0:
                    self.get_logger().info(
                        f"[TRACK | {yolo_src}] "
                        f"dist={distance:.2f}m  bearing={math.degrees(bearing_error):.1f}deg  "
                        f"gate_odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})  "
                        f"last_YOLO_update={coord_age_str}"
                    )

                # Drive forward, correct heading with P controller
                cmd.linear.x  = 0.15
                p_term        = 1.5 * bearing_error
                cmd.angular.z = max(-0.3, min(0.3, p_term))

                # Check if we've reached commit distance
                if distance <= self.COMMIT_DISTANCE:
                    self.commit_tick_count += 1
                    if self.log_tick % LOG_EVERY == 0:
                        self.get_logger().info(
                            f"[TRACK] Within commit zone: {distance:.2f}m "
                            f"({self.commit_tick_count}/{self.COMMIT_TICKS_REQUIRED} ticks)"
                        )
                    if self.commit_tick_count >= self.COMMIT_TICKS_REQUIRED:
                        cmd.linear.x  = 0.0
                        cmd.angular.z = 0.0
                        self.commit_tick_count = 0
                        # Lock the perpendicular approach direction right now.
                        # bearing robot→gate at this moment ≈ gate normal direction.
                        self.approach_yaw = math.atan2(
                            self.gate_odom_y - self.current_y,
                            self.gate_odom_x - self.current_x
                        )
                        next_state = "ALIGN"
                        self.get_logger().info(
                            f"[TRACK → ALIGN] Committing at {distance:.2f}m. "
                            f"Gate odom: ({self.gate_odom_x:.2f}, {self.gate_odom_y:.2f}) "
                            f"| approach_yaw locked: {math.degrees(self.approach_yaw):.1f}deg "
                            f"| last YOLO update: {coord_age_str}"
                        )
                else:
                    self.commit_tick_count = 0

        # ---- ALIGN ---------------------------------------------------
        elif self.state == "ALIGN":
            if self.approach_yaw is None:
                self.get_logger().error("No approach_yaw locked! -> SEARCH")
                next_state = "SEARCH"
            else:
                # Rotate to the FROZEN approach_yaw — the gate's normal direction
                # captured at commit time. Never recomputed so the robot always
                # faces straight through the gate regardless of drift.
                yaw_error = self.normalize_angle(self.approach_yaw - self.current_yaw)

                # Update per-tick debug values
                _dist_dbg = 0.0
                if self.gate_odom_x is not None and self.current_x is not None:
                    _dist_dbg = math.sqrt(
                        (self.gate_odom_x - self.current_x) ** 2 +
                        (self.gate_odom_y - self.current_y) ** 2)
                self._dbg_distance    = _dist_dbg
                self._dbg_bearing_err = yaw_error
                self._dbg_nav_source  = 0

                # Always log alignment status every second
                if self.log_tick % LOG_EVERY == 0:
                    status = "✓ HOLDING" if abs(yaw_error) < self.ALIGN_THRESHOLD else "✗ CORRECTING"
                    self.get_logger().info(
                        f"[ALIGN {status}] yaw_err={math.degrees(yaw_error):.2f}deg  "
                        f"tick={self.aligned_tick_count}/{self.ALIGN_TICKS_REQUIRED}  "
                        f"gate_odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})  "
                        f"last_YOLO_update={coord_age_str}"
                    )

                if abs(yaw_error) < self.ALIGN_THRESHOLD:
                    self.aligned_tick_count += 1
                    if self.aligned_tick_count >= self.ALIGN_TICKS_REQUIRED:
                        # Record start position for CROSS distance tracking
                        self.cross_start_x = self.current_x
                        self.cross_start_y = self.current_y
                        next_state = "CROSS"
                        self.get_logger().info(
                            f"[ALIGN → CROSS] Aligned! Final yaw_err={math.degrees(yaw_error):.2f}deg"
                        )
                else:
                    self.aligned_tick_count = 0
                    p_term = 4.0 * yaw_error   # raised from 2.0 → 4.0 so
                    # small errors still produce ≥ 0.15 rad/s (Gazebo minimum)
                    cmd.angular.z = max(-0.3, min(0.3, p_term))

        # ---- CROSS ---------------------------------------------------
        elif self.state == "CROSS":
            # Distance from start to the stored gate coordinate
            dist_to_gate = math.sqrt(
                (self.gate_odom_x - self.cross_start_x) ** 2 +
                (self.gate_odom_y - self.cross_start_y) ** 2
            )
            # Total target = reach gate + overshoot past it
            total_target = dist_to_gate + self.CROSS_OVERSHOOT

            # Distance we've actually traveled since CROSS began
            dist_traveled = math.sqrt(
                (self.current_x - self.cross_start_x) ** 2 +
                (self.current_y - self.cross_start_y) ** 2
            )

            # Update per-tick debug values
            self._dbg_distance    = max(0.0, total_target - dist_traveled)
            self._dbg_bearing_err = 0.0
            self._dbg_nav_source  = 0

            if self.log_tick % LOG_EVERY == 0:
                self.get_logger().info(
                    f"[CROSS] traveled={dist_traveled:.2f}m / {total_target:.2f}m  "
                    f"(gate at {dist_to_gate:.2f}m + {self.CROSS_OVERSHOOT}m overshoot)  "
                    f"remaining={max(0, total_target - dist_traveled):.2f}m"
                )

            if dist_traveled >= total_target:
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
                next_state    = "STOP"
                self.get_logger().info(
                    f"[CROSS → STOP] Gate crossed! Traveled {dist_traveled:.2f}m total."
                )
            else:
                cmd.linear.x  = 0.2
                cmd.angular.z = 0.0

        # ---- STOP ----------------------------------------------------
        # Pause briefly, then lock return_yaw and begin the U-turn.
        elif self.state == "STOP":
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0

            if not self.mission_complete_logged:
                self.get_logger().info(
                    "MISSION COMPLETE — Gate passed! "
                    f"Pausing {self.STOP_PAUSE_TICKS} ticks before return run."
                )
                self.mission_complete_logged = True

            self.stop_pause_tick += 1

            if self.stop_pause_tick >= self.STOP_PAUSE_TICKS:
                # Lock the reciprocal (180°-flipped) heading for the return leg.
                self.return_yaw = self.normalize_angle(self.approach_yaw + math.pi)
                self.stop_pause_tick = 0   # reset for safety
                next_state = "RETURN_ALIGN"
                self.get_logger().info(
                    f"[STOP → RETURN_ALIGN] Starting U-turn. "
                    f"return_yaw={math.degrees(self.return_yaw):.1f}deg  "
                    f"(approach_yaw was {math.degrees(self.approach_yaw):.1f}deg)"
                )

        # ---- RETURN_ALIGN --------------------------------------------
        # Rotate 180° to face back toward the gate.
        elif self.state == "RETURN_ALIGN":
            if self.return_yaw is None:
                self.get_logger().error("[RETURN_ALIGN] No return_yaw! -> SEARCH")
                next_state = "SEARCH"
            else:
                yaw_error = self.normalize_angle(self.return_yaw - self.current_yaw)

                _dist_dbg = 0.0
                if self.gate_odom_x is not None:
                    _dist_dbg = self._dist_to_gate_odom(self.current_x, self.current_y)
                self._dbg_distance    = _dist_dbg
                self._dbg_bearing_err = yaw_error
                self._dbg_nav_source  = 0

                if self.log_tick % LOG_EVERY == 0:
                    status = "✓ HOLDING" if abs(yaw_error) < self.ALIGN_THRESHOLD else "✗ CORRECTING"
                    self.get_logger().info(
                        f"[RETURN_ALIGN {status}] yaw_err={math.degrees(yaw_error):.2f}deg  "
                        f"tick={self.aligned_tick_count}/{self.ALIGN_TICKS_REQUIRED}  "
                        f"return_yaw={math.degrees(self.return_yaw):.1f}deg"
                    )

                if abs(yaw_error) < self.ALIGN_THRESHOLD:
                    self.aligned_tick_count += 1
                    if self.aligned_tick_count >= self.ALIGN_TICKS_REQUIRED:
                        self.aligned_tick_count = 0
                        self.commit_tick_count  = 0
                        next_state = "RETURN_TRACK"
                        self.get_logger().info(
                            f"[RETURN_ALIGN → RETURN_TRACK] U-turn complete. "
                            f"Final yaw_err={math.degrees(yaw_error):.2f}deg. "
                            f"Gate odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})"
                        )
                else:
                    self.aligned_tick_count = 0
                    p_term = 4.0 * yaw_error
                    cmd.angular.z = max(-0.3, min(0.3, p_term))

        # ---- RETURN_TRACK --------------------------------------------
        # Drive back toward the stored gate odom using dead-reckoning.
        # YOLO accepted as optional refinement if it passes outlier filter.
        elif self.state == "RETURN_TRACK":
            if self.gate_odom_x is None:
                self.get_logger().error("[RETURN_TRACK] No gate_odom! -> SEARCH")
                next_state = "SEARCH"
            else:
                # Always use odom dead-reckoning as the primary bearing source
                # on the return leg — YOLO may behave differently from this side.
                target_bearing = math.atan2(
                    self.gate_odom_y - self.current_y,
                    self.gate_odom_x - self.current_x
                )
                bearing_error = self.normalize_angle(target_bearing - self.current_yaw)
                distance      = self._dist_to_gate_odom(self.current_x, self.current_y)

                self._dbg_distance    = distance
                self._dbg_bearing_err = bearing_error
                self._dbg_nav_source  = 2   # odom-deadreckon

                if self.log_tick % LOG_EVERY == 0:
                    self.get_logger().info(
                        f"[RETURN_TRACK | odom] "
                        f"dist={distance:.2f}m  bearing={math.degrees(bearing_error):.1f}deg  "
                        f"gate_odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})  "
                        f"last_YOLO_update={coord_age_str}"
                    )

                # Drive forward, correct heading with P controller
                cmd.linear.x  = 0.15
                p_term        = 1.5 * bearing_error
                cmd.angular.z = max(-0.3, min(0.3, p_term))

                # Check if we've reached commit distance for the return approach
                if distance <= self.COMMIT_DISTANCE:
                    self.commit_tick_count += 1
                    if self.log_tick % LOG_EVERY == 0:
                        self.get_logger().info(
                            f"[RETURN_TRACK] Within commit zone: {distance:.2f}m "
                            f"({self.commit_tick_count}/{self.COMMIT_TICKS_REQUIRED} ticks)"
                        )
                    if self.commit_tick_count >= self.COMMIT_TICKS_REQUIRED:
                        cmd.linear.x  = 0.0
                        cmd.angular.z = 0.0
                        self.commit_tick_count = 0
                        # Recompute return_yaw from current position at commit
                        # time (slight drift may have occurred during the drive).
                        fresh_bearing = math.atan2(
                            self.gate_odom_y - self.current_y,
                            self.gate_odom_x - self.current_x
                        )
                        self.return_yaw = fresh_bearing
                        next_state = "RETURN_ALIGN2"
                        self.get_logger().info(
                            f"[RETURN_TRACK → RETURN_ALIGN2] Committing at {distance:.2f}m. "
                            f"return_yaw re-locked: {math.degrees(self.return_yaw):.1f}deg"
                        )
                else:
                    self.commit_tick_count = 0

        # ---- RETURN_ALIGN2 -------------------------------------------
        # Final precision alignment before the return gate crossing.
        elif self.state == "RETURN_ALIGN2":
            if self.return_yaw is None:
                self.get_logger().error("[RETURN_ALIGN2] No return_yaw! -> SEARCH")
                next_state = "SEARCH"
            else:
                yaw_error = self.normalize_angle(self.return_yaw - self.current_yaw)

                _dist_dbg = 0.0
                if self.gate_odom_x is not None:
                    _dist_dbg = self._dist_to_gate_odom(self.current_x, self.current_y)
                self._dbg_distance    = _dist_dbg
                self._dbg_bearing_err = yaw_error
                self._dbg_nav_source  = 0

                if self.log_tick % LOG_EVERY == 0:
                    status = "✓ HOLDING" if abs(yaw_error) < self.ALIGN_THRESHOLD else "✗ CORRECTING"
                    self.get_logger().info(
                        f"[RETURN_ALIGN2 {status}] yaw_err={math.degrees(yaw_error):.2f}deg  "
                        f"tick={self.aligned_tick_count}/{self.ALIGN_TICKS_REQUIRED}  "
                        f"gate_odom=({self.gate_odom_x:.2f},{self.gate_odom_y:.2f})"
                    )

                if abs(yaw_error) < self.ALIGN_THRESHOLD:
                    self.aligned_tick_count += 1
                    if self.aligned_tick_count >= self.ALIGN_TICKS_REQUIRED:
                        # Record start position for RETURN_CROSS distance tracking
                        self.return_cross_start_x = self.current_x
                        self.return_cross_start_y = self.current_y
                        next_state = "RETURN_CROSS"
                        self.get_logger().info(
                            f"[RETURN_ALIGN2 → RETURN_CROSS] Aligned! "
                            f"Final yaw_err={math.degrees(yaw_error):.2f}deg"
                        )
                else:
                    self.aligned_tick_count = 0
                    p_term = 4.0 * yaw_error
                    cmd.angular.z = max(-0.3, min(0.3, p_term))

        # ---- RETURN_CROSS --------------------------------------------
        # Dead-reckoning drive back through the gate.
        elif self.state == "RETURN_CROSS":
            # Distance from return-cross start to the stored gate coordinate
            dist_to_gate = math.sqrt(
                (self.gate_odom_x - self.return_cross_start_x) ** 2 +
                (self.gate_odom_y - self.return_cross_start_y) ** 2
            )
            # Total target = reach gate + overshoot past it
            total_target = dist_to_gate + self.CROSS_OVERSHOOT

            # Distance we've actually traveled since RETURN_CROSS began
            dist_traveled = math.sqrt(
                (self.current_x - self.return_cross_start_x) ** 2 +
                (self.current_y - self.return_cross_start_y) ** 2
            )

            self._dbg_distance    = max(0.0, total_target - dist_traveled)
            self._dbg_bearing_err = 0.0
            self._dbg_nav_source  = 0

            if self.log_tick % LOG_EVERY == 0:
                self.get_logger().info(
                    f"[RETURN_CROSS] traveled={dist_traveled:.2f}m / {total_target:.2f}m  "
                    f"(gate at {dist_to_gate:.2f}m + {self.CROSS_OVERSHOOT}m overshoot)  "
                    f"remaining={max(0, total_target - dist_traveled):.2f}m"
                )

            if dist_traveled >= total_target:
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
                next_state    = "DONE"
                self.get_logger().info(
                    f"[RETURN_CROSS → DONE] Return gate crossed! "
                    f"Traveled {dist_traveled:.2f}m. ROUND TRIP COMPLETE."
                )
            else:
                cmd.linear.x  = 0.2
                cmd.angular.z = 0.0

        # ---- DONE ----------------------------------------------------
        elif self.state == "DONE":
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            if not self.done_logged:
                self.get_logger().info(
                    "★ ROUND TRIP COMPLETE — Both gate crossings done! ★"
                )
                self.done_logged = True

        # ---- State transition log ------------------------------------
        if self.state != self.prev_state:
            self.get_logger().info(
                f"[STATE] {self.prev_state} -> {self.state}"
            )
            self.prev_state = self.state

        state_msg      = String()
        state_msg.data = self.state
        self.mission_pub.publish(state_msg)
        self.cmd_pub.publish(cmd)

        # --- Publish full navigator debug every tick ---
        # Format: [gate_odom_x, gate_odom_y, approach_yaw_deg,
        #          distance_m, bearing_err_deg,
        #          commit_ticks, align_ticks,
        #          yolo_fresh, yolo_age_sec, nav_source]
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

        self.state = next_state


def main(args=None):
    rclpy.init(args=args)
    node = GateNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()