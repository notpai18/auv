#!/usr/bin/env python3
"""
green_navigator_node.py
=======================
FSM for post-gate navigation.

Full state machine:
  IDLE
    -> (gate DONE received) RESET_YAW (-> SEARCH_GREEN)

  RESET_YAW  (rotate back to the forward-reference heading published by
              gate_navigator_v2_node — the heading the AUV faced at mission
              start, before FORWARD/SEARCH/ALIGN ever turned it. Used both
              after crossing the gate and again before the blue-bin search,
              so every lateral-strafe search below always scans with the
              camera facing dead ahead, not whatever heading the previous
              approach happened to leave it at.)
    -> (aligned) SEARCH_GREEN or SEARCH_BLUE_BIN, depending on which
       leg of the mission asked for the reset

  SEARCH_GREEN  (advance-and-sweep search for the green mat — same lateral
                 strafe gate_navigator_v2_node's SEARCH state already uses
                 for the gate, ± green_search_sweep_dist holding the reset
                 heading, but if a full pass finds nothing it also creeps
                 forward search_advance_step_m and sweeps again, since the
                 mat can be well beyond camera range from the gate exit.
                 No hardcoded target coordinate anywhere.)
    -> (green found) APPROACH_GREEN
    -> (advance budget exhausted) RESET_YAW (-> SEARCH_BLUE_BIN), because
       blue-bin YOLO detection is reliable on its own — no need to gate
       the whole mission on ever visually confirming the mat first
       [governed by skip_to_bin_if_green_not_found, default True]

  APPROACH_GREEN  (drive toward green mat odom coordinate)
    -> (bottom cam sees green OR within 2 m of target) RISE_FOR_SURVEY

  RISE_FOR_SURVEY  (climb to bin_survey_z — a wide-FOV altitude — before
                    starting the blue-bin search. The front camera loses
                    the bin at close range (~1.1m) and its one-shot depth
                    estimate can't self-correct once frozen (confirmed
                    live: it overshot the real bin by ~1m and the AUV flew
                    straight over it without ever registering "reached").
                    From survey altitude the bottom camera's YOLO detector
                    can see the whole green mat and spot the bin directly,
                    continuously refining its position as the AUV closes
                    in instead of trusting one frozen extrapolation.
                    Bounded well clear of the water surface.)
    -> (at altitude) RESET_YAW (-> SEARCH_BLUE_BIN)

  SEARCH_BLUE_BIN  (same advance-and-sweep pattern as SEARCH_GREEN, scoped
                    tighter since the AUV is already over/near the green
                    zone. "Found" now accepts either a fresh front-camera
                    sighting or a fresh bottom-camera sighting that is also
                    confirmed to be over the green mat — see
                    _bottom_on_green — so a stray blue-water false positive
                    over open pool can't be mistaken for the bin.)
    -> (bin found) APPROACH_BLUE_BIN

  APPROACH_BLUE_BIN  (drive to the stored odom coordinate of the bin — kept
                      continuously fresh by the bottom camera once over the
                      green mat, with the front-camera-only extrapolation as
                      a fallback before that first bottom sighting)
    -> (within 0.35 m, OR bottom cam sees the bin directly over green)
       RISE_FOR_VERIFICATION

  RISE_FOR_VERIFICATION  (rise 0.5 m so bottom cam gets wider view — skipped
                          if already at/above bin_survey_z, since that view
                          already exists)
    -> (ticks done, or already high enough) CENTER_BLUE_BIN

  CENTER_BLUE_BIN  (use bottom camera to fine-center; odom fallback)
    -> (centered for N ticks) HOLD

  HOLD -> (5 s) FINAL_DONE
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray, Float32
import math


class GreenNavigatorNode(Node):

    def __init__(self):
        super().__init__('green_navigator_node')

        # ------------------------------------------------------------------ #
        #  ROS 2 Parameters                                                    #
        # ------------------------------------------------------------------ #
        self.declare_parameter('gate_done_value',        'DONE')

        self.declare_parameter('search_spin_speed',       0.12)   # rad/s
        self.declare_parameter('approach_speed',          0.30)   # m/s
        self.declare_parameter('center_speed',            0.05)   # m/s

        self.declare_parameter('approach_p_gain',         1.2)
        self.declare_parameter('center_p_gain',           0.8)
        self.declare_parameter('angular_clamp',           0.20)   # rad/s

        self.declare_parameter('phase2_coverage_thresh',  0.03)
        self.declare_parameter('center_threshold_m',      0.35)   # m
        self.declare_parameter('center_ticks_required',   10)
        self.declare_parameter('hold_ticks',              50)

        self.declare_parameter('green_freshness_sec',     2.0)
        self.declare_parameter('green_approach_stop_dist', 2.0)   # m, proximity fallback

        # Lateral-strafe search — same pattern gate_navigator_v2_node's
        # SEARCH state already uses successfully for the gate: hold a
        # reference heading and strafe sideways ± sweep_dist, reversing at
        # each limit, until the front camera finds the target. No hardcoded
        # target coordinate, and no give-up condition (matches the gate's
        # own convention) — just keeps oscillating in that bounded lateral
        # corridor until found. Green search sweeps wider (mat could be
        # anywhere reachable from the gate exit); bin search is scoped
        # tighter since it starts already over the green zone.
        self.declare_parameter('search_strafe_speed',        0.25)   # m/s
        self.declare_parameter('green_search_sweep_dist',    6.0)    # m, ± each side
        self.declare_parameter('bin_search_sweep_dist',      3.0)    # m, ± each side
        self.declare_parameter('yaw_reset_threshold_rad',    0.05)   # rad
        # Forward "creeping line" component: if a full lateral pass finds
        # nothing, advance this far and sweep again, until found or the
        # total-advance budget is used up.
        self.declare_parameter('search_advance_step_m',      3.0)    # m per hop
        self.declare_parameter('green_search_max_advance_m', 15.0)   # total budget
        self.declare_parameter('bin_search_max_advance_m',    8.0)   # total budget
        # Blue bin YOLO detection is reliable (confidence ~0.7-0.9,
        # confirmed from live mission logs) even without ever finding the
        # green mat first, so if the green search exhausts its advance
        # budget, hand off straight to the bin search instead of stalling
        # forever waiting on a mat that a pure lateral/creeping search may
        # never reach.
        self.declare_parameter('skip_to_bin_if_green_not_found', True)

        # Blue bin approach/centering
        self.declare_parameter('rise_speed',              0.10)   # m/s upward
        self.declare_parameter('rise_ticks_required',     50)     # ~5 s = 0.5 m

        # Survey altitude before the blue-bin search: high enough for the
        # bottom camera to see across the green mat, well clear of the
        # water surface (WATER_SURFACE_Z=0.0, POOL_FLOOR_Z=-5.0 in the world
        # file — -2.0 leaves 2m of clearance to the surface).
        self.declare_parameter('bin_survey_z',            -2.0)   # m, odom frame
        self.declare_parameter('survey_z_tolerance',       0.15)  # m

        p = self.get_parameter
        self.p_gate_done_value        = p('gate_done_value').value
        self.p_search_spin_speed      = p('search_spin_speed').value
        self.p_approach_speed         = p('approach_speed').value
        self.p_center_speed           = p('center_speed').value
        self.p_approach_p_gain        = p('approach_p_gain').value
        self.p_center_p_gain          = p('center_p_gain').value
        self.p_angular_clamp          = p('angular_clamp').value
        self.p_phase2_coverage_thresh = p('phase2_coverage_thresh').value
        self.p_center_threshold_m     = p('center_threshold_m').value
        self.p_center_ticks_req       = p('center_ticks_required').value
        self.p_hold_ticks             = p('hold_ticks').value
        self.p_green_freshness_sec    = p('green_freshness_sec').value
        self.p_green_approach_stop    = p('green_approach_stop_dist').value
        self.p_search_strafe_speed    = p('search_strafe_speed').value
        self.p_green_search_sweep_dist = p('green_search_sweep_dist').value
        self.p_bin_search_sweep_dist  = p('bin_search_sweep_dist').value
        self.p_yaw_reset_threshold    = p('yaw_reset_threshold_rad').value
        self.p_search_advance_step_m  = p('search_advance_step_m').value
        self.p_green_search_max_advance_m = p('green_search_max_advance_m').value
        self.p_bin_search_max_advance_m   = p('bin_search_max_advance_m').value
        self.p_skip_to_bin_if_green_not_found = p('skip_to_bin_if_green_not_found').value
        self.p_rise_speed             = p('rise_speed').value
        self.p_rise_ticks_req         = p('rise_ticks_required').value
        self.p_bin_survey_z           = p('bin_survey_z').value
        self.p_survey_z_tolerance     = p('survey_z_tolerance').value
        # ------------------------------------------------------------------ #

        self.state      = 'IDLE'
        self.prev_state = 'IDLE'

        # --- Sensor data ---
        self.latest_green_pos       = None
        self.latest_green_time      = None
        self.latest_bottom_det      = None
        self.latest_blue_bin_pos    = None
        self.latest_blue_bin_time   = None
        self.latest_blue_bottom_det = None

        self.gate_mission_done = False
        self.last_gate_msg     = None

        self.current_x   = None
        self.current_y   = None
        self.current_z   = None
        self.current_yaw = None

        # Ticks / counters
        self.center_ticks          = 0
        self.rise_ticks            = 0
        self.hold_tick             = 0
        self.done_logged           = False
        self.hold_logged           = False
        self.search_failed_logged = False

        # World-frame odom coordinates
        self.green_odom_x    = None
        self.green_odom_y    = None
        self.blue_bin_odom_x = None
        self.blue_bin_odom_y = None

        # Heading reference published by gate_navigator_v2_node — the yaw
        # the AUV faced at mission start, before any turning ever happened.
        self.latest_forward_yaw = None
        self._reset_yaw_next    = None   # 'green' or 'bin' — which search
                                          # to enter once RESET_YAW finishes

        # Lateral-strafe search state, one per target class. Each is plain
        # bookkeeping local to _run_lateral_search(); 'ref_yaw' is None
        # until _start_lateral_search() latches the origin/heading at the
        # moment that search actually begins.
        self._lateral = {
            'green': self._new_lateral_state(),
            'bin':   self._new_lateral_state(),
        }

        # --- ROS interfaces ---
        self.create_subscription(Float32MultiArray, '/auv/green_position_3d',         self._green_pos_callback,    10)
        self.create_subscription(Float32MultiArray, '/auv/green_detection_bottom_2d', self._bottom_det_callback,   10)
        self.create_subscription(Float32MultiArray, '/auv/blue_bin_position_3d',      self._blue_bin_pos_callback, 10)
        self.create_subscription(Float32MultiArray, '/auv/blue_bin_detection_bottom_2d', self._blue_bottom_det_callback, 10)
        self.create_subscription(Odometry, '/auv/odom',          self._odom_callback,          10)
        self.create_subscription(String,   '/auv/mission_state', self._gate_mission_callback,   10)
        self.create_subscription(Float32,  '/auv/forward_reference_yaw', self._forward_yaw_callback, 10)

        self.cmd_pub   = self.create_publisher(Twist,            '/model/auv_box/cmd_vel',      10)
        self.state_pub = self.create_publisher(String,           '/auv/green_mission_state',    10)
        self.debug_pub = self.create_publisher(Float32MultiArray,'/auv/green_navigator_debug',  10)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info('Navigator started. Waiting for gate DONE...')

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _green_pos_callback(self, msg):
        self.latest_green_pos  = list(msg.data)
        self.latest_green_time = self.get_clock().now()
        if self.current_x is None or self.current_yaw is None: return
        if len(msg.data) < 5: return
        x_body, y_body, _, _, source = msg.data[:5]
        if source == 1.0:
            cos_y = math.cos(self.current_yaw)
            sin_y = math.sin(self.current_yaw)
            self.green_odom_x = self.current_x + x_body * cos_y - y_body * sin_y
            self.green_odom_y = self.current_y + x_body * sin_y + y_body * cos_y

    def _blue_bin_pos_callback(self, msg):
        self.latest_blue_bin_pos  = list(msg.data)
        self.latest_blue_bin_time = self.get_clock().now()
        if self.current_x is None or self.current_yaw is None: return
        if len(msg.data) < 5: return
        x_body, y_body, _, _, source = msg.data[:5]
        if source == 1.0:
            # FRONT camera: a one-shot depth estimate that goes stale once
            # the bin exits the FOV at close range (~1.1m) and can never
            # self-correct after that. Push the depth by +1.0m to counter
            # depth underestimation — a blind fudge, not a measured
            # correction, so it can just as easily overshoot (confirmed
            # live: the AUV flew straight over the bin without ever
            # registering "reached" because this pushed the target past
            # where the bottom camera then saw it directly underneath).
            # This is now only meant to get roughly close enough for the
            # bottom-camera path below (or the direct-sighting check in
            # _state_approach_blue_bin) to take over.
            x_body += 1.00

            cos_y = math.cos(self.current_yaw)
            sin_y = math.sin(self.current_yaw)
            self.blue_bin_odom_x = self.current_x + x_body * cos_y - y_body * sin_y
            self.blue_bin_odom_y = self.current_y + x_body * sin_y + y_body * cos_y
            self.get_logger().debug(
                f'[BB odom] updated from FRONT -> ({self.blue_bin_odom_x:.2f}, {self.blue_bin_odom_y:.2f})'
            )
        elif source == 2.0 and (self._bottom_on_green() or float(msg.data[3]) >= 0.65):
            # BOTTOM camera: a direct altitude-based measurement (no
            # extrapolation guess), and it keeps refreshing continuously
            # instead of freezing after one reading — this is the more
            # reliable of the two once the AUV is up at survey altitude and
            # over the mat. Gated on _bottom_on_green() OR high YOLO
            # confidence (>= 0.65) so the correction fires even when the
            # green mat is partially occluded by the bin barrel itself.
            cos_y = math.cos(self.current_yaw)
            sin_y = math.sin(self.current_yaw)
            self.blue_bin_odom_x = self.current_x + x_body * cos_y - y_body * sin_y
            self.blue_bin_odom_y = self.current_y + x_body * sin_y + y_body * cos_y
            self.get_logger().debug(
                f'[BB odom] updated from BOTTOM (conf={float(msg.data[3]):.2f}, on_green={self._bottom_on_green()}) '
                f'-> ({self.blue_bin_odom_x:.2f}, {self.blue_bin_odom_y:.2f})'
            )

    def _bottom_det_callback(self, msg):
        self.latest_bottom_det = list(msg.data)

    def _blue_bottom_det_callback(self, msg):
        self.latest_blue_bottom_det = list(msg.data)

    def _odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        siny  = 2.0 * (q.w * q.z + q.x * q.y)
        cosy  = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

    def _gate_mission_callback(self, msg):
        if msg.data != self.last_gate_msg:
            self.get_logger().info(f'[GREEN NAV] mission_state: "{msg.data}"')
            self.last_gate_msg = msg.data
        if msg.data == self.p_gate_done_value and not self.gate_mission_done:
            self.gate_mission_done = True
            self.get_logger().info('Gate DONE — starting post-gate navigation!')

    def _forward_yaw_callback(self, msg):
        self.latest_forward_yaw = msg.data

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    def _clamp(self, val, limit):
        return max(-limit, min(limit, val))

    def _is_green_fresh(self):
        if self.latest_green_time is None: return False
        return (self.get_clock().now() - self.latest_green_time).nanoseconds * 1e-9 < self.p_green_freshness_sec

    def _is_blue_bin_fresh(self):
        if self.latest_blue_bin_time is None: return False
        return (self.get_clock().now() - self.latest_blue_bin_time).nanoseconds * 1e-9 < self.p_green_freshness_sec

    def _bottom_on_green(self):
        """True if the bottom camera currently sees the green mat with
        meaningful coverage. Water is blue too, so any bottom-camera "blue
        bin" sighting needs this as a cross-check — without it, a YOLO
        false positive over open pool water could be mistaken for the bin.
        Requiring the mat underneath as well anchors every bottom-camera
        bin acceptance to "on the mat", which is the only place the bin
        actually is."""
        return (
            self.latest_bottom_det is not None
            and self.latest_bottom_det[0] > 0.5
            and self.latest_bottom_det[4] >= self.p_phase2_coverage_thresh
        )

    def _normalize_angle(self, angle):
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def _bearing_error(self, tx, ty):
        if tx is None or self.current_x is None: return None
        err = math.atan2(ty - self.current_y, tx - self.current_x) - self.current_yaw
        return self._normalize_angle(err)

    def _odom_distance(self, tx, ty):
        if tx is None or self.current_x is None: return None
        return math.hypot(tx - self.current_x, ty - self.current_y)

    # ------------------------------------------------------------------ #
    #  Lateral-strafe search (no hardcoded world coordinates)              #
    # ------------------------------------------------------------------ #

    # Stall detection: every _STALL_CHECK_TICKS, check whether commanded
    # motion actually produced real displacement. A pool wall (or any
    # other obstruction) stops the AUV well short of what a command
    # implies, so this catches "pushing uselessly into a wall" without
    # ever needing to know where that wall actually is.
    _STALL_CHECK_TICKS       = 20    # ~2s at 10 Hz
    _STALL_MIN_PROGRESS_FRAC = 0.3   # must cover >=30% of the theoretical
                                      # distance in that window, else blocked

    def _new_lateral_state(self):
        return {
            'phase': None,   # None -> lazy-init to 'SWEEP' on first call
            'origin_x': None, 'origin_y': None, 'ref_yaw': None, 'dir': -1,
            'reversals': 0, 'total_advance': 0.0,
            'advance_start_x': None, 'advance_start_y': None, 'advance_target': 0.0,
            'stall_x': None, 'stall_y': None, 'stall_tick': 0,
        }

    def _start_lateral_search(self, key):
        """(Re)start a sweep pass at the AUV's current position. Does NOT
        touch 'total_advance' — that persists across sweep restarts within
        one overall advance-and-sweep search; only _new_lateral_state()
        (called when a brand new search begins) resets it to 0."""
        st = self._lateral[key]
        st['phase']    = 'SWEEP'
        st['origin_x'] = self.current_x
        st['origin_y'] = self.current_y
        st['ref_yaw']  = self.current_yaw
        st['dir']      = -1   # start sweeping right, same convention gate_navigator_v2 uses
        st['reversals'] = 0
        st['stall_x']  = None
        st['stall_tick'] = 0

    def _tick_stalled(self, key, nominal_speed):
        """Returns True if the AUV hasn't moved roughly as far as commanded
        over the last _STALL_CHECK_TICKS — i.e. something is physically
        blocking it (typically a pool wall) and the current leg should be
        abandoned rather than pushed into forever."""
        st = self._lateral[key]
        st['stall_tick'] += 1
        if st['stall_x'] is None:
            st['stall_x'], st['stall_y'] = self.current_x, self.current_y
            return False
        if st['stall_tick'] < self._STALL_CHECK_TICKS:
            return False
        moved    = math.hypot(self.current_x - st['stall_x'], self.current_y - st['stall_y'])
        expected = nominal_speed * (self._STALL_CHECK_TICKS * 0.1)
        st['stall_tick'] = 0
        st['stall_x'], st['stall_y'] = self.current_x, self.current_y
        return moved < expected * self._STALL_MIN_PROGRESS_FRAC

    def _run_lateral_search(self, key, is_fresh_fn, get_odom_fn, sweep_dist, strafe_speed,
                             advance_step_m, max_advance_m):
        """
        Advance-and-sweep search: strafe sideways ± sweep_dist holding the
        reset-forward heading (same pattern gate_navigator_v2_node's
        SEARCH state already uses successfully for the gate). If a full
        pass (both directions) doesn't find anything, drive forward
        advance_step_m along that same heading and sweep again from there,
        repeating until found or max_advance_m total forward progress is
        used up. This is a "creeping line" search — the same principle a
        real search-and-rescue sweep uses — needed because a target well
        beyond the camera's range from the start point can never be found
        by lateral strafing alone, no matter how long you wait. Still no
        hardcoded target coordinate anywhere; stall detection (see
        _tick_stalled) keeps a wall from silently trapping the AUV if the
        gate crossing didn't leave it centered enough for the full sweep
        distance to fit.

        Returns 'FOUND', 'SEARCHING', or 'EXHAUSTED'.
        """
        if is_fresh_fn() and get_odom_fn()[0] is not None:
            return 'FOUND'

        st = self._lateral[key]
        if st['phase'] is None:
            self._start_lateral_search(key)

        yaw_err = self._normalize_angle(st['ref_yaw'] - self.current_yaw)

        if st['phase'] == 'ADVANCE':
            dist = math.hypot(self.current_x - st['advance_start_x'],
                               self.current_y - st['advance_start_y'])
            if dist >= st['advance_target'] or self._tick_stalled(key, strafe_speed):
                st['total_advance'] += st['advance_target']
                self._start_lateral_search(key)
                return 'SEARCHING'
            cmd = Twist()
            cmd.linear.x  = strafe_speed
            cmd.angular.z = self._clamp(self.p_approach_p_gain * yaw_err, self.p_angular_clamp)
            self.cmd_pub.publish(cmd)
            return 'SEARCHING'

        # phase == 'SWEEP'
        dx = self.current_x - st['origin_x']
        dy = self.current_y - st['origin_y']
        lateral_dist = -dx * math.sin(st['ref_yaw']) + dy * math.cos(st['ref_yaw'])

        hit_limit = (
            (lateral_dist >= sweep_dist and st['dir'] == 1)
            or (lateral_dist <= -sweep_dist and st['dir'] == -1)
        )
        if hit_limit or self._tick_stalled(key, strafe_speed):
            if not hit_limit:
                self.get_logger().warn(
                    f'[{key}] Search direction blocked (no progress) — reversing early.'
                )
            st['dir'] = -st['dir']
            st['reversals'] += 1
            # Always start a fresh stall-detection window after ANY
            # reversal (limit-triggered or stall-triggered) — otherwise a
            # checkpoint recorded mid-way through the old direction spans
            # the flip, and near-zero net displacement across that flip
            # can look like a false stall on the very next check even
            # though nothing is actually wrong.
            st['stall_x']    = None
            st['stall_tick'] = 0

        if st['reversals'] >= 2:   # covered both extremes once from centre
            if st['total_advance'] >= max_advance_m:
                return 'EXHAUSTED'
            st['phase']          = 'ADVANCE'
            st['advance_start_x'] = self.current_x
            st['advance_start_y'] = self.current_y
            st['advance_target']  = min(advance_step_m, max_advance_m - st['total_advance'])
            st['stall_x']         = None
            st['stall_tick']      = 0
            return 'SEARCHING'

        cmd = Twist()
        cmd.linear.y  = st['dir'] * strafe_speed
        cmd.angular.z = self._clamp(self.p_approach_p_gain * yaw_err, self.p_angular_clamp)
        self.cmd_pub.publish(cmd)
        return 'SEARCHING'

    # ------------------------------------------------------------------ #
    #  State handlers                                                      #
    # ------------------------------------------------------------------ #

    def _state_idle(self):

        if self.gate_mission_done:
            # Wipe any stale blue bin data from before the gate crossing
            self.latest_blue_bin_pos  = None
            self.latest_blue_bin_time = None
            self.blue_bin_odom_x      = None
            self.blue_bin_odom_y      = None
            self._lateral['green'] = self._new_lateral_state()
            self._lateral['bin']   = self._new_lateral_state()
            self._reset_yaw_next = 'green'
            self.get_logger().info('IDLE -> RESET_YAW (blue bin cache cleared)')
            return 'RESET_YAW'
        return 'IDLE'

    def _state_reset_yaw(self):
        """Rotate back to the forward-reference heading published by
        gate_navigator_v2_node (the yaw at mission start, before any
        turning ever happened) before starting a lateral-strafe search.
        Without this, the search would scan facing whatever heading the
        previous ALIGN/CROSS or approach left the AUV at, which can drift
        from dead-straight — this keeps every strafe search consistently
        oriented regardless of the arena layout."""
        next_state = 'SEARCH_GREEN' if self._reset_yaw_next == 'green' else 'SEARCH_BLUE_BIN'
        target_yaw = self.latest_forward_yaw

        if target_yaw is None:
            # No reference received yet (shouldn't normally happen this
            # late) — proceed without resetting rather than stalling.
            self._start_lateral_search(self._reset_yaw_next)
            return next_state

        yaw_err = self._normalize_angle(target_yaw - self.current_yaw)
        if abs(yaw_err) < self.p_yaw_reset_threshold:
            self._publish_zero()
            self._start_lateral_search(self._reset_yaw_next)
            self.get_logger().info(
                f'Yaw reset to forward heading ({math.degrees(target_yaw):.1f}deg) -> {next_state}'
            )
            return next_state

        cmd = Twist()
        cmd.angular.z = self._clamp(self.p_approach_p_gain * yaw_err, self.p_angular_clamp)
        self.cmd_pub.publish(cmd)
        return 'RESET_YAW'

    def _state_search_green(self):
        """Advance-and-sweep search for the green mat — see
        _run_lateral_search. No hardcoded world coordinate is used. If the
        advance budget is exhausted without finding the mat, hands off
        straight to the blue-bin search instead (its YOLO detection is
        reliable on its own — no need to gate the whole mission on ever
        visually confirming the mat first)."""
        result = self._run_lateral_search(
            'green', self._is_green_fresh,
            lambda: (self.green_odom_x, self.green_odom_y),
            self.p_green_search_sweep_dist, self.p_search_strafe_speed,
            self.p_search_advance_step_m, self.p_green_search_max_advance_m,
        )
        if result == 'FOUND':
            self.get_logger().info(
                f'Green mat detected at ({self.green_odom_x:.1f}, {self.green_odom_y:.1f}) '
                '-> APPROACH_GREEN'
            )
            return 'APPROACH_GREEN'
        if result == 'EXHAUSTED':
            if self.p_skip_to_bin_if_green_not_found:
                self.get_logger().warn(
                    f'SEARCH_GREEN exhausted its advance budget '
                    f'(~{self.p_green_search_max_advance_m:.0f}m) without finding the mat. '
                    'Skipping straight to the blue-bin search -> RISE_FOR_SURVEY'
                )
                self._lateral['bin'] = self._new_lateral_state()
                self._reset_yaw_next = 'bin'
                return 'RISE_FOR_SURVEY'
            if not self.search_failed_logged:
                self.get_logger().error(
                    f'SEARCH_GREEN exhausted its advance budget '
                    f'(~{self.p_green_search_max_advance_m:.0f}m) without finding the mat. Holding.'
                )
                self.search_failed_logged = True
            self._publish_zero()
            return 'SEARCH_GREEN'
        return 'SEARCH_GREEN'

    def _state_approach_green(self):
        """Drive toward the green mat odom coordinate."""
        bottom_coverage = 0.0
        if self.latest_bottom_det is not None and self.latest_bottom_det[0] > 0.5:
            bottom_coverage = self.latest_bottom_det[4]

        green_source = 0.0
        if self.latest_green_pos is not None and len(self.latest_green_pos) > 4:
            green_source = self.latest_green_pos[4]

        # Proximity fallback: if close to the target, start blue bin search
        dist_to_green = self._odom_distance(self.green_odom_x, self.green_odom_y)

        in_green_zone = (
            bottom_coverage >= self.p_phase2_coverage_thresh
            or green_source == 2.0
            or (dist_to_green is not None and dist_to_green < self.p_green_approach_stop)
        )

        if in_green_zone:
            reason = (
                f'coverage={bottom_coverage:.3f}' if bottom_coverage >= self.p_phase2_coverage_thresh
                else f'proximity={dist_to_green:.2f}m' if dist_to_green is not None and dist_to_green < self.p_green_approach_stop
                else 'bottom_cam_green'
            )
            self.get_logger().info(f'In green zone ({reason}) -> RISE_FOR_SURVEY')
            self._lateral['bin'] = self._new_lateral_state()
            self._reset_yaw_next = 'bin'
            return 'RISE_FOR_SURVEY'

        bearing_err = self._bearing_error(self.green_odom_x, self.green_odom_y)
        if bearing_err is None:
            cmd = Twist()
            cmd.angular.z = self.p_search_spin_speed
            self.cmd_pub.publish(cmd)
            return 'APPROACH_GREEN'

        cmd = Twist()
        if abs(bearing_err) > 0.5:
            cmd.linear.x = 0.0  # Turn in place if not facing target
        else:
            cmd.linear.x = self.p_approach_speed
            
        cmd.angular.z = self._clamp(self.p_approach_p_gain * bearing_err, self.p_angular_clamp)
        self.cmd_pub.publish(cmd)
        return 'APPROACH_GREEN'

    def _state_rise_for_survey(self):
        """Climb to bin_survey_z before starting the blue-bin search, so the
        bottom camera has a wide view across the green mat instead of the
        AUV relying on the front camera's narrow FOV/limited range to
        acquire the bin (see the FSM docstring for why that's unreliable).
        Bounded well clear of the water surface by bin_survey_z itself
        (declared negative, well below WATER_SURFACE_Z=0.0)."""
        if self.current_z is None:
            self._publish_zero()
            return 'RISE_FOR_SURVEY'

        remaining = self.p_bin_survey_z - self.current_z
        if remaining <= self.p_survey_z_tolerance:
            self._publish_zero()
            self.get_logger().info(
                f'At survey altitude (z={self.current_z:.2f}m) -> RESET_YAW (bin search)'
            )
            return 'RESET_YAW'

        cmd = Twist()
        cmd.linear.z = self.p_rise_speed
        self.cmd_pub.publish(cmd)
        return 'RISE_FOR_SURVEY'

    def _state_search_blue_bin(self):
        """Advance-and-sweep search for the blue bin — see
        _run_lateral_search. Scoped tighter than the green mat search since
        this starts already over (or near) the green zone. No hardcoded
        world coordinate is used."""
        result = self._run_lateral_search(
            'bin', self._is_blue_bin_fresh,
            lambda: (self.blue_bin_odom_x, self.blue_bin_odom_y),
            self.p_bin_search_sweep_dist, self.p_search_strafe_speed,
            self.p_search_advance_step_m, self.p_bin_search_max_advance_m,
        )
        if result == 'FOUND':
            self.get_logger().info(
                f'★ BLUE BIN confirmed by YOLO at '
                f'({self.blue_bin_odom_x:.1f}, {self.blue_bin_odom_y:.1f}) '
                '-> APPROACH_BLUE_BIN'
            )
            return 'APPROACH_BLUE_BIN'
        if result == 'EXHAUSTED':
            if not self.search_failed_logged:
                self.get_logger().error(
                    f'SEARCH_BLUE_BIN exhausted its advance budget '
                    f'(~{self.p_bin_search_max_advance_m:.0f}m) without finding the bin. Holding.'
                )
                self.search_failed_logged = True
            self._publish_zero()
            return 'SEARCH_BLUE_BIN'
        return 'SEARCH_BLUE_BIN'

    def _state_approach_blue_bin(self):
        """Drive to stored front-camera odom coordinate of the blue bin.
        Speed is tapered as the AUV gets closer to avoid overshoot and jerking."""
        bearing_err = self._bearing_error(self.blue_bin_odom_x, self.blue_bin_odom_y)
        if bearing_err is None:
            cmd = Twist()
            cmd.angular.z = self.p_search_spin_speed
            self.cmd_pub.publish(cmd)
            return 'APPROACH_BLUE_BIN'

        dist = self._odom_distance(self.blue_bin_odom_x, self.blue_bin_odom_y)

        # Close enough — rise before centering/holding instead of stopping
        # right on top of the bin. This state used to jump straight to HOLD,
        # which held the AUV at the same depth it used to track the bin with
        # the front camera — close enough to collide with it. Rising first
        # (RISE_FOR_VERIFICATION already existed for exactly this) clears the
        # bin before CENTER_BLUE_BIN/HOLD, and also gives the bottom camera
        # the wider view it needs.
        if dist is not None and dist < self.p_center_threshold_m:
            self.get_logger().info(
                f'Reached blue bin coord (dist={dist:.2f}m) -> RISE_FOR_VERIFICATION'
            )
            self.rise_ticks = 0
            return 'RISE_FOR_VERIFICATION'

        # Bottom camera sees the bin right now -> we are physically over it,
        # regardless of what the odom target says. The odom target comes
        # from a single frozen front-camera reading (plus a fixed +1.0m
        # depth fudge) that can go stale for a minute or more once the bin
        # exits the front camera's FOV at close range; confirmed from a live
        # run where that target ended up ~1m past the real bin, so `dist`
        # never dropped below threshold and the AUV flew straight over the
        # bin without ever triggering RISE_FOR_VERIFICATION. Also require
        # _bottom_on_green(): water is blue too, so a bare YOLO "blue bin"
        # hit on the bottom camera isn't enough on its own — only trust it
        # when the green mat is confirmed underneath as well.
        _bbd = self.latest_blue_bottom_det
        _bin_conf = float(_bbd[5]) if _bbd is not None else 0.0
        if (_bbd is not None
                and _bbd[0] > 0.5
                and (self._bottom_on_green() or _bin_conf >= 0.65)):
            self.get_logger().info(
                f'Bottom camera sees blue bin (conf={_bin_conf:.2f}, '
                f'on_green={self._bottom_on_green()}) -> RISE_FOR_VERIFICATION'
            )
            self.rise_ticks = 0
            return 'RISE_FOR_VERIFICATION'

        # Taper speed: full speed until 2m, then ramp down to center_speed by 0.35m
        if dist is not None and dist < 2.0:
            t = (dist - self.p_center_threshold_m) / (2.0 - self.p_center_threshold_m)
            t = max(0.0, min(1.0, t))
            speed = self.p_center_speed + t * (self.p_approach_speed - self.p_center_speed)
        else:
            speed = self.p_approach_speed

        cmd = Twist()
        if abs(bearing_err) > 0.5:
            cmd.linear.x = 0.0  # Turn in place if not facing target
        else:
            cmd.linear.x = speed
            
        cmd.angular.z = self._clamp(self.p_approach_p_gain * bearing_err, self.p_angular_clamp)
        self.cmd_pub.publish(cmd)
        return 'APPROACH_BLUE_BIN'

    def _state_rise_for_verification(self):
        """Rise ~0.5 m so the bottom camera gets a wider view of the bin.
        Skipped if already at/above bin_survey_z — that wide view already
        exists from RISE_FOR_SURVEY, so climbing further would only creep
        closer to the water surface for no benefit."""
        if self.current_z is not None and self.current_z >= self.p_bin_survey_z:
            self.get_logger().info(
                f'Already at survey altitude (z={self.current_z:.2f}m) -> CENTER_BLUE_BIN (bottom cam)'
            )
            self._publish_zero()
            self.center_ticks = 0
            return 'CENTER_BLUE_BIN'

        self.rise_ticks += 1
        cmd = Twist()
        cmd.linear.z = self.p_rise_speed
        self.cmd_pub.publish(cmd)

        if self.rise_ticks >= self.p_rise_ticks_req:
            self.get_logger().info(
                f'Risen {self.rise_ticks * self.p_rise_speed * 0.1:.2f}m. '
                '-> CENTER_BLUE_BIN (bottom cam)'
            )
            self.center_ticks = 0
            return 'CENTER_BLUE_BIN'
        return 'RISE_FOR_VERIFICATION'

    def _state_center_blue_bin(self):
        """Fine-center using the bottom camera, requiring the bin to be
        centered AND the surrounding background to actually be the green
        mat (not just a well-centered bin reading over the wrong spot).
        Falls back to odom if the bottom cam loses the bin entirely."""
        # --- Priority 1: bottom camera detection ---
        if (self._is_blue_bin_fresh()
                and self.latest_blue_bin_pos is not None
                and self.latest_blue_bin_pos[4] == 2.0):
            x_fwd  = self.latest_blue_bin_pos[0]
            y_left = self.latest_blue_bin_pos[1]

            # Use body-frame strafe commands to centre:
            #   x_fwd  → cmd.linear.x  (forward/back)
            #   y_left → cmd.linear.y  (sideways strafe — NOT angular.z/yaw)
            # Previously y_left was sent to cmd.angular.z, which made the AUV
            # rotate instead of translating sideways — the bin stayed off-centre
            # no matter how long CENTER_BLUE_BIN ran.
            cmd = Twist()
            cmd.linear.x = self._clamp(self.p_center_p_gain * x_fwd,  self.p_center_speed)
            cmd.linear.y = self._clamp(self.p_center_p_gain * y_left,  self.p_center_speed)
            self.cmd_pub.publish(cmd)

            error = math.hypot(x_fwd, y_left)
            on_green = self._bottom_on_green()
            # Leaky counter instead of a hard reset: the bottom cam loses
            # the bin for a tick or two fairly often (motion blur, edge of
            # frame while correcting), and a single bad tick shouldn't
            # erase a near-complete streak — that was causing CENTER_BLUE_BIN
            # to orbit indefinitely without ever reaching HOLD (confirmed in
            # mission logs: 2+ minutes oscillating within ~0.15m of the bin,
            # log ending mid-CENTER_BLUE_BIN with center_ticks never
            # reaching the required streak).
            #
            # Relaxed on_green requirement: when the AUV is nearly directly
            # above the bin (error < 0.15 m), the blue drum barrel may occlude
            # most of the surrounding green mat, temporarily dropping green
            # coverage below the threshold. At that distance the bottom-cam
            # bin detection is reliable enough on its own — we accept it
            # without requiring the green background confirmation.
            very_close = error < self.p_center_threshold_m   # bin occludes mat at any sub-threshold dist
            centered_ok = error < self.p_center_threshold_m and (on_green or very_close)
            self.center_ticks = (
                self.center_ticks + 1 if centered_ok
                else max(0, self.center_ticks - 1)   # drain 1 per bad tick, not 2
            )

            if self.center_ticks >= self.p_center_ticks_req:
                self.get_logger().info(
                    f'Centred over BLUE BIN! error={error:.3f}m on_green={on_green} -> HOLD'
                )
                return 'HOLD'
            return 'CENTER_BLUE_BIN'

        # --- Priority 2: odometry fallback ---
        if self.blue_bin_odom_x is not None:
            dist = self._odom_distance(self.blue_bin_odom_x, self.blue_bin_odom_y)
            if dist is not None and dist < self.p_center_threshold_m:
                self.get_logger().info(f'Odom: over BLUE BIN (dist={dist:.3f}m) -> HOLD')
                return 'HOLD'

            bearing_err = self._bearing_error(self.blue_bin_odom_x, self.blue_bin_odom_y)
            if bearing_err is not None:
                cmd = Twist()
                if abs(bearing_err) > 0.5:
                    cmd.linear.x = 0.0  # Turn in place if not facing target
                else:
                    cmd.linear.x = self.p_center_speed
                    
                cmd.angular.z = self._clamp(self.p_approach_p_gain * bearing_err, self.p_angular_clamp)
                self.cmd_pub.publish(cmd)
                return 'CENTER_BLUE_BIN'

        self._publish_zero()
        return 'CENTER_BLUE_BIN'

    def _state_hold(self):
        self._publish_zero()
        if not self.hold_logged:
            self.get_logger().info('★ HOLDING OVER TARGET ★')
            self.hold_logged = True
        self.hold_tick += 1
        return 'FINAL_DONE' if self.hold_tick >= self.p_hold_ticks else 'HOLD'

    def _state_final_done(self):
        self._publish_zero()
        if not self.done_logged:
            self.get_logger().info('★★★ MISSION COMPLETE ★★★')
            self.done_logged = True
        return 'FINAL_DONE'

    # ------------------------------------------------------------------ #
    #  Main control loop                                                   #
    # ------------------------------------------------------------------ #

    def _control_loop(self):
        if self.current_x is None:
            return

        dispatch = {
            'IDLE':                  self._state_idle,
            'RESET_YAW':             self._state_reset_yaw,
            'SEARCH_GREEN':          self._state_search_green,
            'APPROACH_GREEN':        self._state_approach_green,
            'RISE_FOR_SURVEY':       self._state_rise_for_survey,
            'SEARCH_BLUE_BIN':       self._state_search_blue_bin,
            'APPROACH_BLUE_BIN':     self._state_approach_blue_bin,
            'RISE_FOR_VERIFICATION': self._state_rise_for_verification,
            'CENTER_BLUE_BIN':       self._state_center_blue_bin,
            'HOLD':                  self._state_hold,
            'FINAL_DONE':            self._state_final_done,
        }
        handler = dispatch.get(self.state)
        if handler is None:
            self.get_logger().error(f'Unknown state: {self.state}')
            return

        next_state = handler()

        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

        self.debug_pub.publish(Float32MultiArray())

        if next_state != self.state:
            self.get_logger().info(f'[FSM] {self.state} -> {next_state}')
        self.state = next_state


def main(args=None):
    rclpy.init(args=args)
    node = GreenNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
