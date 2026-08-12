#!/usr/bin/env python3
"""
mission_logger_node.py  —  Rich debug CSV logger for the AUV gate mission.

CSV columns
-----------
wall_time            : ISO-8601 wall-clock timestamp
sim_time_sec         : seconds since node started
mission_state        : SEARCH / TRACK / ALIGN / CROSS / STOP

odom_x               : AUV X in odom frame (m)
odom_y               : AUV Y in odom frame (m)
odom_yaw_deg         : AUV heading (degrees)
odom_vel_x           : actual fwd velocity from odometry twist (m/s)
odom_vel_yaw         : actual yaw rate from odometry twist (rad/s)
dist_from_start      : total distance travelled since mission start (m)

gate_2d_detected     : 1 if YOLO found gate, else 0
gate_2d_px_x/y       : gate bounding-box centre (pixels)
gate_2d_w_px/h_px    : bounding-box size (pixels)
gate_2d_conf         : YOLO confidence

gate_3d_x/y/z        : gate in body frame (m) — fwd/left/up
gate_3d_age_sec      : seconds since last 3D localizer update

nav_gate_odom_x/y    : navigator's stored gate position in odom frame (m)
nav_approach_yaw_deg : locked CROSS heading (deg, nan until committed)
nav_distance_m       : distance navigator is working against this tick
nav_bearing_err_deg  : bearing/yaw error seen by navigator this tick
nav_commit_ticks     : consecutive ticks inside commit zone
nav_align_ticks      : consecutive ticks holding alignment
nav_yolo_fresh       : 1 if YOLO age < 1.5 s, else 0
nav_yolo_age_sec     : seconds since last YOLO detection (-1 if none)
nav_source           : 0=none  1=yolo-live  2=odom-deadreckon

cmd_linear_x         : commanded fwd velocity (m/s)
cmd_angular_z        : commanded yaw rate (rad/s)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import os
import csv
from datetime import datetime


class MissionLoggerNode(Node):
    def __init__(self):
        super().__init__('mission_logger_node')

        # ── Cache variables ──────────────────────────────────────────────
        self.latest_state = None
        self.prev_state   = None   # used to detect state transitions

        self.odom_x       = None
        self.odom_y       = None
        self.odom_yaw     = None
        self.odom_vel_x   = None
        self.odom_vel_yaw = None
        self.start_x      = None
        self.start_y      = None

        self.gate_2d_array = None   # [detected, cx, cy, w, h, conf, _]

        self.gate_3d_x    = None
        self.gate_3d_y    = None
        self.gate_3d_z    = None
        self.gate_3d_time = None

        # [gate_odom_x, gate_odom_y, approach_yaw_deg,
        #  distance_m, bearing_err_deg,
        #  commit_ticks, align_ticks,
        #  yolo_fresh, yolo_age_sec, nav_source]
        self.nav_debug = None

        self.cmd_linear_x  = None
        self.cmd_angular_z = None

        # ── CSV file ─────────────────────────────────────────────────────
        log_dir = os.path.join(os.path.expanduser('~'), 'auv_ws', 'mission_logs')
        os.makedirs(log_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_filepath = os.path.join(log_dir, f'mission_run_{ts}.csv')

        self.csv_file   = open(self.log_filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            # time & state
            'wall_time', 'sim_time_sec', 'mission_state',
            # odometry
            'odom_x', 'odom_y', 'odom_yaw_deg',
            'odom_vel_x', 'odom_vel_yaw',
            'dist_from_start',
            # 2-D YOLO
            'gate_2d_detected', 'gate_2d_px_x', 'gate_2d_px_y',
            'gate_2d_w_px', 'gate_2d_h_px', 'gate_2d_conf',
            # 3-D localizer
            'gate_3d_x', 'gate_3d_y', 'gate_3d_z', 'gate_3d_age_sec',
            # navigator internals
            'nav_gate_odom_x', 'nav_gate_odom_y', 'nav_approach_yaw_deg',
            'nav_distance_m', 'nav_bearing_err_deg',
            'nav_commit_ticks', 'nav_align_ticks',
            'nav_yolo_fresh', 'nav_yolo_age_sec', 'nav_source',
            # commands
            'cmd_linear_x', 'cmd_angular_z',
        ])

        self.get_logger().info(f'Mission Logger started. Logging to: {self.log_filepath}')
        self.node_start_time = self.get_clock().now()

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(String,           '/auv/mission_state',      self.state_callback,     10)
        self.create_subscription(Odometry,         '/auv/odom',               self.odom_callback,      10)
        self.create_subscription(Float32MultiArray,'/auv/gate_detection_2d',  self.gate_2d_callback,   10)
        self.create_subscription(Float32MultiArray,'/auv/gate_position_3d',   self.gate_3d_callback,   10)
        self.create_subscription(Float32MultiArray,'/auv/navigator_debug',    self.nav_debug_callback, 10)
        self.create_subscription(Twist,            '/model/auv_box/cmd_vel',  self.cmd_callback,       10)

        self.create_timer(1.0, self.log_tick)   # 1 Hz — ~60-120 rows per run

    # ── Callbacks ────────────────────────────────────────────────────────

    def state_callback(self, msg):
        self.latest_state = msg.data
        # Force an immediate log row on every state transition
        # so the exact timestamp of SEARCH→TRACK→ALIGN→CROSS→STOP is captured.
        if msg.data != self.prev_state:
            self.prev_state = msg.data
            self.log_tick()

    def odom_callback(self, msg):
        self.odom_x       = msg.pose.pose.position.x
        self.odom_y       = msg.pose.pose.position.y
        self.odom_vel_x   = msg.twist.twist.linear.x
        self.odom_vel_yaw = msg.twist.twist.angular.z
        q = msg.pose.pose.orientation
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.start_x is None:
            self.start_x = self.odom_x
            self.start_y = self.odom_y

    def gate_2d_callback(self, msg):
        self.gate_2d_array = msg.data

    def gate_3d_callback(self, msg):
        if len(msg.data) >= 3:
            self.gate_3d_x    = msg.data[0]
            self.gate_3d_y    = msg.data[1]
            self.gate_3d_z    = msg.data[2]
            self.gate_3d_time = self.get_clock().now()

    def nav_debug_callback(self, msg):
        if len(msg.data) >= 10:
            self.nav_debug = msg.data

    def cmd_callback(self, msg):
        self.cmd_linear_x  = msg.linear.x
        self.cmd_angular_z = msg.angular.z

    # ── Helper ───────────────────────────────────────────────────────────

    def f(self, val, dec=4):
        """Format a float to dec decimal places, or '' for None/NaN."""
        if val is None:
            return ''
        try:
            if math.isnan(float(val)):
                return 'nan'
        except (TypeError, ValueError):
            return str(val)
        return f'{float(val):.{dec}f}'

    # ── 10 Hz logging ────────────────────────────────────────────────────

    def log_tick(self):
        now = self.get_clock().now()

        wall_time    = datetime.now().isoformat()
        sim_time_sec = (now - self.node_start_time).nanoseconds / 1e9
        mission_state = self.latest_state or 'UNKNOWN'

        # odometry
        dist_from_start = None
        if self.start_x is not None and self.odom_x is not None:
            dist_from_start = math.sqrt(
                (self.odom_x - self.start_x)**2 +
                (self.odom_y - self.start_y)**2)

        odom_yaw_deg = math.degrees(self.odom_yaw) if self.odom_yaw is not None else None

        # 2-D detection
        d2_det = d2_px_x = d2_px_y = d2_w = d2_h = d2_conf = None
        if self.gate_2d_array is not None and len(self.gate_2d_array) >= 6:
            d2_det  = self.gate_2d_array[0]
            d2_px_x = self.gate_2d_array[1]
            d2_px_y = self.gate_2d_array[2]
            d2_w    = self.gate_2d_array[3]
            d2_h    = self.gate_2d_array[4]
            d2_conf = self.gate_2d_array[5]

        # 3-D localizer
        g3_x = g3_y = g3_z = g3_age = None
        if self.gate_3d_x is not None and self.gate_3d_time is not None:
            g3_x   = self.gate_3d_x
            g3_y   = self.gate_3d_y
            g3_z   = self.gate_3d_z
            g3_age = (now - self.gate_3d_time).nanoseconds / 1e9

        # navigator debug
        nav_odom_x = nav_odom_y = nav_yaw_deg = None
        nav_dist   = nav_bear   = nav_commit   = nav_align  = None
        nav_fresh  = nav_age    = nav_src      = None
        if self.nav_debug is not None and len(self.nav_debug) >= 10:
            nd         = self.nav_debug
            nav_odom_x = nd[0]
            nav_odom_y = nd[1]
            nav_yaw_deg= nd[2]
            nav_dist   = nd[3]
            nav_bear   = nd[4]
            nav_commit = nd[5]
            nav_align  = nd[6]
            nav_fresh  = nd[7]
            nav_age    = nd[8]
            nav_src    = nd[9]

        row = [
            wall_time,          self.f(sim_time_sec, 2), mission_state,
            # odometry
            self.f(self.odom_x),self.f(self.odom_y), self.f(odom_yaw_deg, 2),
            self.f(self.odom_vel_x, 4), self.f(self.odom_vel_yaw, 4),
            self.f(dist_from_start),
            # 2-D detection
            self.f(d2_det, 0), self.f(d2_px_x, 1), self.f(d2_px_y, 1),
            self.f(d2_w, 1),   self.f(d2_h, 1),    self.f(d2_conf, 3),
            # 3-D localizer
            self.f(g3_x), self.f(g3_y), self.f(g3_z), self.f(g3_age, 2),
            # navigator debug
            self.f(nav_odom_x), self.f(nav_odom_y), self.f(nav_yaw_deg, 2),
            self.f(nav_dist),   self.f(nav_bear, 2),
            self.f(nav_commit, 0), self.f(nav_align, 0),
            self.f(nav_fresh, 0),  self.f(nav_age, 2), self.f(nav_src, 0),
            # commands
            self.f(self.cmd_linear_x, 3), self.f(self.cmd_angular_z, 4),
        ]

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    # ── Shutdown ─────────────────────────────────────────────────────────

    def destroy_node(self):
        if not self.csv_file.closed:
            self.csv_file.close()
        self.get_logger().info(f'Shutdown — log saved: {self.log_filepath}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
