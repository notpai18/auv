#!/usr/bin/env python3
"""
mission_monitor_node.py  —  Premium live terminal dashboard for AUV missions.

Shows a beautiful, live-updating ANSI interface for:
- Telemetry (Odom, Cmd Vel)
- Gate Phase (Gate detection, FSM state)
- Target Zone Phase (Green Mat & Blue Bin detection, FSM state)

Usage:
  ros2 run auv_telemetry mission_monitor_node
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import time
from datetime import datetime

# ── ANSI colour helpers ────────────────────────────────────────────────────
RESET   = '\033[0m'
BOLD    = '\033[1m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
RED     = '\033[91m'
CYAN    = '\033[96m'
MAGENTA = '\033[95m'
BLUE    = '\033[94m'
DIM     = '\033[2m'
CLEAR   = '\033[2J\033[H'

# ── Human-readable state descriptions ─────────────────────────────────────
GATE_STATES = {
    'SEARCH':         'Spinning in place — scanning for gate',
    'TRACK':          'Gate found! Driving toward gate',
    'ALIGN':          'Aligning heading with gate normal',
    'CROSS':          'Crossing the gate!',
    'STOP':           'Crossed! Pausing before return run',
    'RETURN_ALIGN':   'U-turning to face gate again',
    'RETURN_TRACK':   'Driving back toward gate',
    'RETURN_ALIGN2':  'Final alignment for return crossing',
    'RETURN_CROSS':   'Crossing gate on return leg!',
    'DONE':           'ROUND TRIP COMPLETE',
}

GREEN_STATES = {
    'IDLE':               'Waiting for gate phase to finish...',
    'SEARCH_GREEN':       'Searching for green mat (turning)',
    'APPROACH_GREEN':     'Driving to green mat center',
    'SEARCH_BLUE_BIN':    'Scanning green mat for blue bin',
    'APPROACH_BLUE_BIN':  'Approaching blue bin',
    'HOLD':               'Holding position directly over target',
    'FINAL_DONE':         'MISSION COMPLETE',
}

W = 75   # panel width

class MissionMonitorNode(Node):
    def __init__(self):
        super().__init__('mission_monitor_node')

        self._start_wall = time.time()

        # Telemetry
        self._odom_x = self._odom_y = self._odom_z = self._odom_yaw = 0.0
        self._cmd_lin = self._cmd_ang = 0.0
        self._ready_odom = False
        self._ready_cmd = False

        # Gate Phase
        self._gate_state = 'INACTIVE'
        self._gate_2d = []
        self._gate_3d = []

        # Green Phase
        self._green_state = 'INACTIVE'
        self._green_pos = []
        self._blue_bin_pos = []

        # Subscriptions
        self.create_subscription(Odometry, '/auv/odom', self._odom_cb, 10)
        self.create_subscription(Twist, '/model/auv_box/cmd_vel', self._cmd_cb, 10)

        self.create_subscription(String, '/auv/mission_state', self._gate_state_cb, 10)
        self.create_subscription(Float32MultiArray, '/auv/gate_detection_2d', self._gate2d_cb, 10)
        self.create_subscription(Float32MultiArray, '/auv/gate_position_3d', self._gate3d_cb, 10)

        self.create_subscription(String, '/auv/green_mission_state', self._green_state_cb, 10)
        self.create_subscription(Float32MultiArray, '/auv/green_position_3d', self._green_pos_cb, 10)
        self.create_subscription(Float32MultiArray, '/auv/blue_bin_position_3d', self._blue_bin_pos_cb, 10)

        self.create_timer(1.0, self._draw)
        self.get_logger().info('Premium Mission Monitor active.')

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _odom_cb(self, msg):
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        self._odom_z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        self._odom_yaw = math.degrees(math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        self._ready_odom = True

    def _cmd_cb(self, msg):
        self._cmd_lin = msg.linear.x
        self._cmd_ang = msg.angular.z
        self._ready_cmd = True

    def _gate_state_cb(self, msg): self._gate_state = msg.data
    def _gate2d_cb(self, msg): self._gate_2d = msg.data
    def _gate3d_cb(self, msg): self._gate_3d = msg.data

    def _green_state_cb(self, msg): self._green_state = msg.data
    def _green_pos_cb(self, msg): self._green_pos = msg.data
    def _blue_bin_pos_cb(self, msg): self._blue_bin_pos = msg.data

    # ── Drawing helpers ────────────────────────────────────────────────────
    def _line(self, text='', color='', align='left'):
        inner = W - 4
        if align == 'center':
            content = text.center(inner)
        else:
            content = text.ljust(inner)
        visible_len = len(text)
        pad = inner - visible_len
        if align == 'left':
            content = text + ' ' * max(0, pad)
        return f'║ {color}{content}{RESET} ║'

    def _sep(self, char='─'): return '╠' + char * (W - 2) + '╣'
    def _top(self): return '╔' + '═' * (W - 2) + '╗'
    def _bot(self): return '╚' + '═' * (W - 2) + '╝'

    # ── Main draw ──────────────────────────────────────────────────────────
    def _draw(self):
        elapsed  = time.time() - self._start_wall
        runtime  = f'{int(elapsed)//60:02d}:{int(elapsed)%60:02d}'
        now_str  = datetime.now().strftime('%H:%M:%S')

        lines = [self._top()]
        lines.append(self._line(f'🚀 AUV MISSION DASHBOARD        {DIM}{now_str}  |  Runtime: {runtime}{RESET}', BOLD, 'left'))
        lines.append(self._sep())

        # ── TELEMETRY ──────────────────────────────────────────────────
        lines.append(self._line(f'{BOLD}📡 TELEMETRY{RESET}'))
        if self._ready_odom:
            pos_str = f'X: {self._odom_x:+.2f}m   Y: {self._odom_y:+.2f}m   Z: {self._odom_z:+.2f}m   Yaw: {self._odom_yaw:+.1f}°'
            lines.append(self._line(f'  {DIM}Pos:{RESET} {CYAN}{pos_str}{RESET}'))
        else:
            lines.append(self._line(f'  {DIM}Pos: waiting for /auv/odom...{RESET}'))

        if self._ready_cmd:
            cmd_str = f'Fwd: {self._cmd_lin:+.3f} m/s   Rot: {self._cmd_ang:+.3f} rad/s'
            lines.append(self._line(f'  {DIM}Cmd:{RESET} {YELLOW}{cmd_str}{RESET}'))
        else:
            lines.append(self._line(f'  {DIM}Cmd: waiting for commands...{RESET}'))
        lines.append(self._sep())

        # ── GATE PHASE ─────────────────────────────────────────────────
        gate_active = self._gate_state not in ['INACTIVE', 'DONE']
        gc = GREEN if gate_active else DIM
        lines.append(self._line(f'{gc}{BOLD}🔲 GATE PHASE{RESET}'))
        
        state_str = GATE_STATES.get(self._gate_state, self._gate_state)
        lines.append(self._line(f'  {DIM}State:{RESET} {BOLD}{self._gate_state:<13}{RESET} {DIM}{state_str}{RESET}'))

        if self._gate_2d and len(self._gate_2d) >= 6:
            det = self._gate_2d[0] == 1.0
            conf = self._gate_2d[5]
            dist = self._gate_3d[0] if self._gate_3d else 0.0
            if det:
                det_str = f'{GREEN}DETECTED{RESET} (conf: {conf:.2f}, dist: {dist:.1f}m)'
            else:
                det_str = f'{RED}SEARCHING{RESET}'
        else:
            det_str = f'{DIM}No camera data{RESET}'
        lines.append(self._line(f'  {DIM}Gate YOLO:{RESET} {det_str}'))
        lines.append(self._sep())

        # ── TARGET ZONE PHASE ──────────────────────────────────────────
        green_active = self._green_state not in ['INACTIVE', 'FINAL_DONE']
        grc = GREEN if green_active else DIM
        lines.append(self._line(f'{grc}{BOLD}🎯 TARGET ZONE PHASE (GREEN MAT & BLUE BIN){RESET}'))
        
        gstate_str = GREEN_STATES.get(self._green_state, self._green_state)
        lines.append(self._line(f'  {DIM}State:{RESET} {BOLD}{self._green_state:<17}{RESET} {DIM}{gstate_str}{RESET}'))

        if self._green_pos and len(self._green_pos) >= 4:
            dist = math.sqrt(self._green_pos[0]**2 + self._green_pos[1]**2)
            g_str = f'{GREEN}DETECTED{RESET} (dist: {dist:.1f}m)'
        else:
            g_str = f'{RED}SEARCHING{RESET}'
        lines.append(self._line(f'  {DIM}Green Mat Tracker:{RESET} {g_str}'))

        if self._blue_bin_pos and len(self._blue_bin_pos) >= 5:
            conf = self._blue_bin_pos[3]
            dist = math.sqrt(self._blue_bin_pos[0]**2 + self._blue_bin_pos[1]**2)
            src = 'Front' if self._blue_bin_pos[4] == 1.0 else 'Bottom'
            b_str = f'{BLUE}DETECTED{RESET} (dist: {dist:.1f}m, conf: {conf:.2f}, src: {src})'
        else:
            b_str = f'{RED}SEARCHING{RESET}'
        lines.append(self._line(f'  {DIM}Blue Bin YOLO:{RESET}     {b_str}'))

        lines.append(self._line())
        lines.append(self._bot())

        print(CLEAR + '\n'.join(lines), flush=True)

def main(args=None):
    rclpy.init(args=args)
    node = MissionMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
