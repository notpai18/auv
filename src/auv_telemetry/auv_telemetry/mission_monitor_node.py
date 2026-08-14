#!/usr/bin/env python3
"""
mission_monitor_node.py  —  Live terminal dashboard for the AUV gate mission.

Prints a self-refreshing panel (1 Hz) that shows:
  Phase 1 — STARTUP : which nodes have come up (tracked by topic activity)
  Phase 2 — MISSION : live state, position, gate status, commands, runtime

Usage
-----
  ros2 run auv_telemetry mission_monitor_node
  (also launched automatically by sim_mission.launch.py)
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
DIM     = '\033[2m'
CLEAR   = '\033[2J\033[H'   # clear screen + move cursor to top-left

# ── Human-readable state descriptions ─────────────────────────────────────
STATE_DESC = {
    'SEARCH':         'Spinning in place — scanning for gate',
    'TRACK':          'Gate found! Driving toward gate',
    'ALIGN':          'Aligning heading with gate normal',
    'CROSS':          'Crossing the gate!',
    'STOP':           'Crossed! Pausing before return run',
    'RETURN_ALIGN':   'U-turning to face gate again',
    'RETURN_TRACK':   'Driving back toward gate',
    'RETURN_ALIGN2':  'Final alignment for return crossing',
    'RETURN_CROSS':   'Crossing gate on return leg!',
    'DONE':           'ROUND TRIP COMPLETE  ★',
    'UNKNOWN':        'Waiting for navigator...',
}

STATE_COLOR = {
    'SEARCH':         YELLOW,
    'TRACK':          CYAN,
    'ALIGN':          MAGENTA,
    'CROSS':          GREEN,
    'STOP':           YELLOW,
    'RETURN_ALIGN':   MAGENTA,
    'RETURN_TRACK':   CYAN,
    'RETURN_ALIGN2':  MAGENTA,
    'RETURN_CROSS':   GREEN,
    'DONE':           GREEN,
    'UNKNOWN':        DIM,
}

W = 58   # panel width


class MissionMonitorNode(Node):
    def __init__(self):
        super().__init__('mission_monitor_node')

        self._start_wall = time.time()

        # ── Topic data cache ───────────────────────────────────────────
        self._state        = 'UNKNOWN'
        self._odom_x       = None
        self._odom_y       = None
        self._odom_yaw_deg = None
        self._odom_vel_x   = None
        self._odom_vel_yaw = None
        self._gate_2d      = None   # [detected, cx, cy, w, h, conf, _]
        self._gate_3d      = None   # [x_fwd, y_left, z_up, conf]
        self._nav_debug    = None   # 10-element array
        self._cmd_lin      = None
        self._cmd_ang      = None

        # ── Startup readiness flags (set True on first message) ────────
        self._ready_odom     = False   # Gazebo + bridge up
        self._ready_detector = False   # gate_detector_node up
        self._ready_localizer= False   # gate_localizer_node up
        self._ready_navigator= False   # gate_navigator_node up
        self._ready_logger   = False   # mission_logger_node up (inferred)
        self._ready_mapper   = False   # trail_mapper_node up (inferred via nav_debug)

        # ── Subscriptions ──────────────────────────────────────────────
        self.create_subscription(String,           '/auv/mission_state',     self._state_cb,    10)
        self.create_subscription(Odometry,         '/auv/odom',              self._odom_cb,     10)
        self.create_subscription(Float32MultiArray,'/auv/gate_detection_2d', self._det2d_cb,    10)
        self.create_subscription(Float32MultiArray,'/auv/gate_position_3d',  self._gate3d_cb,   10)
        self.create_subscription(Float32MultiArray,'/auv/navigator_debug',   self._navdbg_cb,   10)
        self.create_subscription(Twist,            '/model/auv_box/cmd_vel', self._cmd_cb,      10)

        self.create_timer(1.0, self._draw)
        self.get_logger().info('Mission Monitor started — live dashboard active.')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _state_cb(self, msg):
        self._state = msg.data
        self._ready_navigator = True

    def _odom_cb(self, msg):
        self._odom_x     = msg.pose.pose.position.x
        self._odom_y     = msg.pose.pose.position.y
        self._odom_vel_x   = msg.twist.twist.linear.x
        self._odom_vel_yaw = msg.twist.twist.angular.z
        q = msg.pose.pose.orientation
        self._odom_yaw_deg = math.degrees(math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        ))
        self._ready_odom = True

    def _det2d_cb(self, msg):
        self._gate_2d = msg.data
        self._ready_detector = True

    def _gate3d_cb(self, msg):
        self._gate_3d = msg.data
        self._ready_localizer = True

    def _navdbg_cb(self, msg):
        self._nav_debug = msg.data
        self._ready_mapper = True   # nav_debug implies navigator is up

    def _cmd_cb(self, msg):
        self._cmd_lin = msg.linear.x
        self._cmd_ang = msg.angular.z

    # ── Drawing helpers ────────────────────────────────────────────────────

    def _line(self, text='', color='', align='left'):
        inner = W - 4
        if align == 'center':
            content = text.center(inner)
        else:
            content = text.ljust(inner)
        # Strip ANSI for length calculation
        visible_len = len(text)
        pad = inner - visible_len
        if align == 'left':
            content = text + ' ' * max(0, pad)
        return f'║ {color}{content}{RESET} ║'

    def _sep(self, char='─'):
        return '╠' + char * (W - 2) + '╣'

    def _top(self):  return '╔' + '═' * (W - 2) + '╗'
    def _bot(self):  return '╚' + '═' * (W - 2) + '╝'

    def _tick(self, ready):
        return f'{GREEN}✓{RESET}' if ready else f'{YELLOW}…{RESET}'

    # ── Main draw ──────────────────────────────────────────────────────────

    def _draw(self):
        elapsed  = time.time() - self._start_wall
        mins     = int(elapsed) // 60
        secs     = int(elapsed) % 60
        runtime  = f'{mins:02d}:{secs:02d}'
        now_str  = datetime.now().strftime('%H:%M:%S')

        lines = []
        lines.append(self._top())
        lines.append(self._line(f'AUV GATE MISSION — LIVE MONITOR    {DIM}{now_str}{RESET}',
                                 BOLD, 'left'))
        lines.append(self._sep())

        # ── Startup status ─────────────────────────────────────────────
        lines.append(self._line(f'{BOLD}STARTUP{RESET}'))
        lines.append(self._line(
            f'  {self._tick(self._ready_odom)}  Gazebo + Odom bridge'
            + (f'  {GREEN}(live){RESET}' if self._ready_odom else f'  {DIM}waiting...{RESET}')
        ))
        lines.append(self._line(
            f'  {self._tick(self._ready_detector)}  Gate detector (YOLO)'
            + (f'  {GREEN}(live){RESET}' if self._ready_detector else f'  {DIM}waiting...{RESET}')
        ))
        lines.append(self._line(
            f'  {self._tick(self._ready_localizer)}  Gate localizer (3D)'
            + (f'  {GREEN}(live){RESET}' if self._ready_localizer else f'  {DIM}waiting...{RESET}')
        ))
        lines.append(self._line(
            f'  {self._tick(self._ready_navigator)}  Gate navigator (FSM)'
            + (f'  {GREEN}(live){RESET}' if self._ready_navigator else f'  {DIM}waiting...{RESET}')
        ))

        all_ready = all([self._ready_odom, self._ready_detector,
                         self._ready_localizer, self._ready_navigator])

        lines.append(self._sep())

        # ── Mission state ──────────────────────────────────────────────
        sc    = STATE_COLOR.get(self._state, DIM)
        sdesc = STATE_DESC.get(self._state, '')
        lines.append(self._line(f'{BOLD}MISSION STATE{RESET}'))
        lines.append(self._line(f'  {sc}{BOLD}{self._state:<14}{RESET} {sdesc}'))
        lines.append(self._line())

        # ── Position ───────────────────────────────────────────────────
        if self._odom_x is not None:
            lines.append(self._line(
                f'  {BOLD}Position{RESET}  '
                f'x={self._odom_x:+.2f}m  '
                f'y={self._odom_y:+.2f}m  '
                f'yaw={self._odom_yaw_deg:+.1f}°'
            ))
        else:
            lines.append(self._line(f'  {BOLD}Position{RESET}  {DIM}no odom yet{RESET}'))

        # ── Gate detection ─────────────────────────────────────────────
        if self._gate_2d is not None and len(self._gate_2d) >= 6:
            detected = self._gate_2d[0] == 1.0
            conf     = self._gate_2d[5]
            if detected:
                g3x = self._gate_3d[0] if self._gate_3d else 0.0
                gate_str = (f'{GREEN}DETECTED{RESET}  '
                            f'conf={conf:.2f}  dist≈{g3x:.1f}m fwd')
            else:
                gate_str = f'{RED}NOT DETECTED{RESET}  (conf={conf:.2f})'
            lines.append(self._line(f'  {BOLD}Gate     {RESET}  {gate_str}'))
        else:
            lines.append(self._line(f'  {BOLD}Gate     {RESET}  {DIM}no data yet{RESET}'))

        # ── Navigator internals (distance + bearing) ───────────────────
        if self._nav_debug is not None and len(self._nav_debug) >= 5:
            nd = self._nav_debug
            dist   = nd[3]
            bear   = nd[4]
            src    = int(nd[9]) if len(nd) > 9 else 0
            src_lbl = ['—', 'YOLO-live', 'odom-DR'][src] if src < 3 else '?'
            lines.append(self._line(
                f'  {BOLD}Navigator{RESET}  '
                f'dist={dist:.2f}m  '
                f'bearing={bear:+.1f}°  '
                f'src={src_lbl}'
            ))
        else:
            lines.append(self._line(f'  {BOLD}Navigator{RESET}  {DIM}no data yet{RESET}'))

        # ── Commands ───────────────────────────────────────────────────
        if self._cmd_lin is not None:
            lines.append(self._line(
                f'  {BOLD}Commands {RESET}  '
                f'fwd={self._cmd_lin:+.3f} m/s  '
                f'rot={self._cmd_ang:+.3f} rad/s'
            ))
        else:
            lines.append(self._line(f'  {BOLD}Commands {RESET}  {DIM}no commands yet{RESET}'))

        # ── Runtime ───────────────────────────────────────────────────
        lines.append(self._line())
        lines.append(self._line(f'  {DIM}Runtime: {runtime}  |  Ctrl+C to stop{RESET}'))
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
