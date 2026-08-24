#!/usr/bin/env python3
"""
bottom_cam_altitude_test.py
============================
Spawns (teleports) the AUV to a grid of positions above the blue bin
and green mat, captures one bottom-camera frame at each position, and
saves annotated PNG images so you can visually inspect what the
detector sees at different altitudes and offsets.

Blue bin (drum_blue):  world (7, 22, -4.75)  — top face at z = -4.45
Green mat:             roughly world (10-14, 19-22)

Usage
-----
  # Terminal 1 — start Gazebo with the mission world:
  source ~/auv_ws/install/setup.bash
  gz sim -r ~/auv_ws/camera_test.sdf

  # Terminal 2 — run this script:
  source ~/auv_ws/install/setup.bash
  python3 ~/auv_ws/src/auv_vision/auv_vision/bottom_cam_altitude_test.py

Results saved to:  ~/auv_ws/bottom_cam_test/
"""

import os
import sys
import time
import json
import subprocess
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  TEST GRID
#  Each entry is (dx, dy, label) relative to the blue bin world position.
#  z is the AUV base_link height (negative = below water surface).
#  The bin top is at z = -4.45, camera is 0.28 m below base_link, so:
#    auv_z = -2.0  →  cam_z = -2.28  →  1.17 m above bin top
#    auv_z = -2.5  →  cam_z = -2.78  →  1.67 m above bin top
#    auv_z = -3.0  →  cam_z = -3.28  →  2.17 m above bin top  (current param)
#    auv_z = -3.5  →  cam_z = -3.78  →  2.67 m above bin top
#    auv_z = -4.0  →  cam_z = -4.28  →  0.17 m above bin top  (very close!)
# ─────────────────────────────────────────────────────────────────────────────

BIN_X, BIN_Y = 7.0, 22.0          # drum_blue world position
BOTTOM_CAM_OFFSET_Z = -0.28       # camera below base_link (matches model.sdf)
BIN_TOP_Z = -4.45                  # top face of drum

# Altitudes to test (AUV base_link z)
TEST_ALTITUDES_Z = [-2.0, -2.5, -3.0, -3.5, -4.0]

# Horizontal offsets (relative to bin) to test at each altitude
HORIZONTAL_OFFSETS = [
    (0.0,  0.0,  "over_bin"),
    (1.0,  0.0,  "1m_right"),
    (-1.0, 0.0,  "1m_left"),
    (0.0,  1.0,  "1m_fwd"),
    (2.0,  0.0,  "2m_right"),
    (0.0, -1.5,  "1.5m_back"),
]

WORLD_NAME       = "auv_pool_world"
MODEL_NAME       = "auv_box"
BOTTOM_CAM_TOPIC = "/model/auv_box/bottom/image_raw"
OUTPUT_DIR       = os.path.expanduser("~/auv_ws/bottom_cam_test")


def _gz_set_pose(model_name: str, world_name: str,
                 x: float, y: float, z: float, yaw: float = 1.5708) -> bool:
    """Teleport a Gazebo model via gz service (gz-sim 8 / Harmonic)."""
    req = (
        f"entity: {{name: '{model_name}', type: 2}}, "
        f"pose: {{position: {{x: {x}, y: {y}, z: {z}}}, "
        f"orientation: {{x: 0, y: 0, "
        f"z: {math.sin(yaw/2):.6f}, w: {math.cos(yaw/2):.6f}}}}}"
    )
    cmd = [
        "gz", "service",
        "-s", f"/world/{world_name}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", req,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        print("  [gz set_pose] TIMEOUT — is Gazebo running?")
        return False
    if result.returncode != 0:
        print(f"  [gz set_pose] WARN: {result.stderr.strip()}")
        return False
    return True


class BottomCamCapture(Node):
    """Minimal node: grabs one frame from the bottom camera on demand."""

    def __init__(self):
        super().__init__("bottom_cam_altitude_test")
        self.bridge = CvBridge()
        self._frame = None
        self.create_subscription(Image, BOTTOM_CAM_TOPIC, self._img_cb, 1)
        self.get_logger().info(f"Subscribed to {BOTTOM_CAM_TOPIC}")

    def _img_cb(self, msg: Image):
        self._frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def capture(self, timeout_sec: float = 5.0):
        """Block until one fresh frame arrives. Returns ndarray or None."""
        self._frame = None
        deadline = time.monotonic() + timeout_sec
        while self._frame is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() > deadline:
                self.get_logger().warn("Timeout waiting for bottom camera frame!")
                return None
        return self._frame.copy()


def annotate(img: np.ndarray, label: str, auv_z: float,
             cam_above_bin: float) -> np.ndarray:
    """Burn position metadata and a centre crosshair onto the image."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        label,
        f"AUV z = {auv_z:.2f} m",
        f"Cam above bin top = {cam_above_bin:.2f} m",
    ]
    for i, line in enumerate(lines):
        y = 30 + i * 30
        cv2.putText(img, line, (9,  y + 1), font, 0.75, (0, 0, 0),       2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y),     font, 0.75, (255, 255, 255),  1, cv2.LINE_AA)

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 25, cy), (cx + 25, cy), (0, 255, 255), 1)
    cv2.line(img, (cx, cy - 25), (cx, cy + 25), (0, 255, 255), 1)
    return img


def build_contact_sheet(results: list, ncols: int,
                        thumb_w=320, thumb_h=240) -> np.ndarray:
    thumbs = []
    for r in results:
        if r["ok"]:
            img = cv2.imread(r["file"])
            thumbs.append(cv2.resize(img, (thumb_w, thumb_h)))
        else:
            blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            cv2.putText(blank, r["label"], (10, thumb_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
            thumbs.append(blank)

    nrows = math.ceil(len(thumbs) / ncols)
    rows  = []
    for ri in range(nrows):
        row = thumbs[ri * ncols: (ri + 1) * ncols]
        while len(row) < ncols:
            row.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rclpy.init()
    node = BottomCamCapture()

    print(f"\nBlue bin at world ({BIN_X}, {BIN_Y}, -4.75)  |  bin top at z={BIN_TOP_Z}")
    print(f"Output dir: {OUTPUT_DIR}\n")
    print("Waiting 2 s for Gazebo to settle…")
    time.sleep(2.0)

    results = []
    total   = len(TEST_ALTITUDES_Z) * len(HORIZONTAL_OFFSETS)
    idx     = 0

    for auv_z in TEST_ALTITUDES_Z:
        cam_z         = auv_z + BOTTOM_CAM_OFFSET_Z
        cam_above_bin = abs(cam_z - BIN_TOP_Z)

        for (dx, dy, hlabel) in HORIZONTAL_OFFSETS:
            idx += 1
            tx    = BIN_X + dx
            ty    = BIN_Y + dy
            label = f"z{auv_z:.1f}_{hlabel}".replace("-", "n")
            fname = os.path.join(OUTPUT_DIR, f"{label}.png")

            print(f"[{idx:>2}/{total}]  ({tx:.1f}, {ty:.1f}, {auv_z:.1f})  "
                  f"cam {cam_above_bin:.2f}m above bin  [{hlabel}]")

            ok = _gz_set_pose(MODEL_NAME, WORLD_NAME, tx, ty, auv_z)
            if not ok:
                print("         ↳ gz set_pose failed — is Gazebo running?")

            # Let the physics engine and renderer settle before grabbing frame
            settle_sec = 2.5 if idx == 1 else 1.2
            time.sleep(settle_sec)

            frame = node.capture(timeout_sec=4.0)
            if frame is None:
                print(f"         ↳ No frame — skipping")
                results.append({"label": label, "ok": False, "file": fname})
                continue

            frame = annotate(frame, label, auv_z, cam_above_bin)
            cv2.imwrite(fname, frame)
            print(f"         ↳ Saved {os.path.basename(fname)}  "
                  f"({frame.shape[1]}x{frame.shape[0]})")
            results.append({
                "label":           label,
                "ok":              True,
                "auv_xyz":         [tx, ty, auv_z],
                "cam_above_bin_m": round(cam_above_bin, 3),
                "file":            fname,
            })

    # ── Write summary JSON ───────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Contact sheet: all frames in a grid ─────────────────────────────
    ncols = len(HORIZONTAL_OFFSETS)
    sheet = build_contact_sheet(results, ncols)
    sheet_path = os.path.join(OUTPUT_DIR, "contact_sheet.png")
    cv2.imwrite(sheet_path, sheet)

    good = sum(r["ok"] for r in results)
    print(f"\n{'='*60}")
    print(f"Done.  {good}/{total} frames captured.")
    print(f"Images:        {OUTPUT_DIR}/")
    print(f"Contact sheet: {sheet_path}")
    print(f"Summary JSON:  {summary_path}")
    print(f"{'='*60}\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
