"""Move both Trossen arms to the BiNoMaP goal-capture "park" pose and hold it.

Identical pose construction to sim_3d_bimanual-main/generate_goal_pc.py
(LIFT_J1_RAD=1.0, SWING_JOINT_IDX=2, SWING_ANGLE_RAD=0.7) — arms keep their
home j0/j3, lift the shoulder (j1 → ~0.047 rad), and swing the elbow (j2)
forward by 0.7 rad so both grippers clear the workspace crop. Both arms get
+0.7 on j2 (the existing script does the same — comment "left = -, right = +"
is misleading; code is identical for both sides).

The move uses arm.driver.set_all_positions() directly (NOT lerobot's
robot.send_action) because send_action's ~0.1 s MIN_TIME_TO_MOVE triggers
joint-velocity-limit faults on a multi-rad move. We give the firmware 6 s to
blend in.

After arrival the script blocks indefinitely. Trossen firmware holds the last
commanded position-mode target — no need to re-send. Ctrl-C falls through to
robot.disconnect(), which slow-moves home then sleep_pose.

Usage::

    python park_pose.py
"""

import argparse
import time

import numpy as np

from lerobot.common.robot_devices.robots.configs import TrossenAIStationaryRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config


# Pose constants — verbatim from sim_3d_bimanual-main/generate_goal_pc.py lines 72-83.
HOME_POSE_PER_ARM = [0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0]
SWING_JOINT_IDX   = 2       # j2 (elbow)
SWING_ANGLE_RAD   = 0.7000
LIFT_J1_RAD       = 1.0     # subtract from j1 to lift the shoulder upward

_lifted_base = list(HOME_POSE_PER_ARM)
_lifted_base[1] = max(0.0, HOME_POSE_PER_ARM[1] - LIFT_J1_RAD)

LEFT_PARK_POSE  = list(_lifted_base);  LEFT_PARK_POSE[SWING_JOINT_IDX]  = +SWING_ANGLE_RAD
RIGHT_PARK_POSE = list(_lifted_base);  RIGHT_PARK_POSE[SWING_JOINT_IDX] = +SWING_ANGLE_RAD
PER_ARM_PARK_POSES = {"left": LEFT_PARK_POSE, "right": RIGHT_PARK_POSE}

PARK_BLEND_S   = 6.0   # blend time the arm interpolates over
SETTLE_SECONDS = PARK_BLEND_S + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-s", type=float, default=PARK_BLEND_S,
                    help="Seconds the firmware blends from current → park pose.")
    args = ap.parse_args()

    print(f"Park pose per arm (7-D): {LEFT_PARK_POSE}")

    robot_cfg = TrossenAIStationaryRobotConfig(
        camera_interface="intel_realsense",
    )
    # Skip cameras: this script only commands arms.
    robot_cfg.cameras = {}
    robot = make_robot_from_config(robot_cfg)
    robot.leader_arms = {}
    robot.cameras = {}
    robot.connect()

    try:
        print(f"\nMoving {len(robot.follower_arms)} arm(s) to park pose over {args.blend_s:.1f}s...")
        for arm_name, arm in robot.follower_arms.items():
            pose = PER_ARM_PARK_POSES.get(arm_name, list(HOME_POSE_PER_ARM))
            print(f"  {arm_name}: {[round(x, 4) for x in pose]}")
            arm.driver.set_all_positions(pose, args.blend_s, False)
        time.sleep(args.blend_s + 1.0)
        print("\nHolding park pose. Position-mode firmware will keep the arms here.")
        print("Press Ctrl-C to release — script will slow-move HOME and then to sleep pose.\n")

        # Trossen firmware actively holds the last position-mode target, so we
        # just block. A periodic dot keeps the terminal alive so you can tell
        # the script hasn't crashed.
        tick = 0
        while True:
            time.sleep(10.0)
            tick += 1
            print(f"  still holding ({tick * 10}s)…", flush=True)
    except KeyboardInterrupt:
        print("\nCtrl-C — releasing arms (driver disconnect will slow-move HOME → sleep pose).")
    finally:
        if robot.is_connected:
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"robot.disconnect() raised: {exc}")


if __name__ == "__main__":
    main()
