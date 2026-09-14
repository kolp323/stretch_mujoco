#!/usr/bin/env python3
"""Capture a realistic aligned RGB-D test pair from a MuJoCo scene.

Useful for testing ``http_internvla_server.py`` (``/eval_dual``) with images and
depth that actually match what the robot would see.  The render resolution and
vertical FOV are chosen to match the server's fixed D455 camera intrinsic
(``fx=386.5`` at 640x480), so pixel-goal trajectory projection stays correct.

Saves ``rgb.jpg`` + ``depth.png`` (uint16, metres * 10000) into an output dir
and prints a ready-to-run ``curl`` command.

Usage
-----
  .venv/bin/python stretch_mujoco/navigations/InternVLAS2/capture_test_pair.py \
      --pose "1.0,0.0,0.0" --out outputs/internvla/test_eval_dual/mujoco
"""

from __future__ import annotations

import math
from pathlib import Path

import click

from stretch_mujoco.navigations.NavDP.navdp_client import _encode_depth_png
from stretch_mujoco.navigations.VLFM.image_capture import ImageCapture


def _default_scene() -> str:
    import stretch_mujoco

    return str(Path(stretch_mujoco.__file__).resolve().parent / "models" / "office_scene.xml")


@click.command()
@click.option("--scene", default=None, show_default=True, help="MuJoCo scene XML path.")
@click.option("--pose", default="1.0,0.0,0.0", show_default=True, help="World 'x,y,yaw' of the camera.")
@click.option("--camera-height", default=1.0, show_default=True)
@click.option("--pitch", default=-0.15, show_default=True)
@click.option("--out", default="test_eval_dual/mujoco", show_default=True, help="Output directory.")
def main(scene, pose, camera_height, pitch, out):
    """Capture one aligned RGB-D pair from the scene."""
    import mujoco

    scene_path = scene or _default_scene()
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    x, y, yaw = (float(v) for v in pose.split(","))
    # 640x480 at vfov = 2*atan((H/2)/fx) matches the server's D455 intrinsic.
    width, height = 640, 480
    fovy = 2.0 * math.degrees(math.atan((height / 2.0) / 386.5))

    capture = ImageCapture(model, data, width=width, height=height)
    rgb_bytes, depth_m = capture.capture_at_robot(
        (x, y), yaw, camera_height=camera_height, pitch=pitch, fovy=fovy
    )
    capture.close()

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rgb.jpg").write_bytes(rgb_bytes)
    (out_dir / "depth.png").write_bytes(_encode_depth_png(depth_m))

    print(f"Scene: {scene_path}")
    print(f"Pose:  x={x}, y={y}, yaw={yaw:.2f}  (camera_height={camera_height}, pitch={pitch})")
    print(f"Render: {width}x{height} @ fovy={fovy:.1f} deg  (matches server fx=386.5)")
    print(f"Saved: {out_dir / 'rgb.jpg'} + {out_dir / 'depth.png'}")
    print()
    print("测试 /eval_dual：")
    print("  curl -X POST http://127.0.0.1:5801/eval_dual \\")
    print(f"    -F 'image=@{out_dir / 'rgb.jpg'}' \\")
    print(f"    -F 'depth=@{out_dir / 'depth.png'}' \\")
    print("    -F 'json={\"reset\": true}'")


if __name__ == "__main__":
    main()
