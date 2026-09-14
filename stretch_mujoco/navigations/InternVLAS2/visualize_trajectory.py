#!/usr/bin/env python3
"""Draw a body-frame NavDP trajectory on the RGB image for visual comparison.

The projection replicates the model's own ``project_trajectory`` (see
``NavDP/baselines/navdp/policy_agent.py``): trajectory waypoints lie on a
horizontal plane 0.2 m below the camera and are pinhole-projected with the
camera intrinsic under a horizontal-camera assumption.

Run against the running ``/eval_dual`` server: it sends the given RGB-D pair,
and if the response contains a ``trajectory`` it draws it (green polyline,
start=green dot, end=red dot) onto the image; if it returns a
``discrete_action`` it prints it instead.

Usage
-----
  .venv/bin/python stretch_mujoco/navigations/InternVLAS2/visualize_trajectory.py \
      --rgb outputs/internvla/test_eval_dual/rgb.jpg \
      --depth outputs/internvla/test_eval_dual/depth.png \
      --server-url http://127.0.0.1:5801
"""

from __future__ import annotations

import click
import cv2
import numpy as np
from PIL import Image

from stretch_mujoco.navigations.InternVLAS2 import InternVLAClient


def project_trajectory_to_image(
    rgb: np.ndarray,
    traj_body: np.ndarray,
    intrinsic: np.ndarray,
    *,
    camera_height: float = 0.2,
    camera_pitch: float = 0.0,
) -> np.ndarray:
    """Project body-frame waypoints onto an RGB image (ground-plane model).

    The waypoints ``[x, y]`` (x forward, y left) lie on the floor (z=0).  The
    camera is ``camera_height`` m above the floor and pitched down by
    ``camera_pitch`` rad.  Standard pinhole projection:

        depth = x*cos(pitch) + height*sin(pitch)
        down  = height*cos(pitch) - x*sin(pitch)
        u = fx * y / depth + cx
        v = fy * down / depth + cy

    Parameters
    ----------
    rgb:
        BGR image ``(H, W, 3)``.
    traj_body:
        ``(N, 2)`` body-frame ``[x, y]`` waypoints (x forward).
    intrinsic:
        ``3x3`` camera K matrix (fx, fy, cx, cy).
    camera_height:
        Camera height above the floor in metres.
    camera_pitch:
        Camera pitch down in radians.
    """
    if rgb.ndim != 3:
        raise ValueError("rgb must be a 3-channel image")
    traj = np.asarray(traj_body, dtype=float)
    if traj.ndim == 1:
        traj = traj.reshape(-1, 2)
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError(f"traj_body must have shape (N, 2), got {traj.shape}")

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    h, w = rgb.shape[:2]

    x = traj[:, 0]
    y = traj[:, 1]
    p = float(camera_pitch)
    hcam = float(camera_height)
    cos_p, sin_p = np.cos(p), np.sin(p)

    depth = x * cos_p + hcam * sin_p
    down = hcam * cos_p - x * sin_p
    u = fx * y / (depth + 1e-8) + cx
    v = fy * down / (depth + 1e-8) + cy

    def pixel(i):
        if not (np.isfinite(u[i]) and np.isfinite(v[i])):
            return None
        if 0 <= u[i] < w and 0 <= v[i] < h:
            return (int(u[i]), int(v[i]))
        return None

    out = rgb.copy()
    for i in range(len(u) - 1):
        a, b = pixel(i), pixel(i + 1)
        if a is not None and b is not None:
            cv2.line(out, a, b, (0, 255, 0), 2)
    if len(u):
        start, end = pixel(0), pixel(len(u) - 1)
        if start is not None:
            cv2.circle(out, start, 6, (0, 255, 0), -1)
        if end is not None:
            cv2.circle(out, end, 6, (0, 0, 255), -1)
    return out


@click.command()
@click.option("--rgb", required=True, type=click.Path(exists=True), help="RGB image path.")
@click.option("--depth", required=True, type=click.Path(exists=True), help="Depth PNG (uint16, metres*10000).")
@click.option("--server-url", default="http://127.0.0.1:5801", show_default=True)
@click.option(
    "--intrinsic",
    default="386.5,386.5,328.9,244",
    show_default=True,
    help="Camera K (fx,fy,cx,cy) matching the image resolution.",
)
@click.option("--camera-height", default=0.2, show_default=True, help="Camera height above floor (m).")
@click.option("--camera-pitch", default=0.0, show_default=True, help="Camera pitch down (rad).")
@click.option("--instruction", default="", help="Optional instruction (server default if empty).")
@click.option("--out", default="annotated.jpg", show_default=True, help="Output image path.")
def main(rgb, depth, server_url, intrinsic, camera_height, camera_pitch, instruction, out):
    """Fetch a /eval_dual trajectory and draw it on the RGB image."""
    fx, fy, cx, cy = (float(v) for v in intrinsic.split(","))
    intrinsic_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    rgb_bgr = cv2.imread(rgb)
    depth_m = np.asarray(Image.open(depth).convert("I"), dtype=np.float32) / 10000.0
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", rgb_rgb)

    client = InternVLAClient(base_url=server_url)
    response = client.step(
        buf.tobytes(), depth_m, reset=True, instruction=instruction or None
    )

    if "trajectory" in response:
        traj = np.asarray(response["trajectory"], dtype=float)
        annotated = project_trajectory_to_image(
            rgb_bgr, traj, intrinsic_k,
            camera_height=camera_height, camera_pitch=camera_pitch,
        )
        cv2.imwrite(out, annotated)
        print(f"trajectory {len(traj)} 点 → 已绘制到 {out}")
        print(f"  body 首点: {traj[0].round(3)}  末点: {traj[-1].round(3)}")
    elif "discrete_action" in response:
        cv2.putText(
            rgb_bgr,
            f"discrete_action: {response['discrete_action']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(out, rgb_bgr)
        print(f"本次返回离散动作 {response['discrete_action']}（无轨迹可绘），已存到 {out}")
    else:
        print(f"未识别的响应: {response}")


if __name__ == "__main__":
    main()
