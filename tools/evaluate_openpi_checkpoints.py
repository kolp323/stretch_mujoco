"""Evaluate retained OpenPI checkpoints on the fixed held-out grasp scenarios."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from stretch_mujoco.paths import configured_path, require_external_directory


REPO = Path(__file__).resolve().parents[1]


def checkpoint_steps(directory: Path, interval: int = 5000) -> list[tuple[int, Path]]:
    checkpoints = sorted(
        (int(path.name), path)
        for path in directory.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not checkpoints:
        return []
    latest = checkpoints[-1][0]
    return [(step, path) for step, path in checkpoints if step % interval == 0 or step == latest]


def summarize(step: int, checkpoint: Path, results: dict) -> dict:
    episodes = results.get("episodes", [])
    completed = len(episodes)
    return {
        "step": step,
        "checkpoint": str(checkpoint),
        "completed_episodes": completed,
        "success_rate": float(results.get("success_rate", 0.0)),
        "contact_rate": (
            sum(bool(item.get("bilateral_contact")) for item in episodes) / completed
            if completed
            else 0.0
        ),
        "mean_max_lift_m": (
            sum(float(item.get("max_lift_m", 0.0)) for item in episodes) / completed
            if completed
            else 0.0
        ),
    }


def select_best(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["success_rate"],
            row["contact_rate"],
            row["mean_max_lift_m"],
            -row["step"],
        ),
    )


def wait_for_server(process: subprocess.Popen, host: str, port: int, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited with code {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"policy server did not listen on {host}:{port}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=configured_path("STRETCH_MUJOCO_OPENPI_ROOT"),
        help="OpenPI checkout root (or set STRETCH_MUJOCO_OPENPI_ROOT)",
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--config", default="pi05_stretch")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        openpi_root = require_external_directory(
            args.openpi_root,
            environment_variable="STRETCH_MUJOCO_OPENPI_ROOT",
            description="OpenPI checkout root",
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    checkpoints = checkpoint_steps(args.experiment, args.interval)
    if not checkpoints:
        raise SystemExit(f"no numeric checkpoints under {args.experiment}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step, checkpoint in checkpoints:
        evaluation = args.output_dir / f"checkpoint_{step:05d}"
        results_path = evaluation / "results.json"
        if results_path.exists() and not args.overwrite:
            rows.append(summarize(step, checkpoint, json.loads(results_path.read_text())))
            continue
        if evaluation.exists():
            shutil.rmtree(evaluation)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.device
        server = subprocess.Popen(
            [
                "uv",
                "run",
                "scripts/serve_policy.py",
                f"--port={args.port}",
                "policy:checkpoint",
                f"--policy.config={args.config}",
                f"--policy.dir={checkpoint}",
            ],
            cwd=openpi_root,
            env=env,
        )
        try:
            wait_for_server(server, "127.0.0.1", args.port)
            eval_env = os.environ.copy()
            eval_env.setdefault("MUJOCO_GL", "egl")
            eval_env.setdefault("PYOPENGL_PLATFORM", "egl")
            subprocess.run(
                [
                    str(REPO / ".venv/bin/python"),
                    "examples/evaluate_openpi_policy.py",
                    "--host=127.0.0.1",
                    f"--port={args.port}",
                    "--object=blue",
                    "--scene-set=validation",
                    f"--episodes={args.episodes}",
                    f"--max-steps={args.max_steps}",
                    "--open-loop-horizon=4",
                    f"--output-dir={evaluation}",
                ],
                cwd=REPO,
                env=eval_env,
                check=True,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

        rows.append(summarize(step, checkpoint, json.loads(results_path.read_text())))
        leaderboard = {"best": select_best(rows), "checkpoints": rows}
        (args.output_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2) + "\n")

    leaderboard = {"best": select_best(rows), "checkpoints": rows}
    (args.output_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2) + "\n")
    print(json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    main()
