"""Render a portable JSONL office recording as a 2-D MP4."""

from pathlib import Path

import click

from stretch_mujoco.recording.renderers import render_topdown_video
from stretch_mujoco.recording.snapshots import read_snapshots


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--fps", type=int, default=10, show_default=True)
@click.option("--scene-manifest", type=click.Path(exists=True, path_type=Path), default=None)
def main(input_path: Path, output: Path, fps: int, scene_manifest: Path | None) -> None:
    """Render a 2-D MP4 without starting MuJoCo or requiring an X display."""
    if fps <= 0:
        raise click.BadParameter("fps must be positive")
    frames = render_topdown_video(
        read_snapshots(input_path), output, fps=fps, scene_manifest=scene_manifest
    )
    click.echo(f"2-D MP4 saved -> {output} ({frames} frames)")


if __name__ == "__main__":
    main()
