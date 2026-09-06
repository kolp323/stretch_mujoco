"""Render a portable JSONL office recording as an offscreen MuJoCo 3-D MP4."""

from pathlib import Path

import click

from stretch_mujoco.recording.renderers import render_mujoco_video
from stretch_mujoco.recording.snapshots import read_snapshots

DEFAULT_SCENE = (
    Path(__file__).resolve().parents[1]
    / "stretch_mujoco"
    / "models"
    / "office_scene2_multi_npc.xml"
)


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option(
    "--scene",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_SCENE,
    show_default=True,
)
@click.option("--fps", type=int, default=10, show_default=True)
@click.option("--width", type=int, default=1280, show_default=True)
@click.option("--height", type=int, default=720, show_default=True)
def main(input_path: Path, output: Path, scene: Path, fps: int, width: int, height: int) -> None:
    """Render 3-D MP4 offscreen; set MUJOCO_GL=egl on headless Linux hosts."""
    frames = render_mujoco_video(
        read_snapshots(input_path),
        output,
        scene_path=scene,
        fps=fps,
        width=width,
        height=height,
    )
    click.echo(f"3-D MP4 saved -> {output} ({frames} frames)")


if __name__ == "__main__":
    main()
