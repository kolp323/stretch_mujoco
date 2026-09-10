"""Convert a locally licensed SMPL / SMPL-X model into a MuJoCo visual mesh."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
import numpy as np


SUPPORTED_MODEL_TYPES = ("smpl", "smplx")
SUPPORTED_GENDERS = ("neutral", "male", "female")


class SmplxAssetError(RuntimeError):
    """Raised when licensed model parameters or optional dependencies are missing."""


def find_model_file(model_root: Path, model_type: str, gender: str) -> Path:
    """Find an official SMPL-family parameter file without assuming folder layout."""
    model_type = model_type.lower()
    gender = gender.lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise SmplxAssetError(f"Unsupported model type: {model_type}")
    if gender not in SUPPORTED_GENDERS:
        raise SmplxAssetError(f"Unsupported gender: {gender}")

    expected_stem = f"{model_type}_{gender}".upper()
    candidates: list[Path] = []
    visited_directories: set[Path] = set()
    for directory, directory_names, file_names in os.walk(model_root, followlinks=True):
        resolved_directory = Path(directory).resolve()
        if resolved_directory in visited_directories:
            directory_names.clear()
            continue
        visited_directories.add(resolved_directory)
        candidates.extend(
            Path(directory) / file_name
            for file_name in file_names
            if Path(file_name).suffix.lower() in {".npz", ".pkl"}
            and Path(file_name).stem.upper() == expected_stem
        )
    candidates.sort()
    if not candidates:
        expected = f"{expected_stem}.npz or {expected_stem}.pkl"
        raise SmplxAssetError(
            f"No licensed {model_type.upper()} parameters found below {model_root}. "
            f"Expected {expected}."
        )
    return candidates[0]


def _load_runtime() -> tuple[Any, Any]:
    try:
        import smplx
        import torch
    except ImportError as error:
        raise SmplxAssetError(
            "SMPL runtime is not installed. Run: " "uv pip install --torch-backend cpu torch smplx"
        ) from error
    return smplx, torch


def _write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    texture_coordinates: np.ndarray | None = None,
    texture_faces: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        output.write("# Generated locally from licensed SMPL-family parameters.\n")
        for x, y, z in vertices:
            output.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        if texture_coordinates is not None:
            for u, v in texture_coordinates:
                output.write(f"vt {u:.8f} {v:.8f}\n")
        if texture_coordinates is not None and texture_faces is not None:
            for vertex_face, texture_face in zip(faces + 1, texture_faces + 1):
                corners = " ".join(
                    f"{vertex_index}/{texture_index}"
                    for vertex_index, texture_index in zip(vertex_face, texture_face)
                )
                output.write(f"f {corners}\n")
        else:
            for a, b, c in faces + 1:
                output.write(f"f {a} {b} {c}\n")


def _load_texture_topology(model_file: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if model_file.suffix.lower() != ".npz":
        return None, None
    with np.load(model_file, allow_pickle=True) as model_data:
        if "vt" not in model_data or "ft" not in model_data:
            return None, None
        return np.asarray(model_data["vt"]), np.asarray(model_data["ft"], dtype=np.int64)


def _relaxed_body_pose(torch: Any) -> Any:
    """Create a standing SMPL-X pose with arms lowered from the rest T-pose."""
    body_pose = torch.zeros((1, 21, 3), dtype=torch.float32)
    body_pose[0, 15, 2] = -1.25
    body_pose[0, 16, 2] = 1.25
    body_pose[0, 17, 2] = -0.12
    body_pose[0, 18, 2] = 0.12
    return body_pose.reshape(1, -1)


def _to_mujoco_coordinates(points: np.ndarray) -> np.ndarray:
    """Convert SMPL's Y-up coordinates to MuJoCo's Z-up convention."""
    return np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))


def convert_smplx_model(
    model_root: Path,
    output_obj: Path,
    model_type: str = "smplx",
    gender: str = "neutral",
    target_height: float = 1.72,
    pose: str = "relaxed",
) -> dict[str, Any]:
    """Generate a normalized standing mesh and joint metadata for the NPC pipeline."""
    if target_height <= 0:
        raise SmplxAssetError("Target height must be positive.")

    model_file = find_model_file(model_root, model_type, gender)
    smplx, torch = _load_runtime()
    model = smplx.create(
        str(model_file),
        model_type=model_type,
        gender=gender,
        ext=model_file.suffix.lstrip("."),
        use_pca=False,
        batch_size=1,
    )

    model_parameters = {}
    if pose == "relaxed":
        model_parameters["body_pose"] = _relaxed_body_pose(torch)
    elif pose != "neutral":
        raise SmplxAssetError(f"Unsupported pose: {pose}")

    with torch.no_grad():
        result = model(return_verts=True, **model_parameters)

    vertices = _to_mujoco_coordinates(result.vertices[0].detach().cpu().numpy())
    joints = _to_mujoco_coordinates(result.joints[0].detach().cpu().numpy())
    source_height = float(vertices[:, 2].max() - vertices[:, 2].min())
    if source_height <= 0:
        raise SmplxAssetError("Generated model has an invalid height.")

    scale = target_height / source_height
    vertices *= scale
    joints *= scale
    floor_offset = float(vertices[:, 2].min())
    vertices[:, 2] -= floor_offset
    joints[:, 2] -= floor_offset

    texture_coordinates, texture_faces = _load_texture_topology(model_file)
    _write_obj(
        output_obj,
        vertices,
        np.asarray(model.faces, dtype=np.int64),
        texture_coordinates,
        texture_faces,
    )
    metadata = {
        "source_model": str(model_file),
        "model_type": model_type,
        "gender": gender,
        "target_height_m": target_height,
        "pose": pose,
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(np.asarray(model.faces).shape[0]),
        "texture_coordinate_count": (
            int(texture_coordinates.shape[0]) if texture_coordinates is not None else 0
        ),
        "joints_m": joints.tolist(),
        "note": "The source parameter file remains local and is not copied.",
    }
    metadata_path = output_obj.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@click.command()
@click.option(
    "--model-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Directory containing the locally licensed SMPL-family parameter file.",
)
@click.option("--model-type", type=click.Choice(SUPPORTED_MODEL_TYPES), default="smplx")
@click.option("--gender", type=click.Choice(SUPPORTED_GENDERS), default="neutral")
@click.option("--height", type=float, default=1.72, show_default=True)
@click.option("--pose", type=click.Choice(("relaxed", "neutral")), default="relaxed")
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("stretch_mujoco/models/assets/humanoid/generated/employee_visual.obj"),
    show_default=True,
)
def main(
    model_root: Path,
    model_type: str,
    gender: str,
    height: float,
    pose: str,
    output: Path,
) -> None:
    """Prepare an SMPL-family visual mesh for the office NPC."""
    try:
        metadata = convert_smplx_model(model_root, output, model_type, gender, height, pose)
    except SmplxAssetError as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"Generated {output} with {metadata['vertex_count']} vertices at "
        f"{metadata['target_height_m']:.2f} m."
    )


if __name__ == "__main__":
    main()
