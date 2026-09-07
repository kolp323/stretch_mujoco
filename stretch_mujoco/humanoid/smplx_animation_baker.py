"""Bake lightweight SMPL-X mesh animations for the MuJoCo office NPC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import click
import numpy as np

from stretch_mujoco.npc.animation.graph import OFFICE_CLIPS

from .smplx_converter import (
    SmplxAssetError,
    _load_runtime,
    _load_texture_topology,
    _to_mujoco_coordinates,
    _write_obj,
    find_model_file,
)

MATERIAL_GROUPS = ("body",)


def write_npc_asset_manifest(output_dir: Path, target_height: float) -> dict[str, Any]:
    """Write the distributable contract for already baked, locally licensed frames."""
    texture_name = "smplx_employee_diffuse.png"
    clip_files = {
        clip_name: [
            path.name for path in sorted(output_dir.glob(f"humanoid_{clip_name}_*_body.obj"))
        ]
        for clip_name in OFFICE_CLIPS
    }
    missing = [clip_name for clip_name, paths in clip_files.items() if not paths]
    if not (output_dir / texture_name).is_file():
        missing.append(texture_name)
    if missing:
        raise SmplxAssetError(f"Cannot build NPC manifest; missing: {', '.join(missing)}")
    referenced_files = [texture_name, *(path for paths in clip_files.values() for path in paths)]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_from": "licensed_smplx_parameters",
        "bundles": {
            "smplx_office_neutral_v1": {
                "format": "mesh_sequence",
                "topology_id": "smplx-neutral-10475-v1",
                "coordinate_system": "mujoco_z_up",
                "unit": "meter",
                "height_m": target_height,
                "material_slots": list(MATERIAL_GROUPS),
                "asset_quality": "restricted",
                "appearances": {
                    "employee_default_v1": {
                        "textures": {"body": texture_name},
                        "texture_topology_id": "smplx-neutral-10475-v1",
                    }
                },
                "clips": {
                    clip_name: {
                        "fps": 8.0,
                        "loop": OFFICE_CLIPS[clip_name].loop,
                        "root_motion": "in_place",
                        "frames": paths,
                        "markers": [
                            {"name": name, "phase": phase}
                            for name, phase in OFFICE_CLIPS[clip_name].markers
                        ],
                    }
                    for clip_name, paths in clip_files.items()
                },
                "sha256": {
                    filename: hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
                    for filename in referenced_files
                },
            }
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _load_retargeted_body_pose_clips(motion_root: Path, torch: Any) -> dict[str, list[Any]]:
    """Load all locally selected, provenance-backed baker inputs."""
    clips: dict[str, list[Any]] = {}
    missing: list[str] = []
    for clip_name in OFFICE_CLIPS:
        path = motion_root / f"{clip_name}.npz"
        if not path.is_file():
            missing.append(clip_name)
            continue
        with np.load(path, allow_pickle=False) as payload:
            if "body_pose" not in payload:
                raise SmplxAssetError(f"Motion clip '{path}' is missing body_pose")
            poses = np.asarray(payload["body_pose"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[0] < 2 or poses.shape[1] != 63:
            raise SmplxAssetError(
                f"Motion clip '{path}' body_pose must have shape (frames >= 2, 63)"
            )
        if not np.isfinite(poses).all():
            raise SmplxAssetError(f"Motion clip '{path}' body_pose contains non-finite values")
        clips[clip_name] = [torch.from_numpy(pose).reshape(1, -1) for pose in poses]
    if missing:
        raise SmplxAssetError(
            "Restricted motion input is incomplete; missing clips: " + ", ".join(missing)
        )
    return clips


def _face_material_groups(model: Any, faces: np.ndarray) -> dict[str, np.ndarray]:
    weights = model.lbs_weights.detach().cpu().numpy()
    dominant_joints = weights[faces].mean(axis=1).argmax(axis=1)

    skin_joints = {15, 20, 21, 22, 23, 24, *range(25, weights.shape[1])}
    pants_joints = {0, 1, 2, 4, 5, 7, 8}
    shoe_joints = {10, 11}

    template_vertices = _to_mujoco_coordinates(model.v_template.detach().cpu().numpy())
    template_height = float(np.ptp(template_vertices[:, 2]))
    normalized_height = (
        (template_vertices[:, 2] - float(template_vertices[:, 2].min())) / template_height * 1.72
    )
    face_heights = normalized_height[faces].mean(axis=1)
    # Keep shoulders and upper chest inside the shirt. The previous broad height
    # cutoff painted those surfaces as skin and produced detached shoulder patches.
    skin_mask = np.isin(dominant_joints, list(skin_joints)) | (face_heights > 1.50)

    return {
        "skin": np.flatnonzero(skin_mask),
        "shirt": np.arange(faces.shape[0]),
        "pants": np.flatnonzero(np.isin(dominant_joints, list(pants_joints))),
        "shoes": np.flatnonzero(np.isin(dominant_joints, list(shoe_joints))),
    }


def _write_compact_group(
    output_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    texture_coordinates: np.ndarray | None,
    texture_faces: np.ndarray | None,
    vertex_normals: np.ndarray,
    thickness: float = 0.0,
) -> None:
    selected_faces = faces[face_ids]
    vertex_ids, inverse_vertices = np.unique(selected_faces, return_inverse=True)
    compact_faces = inverse_vertices.reshape(selected_faces.shape)

    compact_uvs = None
    compact_texture_faces = None
    if texture_coordinates is not None and texture_faces is not None:
        selected_texture_faces = texture_faces[face_ids]
        texture_ids, inverse_textures = np.unique(selected_texture_faces, return_inverse=True)
        compact_uvs = texture_coordinates[texture_ids]
        compact_texture_faces = inverse_textures.reshape(selected_texture_faces.shape)

    compact_vertices = vertices[vertex_ids].copy()
    compact_vertices += vertex_normals[vertex_ids] * thickness
    _write_obj(
        output_path,
        compact_vertices,
        compact_faces,
        compact_uvs,
        compact_texture_faces,
    )


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-9)


def _write_material_texture(
    output_path: Path,
    texture_coordinates: np.ndarray,
    texture_faces: np.ndarray,
    face_groups: dict[str, np.ndarray],
    size: int = 1024,
) -> None:
    import cv2

    colors_rgb = {
        "skin": (184, 133, 102),
        "shirt": (43, 92, 122),
        "pants": (26, 36, 46),
        "shoes": (9, 10, 11),
    }
    face_materials = np.full(texture_faces.shape[0], "shirt", dtype=object)
    for material in ("pants", "shoes", "skin"):
        face_materials[face_groups[material]] = material

    image = np.full((size, size, 3), colors_rgb["shirt"][::-1], dtype=np.uint8)
    for face_index, uv_face in enumerate(texture_faces):
        uv = texture_coordinates[uv_face]
        pixels = np.column_stack((uv[:, 0], 1.0 - uv[:, 1]))
        pixels = np.rint(pixels * (size - 1)).astype(np.int32)
        color_bgr = colors_rgb[str(face_materials[face_index])][::-1]
        cv2.fillConvexPoly(image, pixels, color_bgr, lineType=cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), image):
        raise SmplxAssetError(f"Failed to write humanoid texture: {output_path}")


def bake_smplx_animations(
    model_root: Path,
    output_dir: Path,
    include_xml: Path,
    gender: str = "neutral",
    target_height: float = 1.72,
    asset_prefix: str = "humanoid/generated/animations",
    motion_root: Path | None = None,
) -> dict[str, Any]:
    model_file = find_model_file(model_root, "smplx", gender)
    smplx, torch = _load_runtime()
    model = smplx.create(
        str(model_file),
        model_type="smplx",
        gender=gender,
        ext=model_file.suffix.lstrip("."),
        use_pca=False,
        batch_size=1,
    )
    faces = np.asarray(model.faces, dtype=np.int64)
    texture_coordinates, texture_faces = _load_texture_topology(model_file)
    if texture_coordinates is None or texture_faces is None:
        raise SmplxAssetError("SMPL-X model does not contain UV topology.")
    face_groups = _face_material_groups(model, faces)
    if motion_root is None:
        raise SmplxAssetError("Restricted production baking requires a local --motion-root")
    clips = _load_retargeted_body_pose_clips(motion_root, torch)

    with torch.no_grad():
        reference = model(body_pose=clips["idle"][0], return_verts=True)
    reference_vertices = _to_mujoco_coordinates(reference.vertices[0].detach().cpu().numpy())
    scale = target_height / float(reference_vertices[:, 2].ptp())

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_mesh in output_dir.glob("humanoid_*.obj"):
        stale_mesh.unlink()
    _write_material_texture(
        output_dir / "smplx_employee_diffuse.png",
        texture_coordinates,
        texture_faces,
        face_groups,
    )
    mesh_elements = []

    for clip_name, poses in clips.items():
        for frame_index, body_pose in enumerate(poses):
            with torch.no_grad():
                result = model(body_pose=body_pose, return_verts=True)
            vertices = _to_mujoco_coordinates(result.vertices[0].detach().cpu().numpy())
            vertices *= scale
            vertices[:, 2] -= float(vertices[:, 2].min())
            mesh_name = f"humanoid_{clip_name}_{frame_index:02d}_body"
            filename = f"{mesh_name}.obj"
            _write_obj(
                output_dir / filename,
                vertices,
                faces,
                texture_coordinates,
                texture_faces,
            )
            mesh_elements.append(f'    <mesh name="{mesh_name}" file="{asset_prefix}/{filename}"/>')
    include_xml.parent.mkdir(parents=True, exist_ok=True)
    include_xml.write_text(
        "\n".join(["<mujoco>", "  <asset>", *mesh_elements, "  </asset>", "</mujoco>", ""]),
        encoding="utf-8",
    )
    return write_npc_asset_manifest(output_dir, target_height)


@click.command()
@click.option("--model-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--gender", type=click.Choice(("neutral", "male", "female")), default="neutral")
@click.option("--height", type=float, default=1.72, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("stretch_mujoco/models/assets/humanoid/generated/animations"),
)
@click.option(
    "--include-xml",
    type=click.Path(path_type=Path),
    default=Path("stretch_mujoco/models/assets/humanoid/generated/smplx_humanoid_assets.xml"),
)
@click.option(
    "--motion-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Directory of local AMASS intake <clip>.npz files.",
)
def main(
    model_root: Path,
    gender: str,
    height: float,
    output_dir: Path,
    include_xml: Path,
    motion_root: Path | None,
) -> None:
    """Bake low-cost office NPC mesh clips from licensed SMPL-X parameters."""
    try:
        manifest = bake_smplx_animations(
            model_root, output_dir, include_xml, gender, height, motion_root=motion_root
        )
    except SmplxAssetError as error:
        raise click.ClickException(str(error)) from error
    bundle = manifest["bundles"]["smplx_office_neutral_v1"]
    frame_count = sum(len(clip["frames"]) for clip in bundle["clips"].values())
    click.echo(f"Baked {frame_count} SMPL-X frames into {output_dir}.")


if __name__ == "__main__":
    main()
