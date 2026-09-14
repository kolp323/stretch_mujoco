from pathlib import Path

import cv2
import numpy as np

from stretch_mujoco.npc.appearance_pipeline.accessory_recipe import SmplxHeadSurfaceFallback
from stretch_mujoco.npc.appearance_pipeline.obj_accessory_surface import (
    bake_smplx_head_surface_fallback,
)


def test_smplx_head_surface_fallback_bakes_an_opaque_uv_region(tmp_path: Path) -> None:
    source_atlas = tmp_path / "source.png"
    assert cv2.imwrite(str(source_atlas), np.full((32, 32, 3), 200, dtype=np.uint8))
    body = tmp_path / "body.obj"
    body.write_text(
        "\n".join(
            [
                "v 0 0 1.7",
                "v 0.05 0 1.7",
                "v 0 0.05 1.7",
                "vt 0.25 0.25",
                "vt 0.75 0.25",
                "vt 0.25 0.75",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n"
    )
    fallback = SmplxHeadSurfaceFallback(
        color_bgr=(36, 52, 83),
        front_hairline_z_m=1.66,
        temple_min_z_m=1.565,
        temple_min_y_m=-0.035,
        rear_min_z_m=1.515,
        rear_min_y_m=0.055,
        head_min_z_m=1.48,
        head_radius_m=0.18,
    )

    artifacts = bake_smplx_head_surface_fallback(
        source_atlas=source_atlas,
        body_frame=body,
        destination_texture=tmp_path / "fallback.png",
        destination_mask=tmp_path / "fallback.mask.png",
        fallback=fallback,
    )

    texture = cv2.imread(str(artifacts.texture), cv2.IMREAD_UNCHANGED)
    assert artifacts.pixels > 100
    assert artifacts.mask.is_file()
    assert texture is not None and texture.shape == (32, 32, 3)
    assert tuple(texture[16, 12]) != (200, 200, 200)
