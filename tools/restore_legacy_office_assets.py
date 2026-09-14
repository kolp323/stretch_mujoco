"""Restore the original 39 textured HSSD office assets beside the newer MJCF assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import click

from stretch_mujoco.office_asset_gallery import (
    ASSET_ROOT,
    GalleryAsset,
    convert_gallery_assets,
    write_office_asset_registry,
)
from stretch_mujoco.paths import cache_root, configured_path, require_external_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HSSD_ROOT = configured_path("STRETCH_MUJOCO_HSSD_ROOT")
DEFAULT_KTX = cache_root() / "ktx-software" / "usr" / "bin" / "ktx"

LEGACY_ASSETS = {
    "furniture/chairs": (
        "fa26f3ac2b4003fae1a5025526aabaaaa8458d37",
        "d6f9b4561838f68f34d3fc36a7b700bc2c180253",
        "a5f9a4e0b89f85a3f8e117c3662541608f938b38",
    ),
    "furniture/desks": (
        "f8c1be8fafd7d652e850774aa20a4e33feddea4c",
        "f84624a16fc343ce94373e0ec45362127c70451a",
        "da6ab8a359f639f9bbd92f51bf7ffc89f65714ee",
    ),
    "furniture/sofas": (
        "e94ede305ff3f873dd34f8e95274bb54679df6b4",
        "9a0e617b83c901d346ce9fb020b49bf91442273c",
        "5274751529e50c02430fb9b3ee3019ea9c091520",
    ),
    "furniture/storage": (
        "ed95c877f692c3fad1b50a96e21ea1e02101dd42",
        "aa69ad95160221cb20a99652a2d8b905bcefdac4",
        "aa48e0adc7ea84c3c8dbb9530d5a56be5c22c13f",
    ),
    "electronics": (
        "958b2b2572ebb43a525c6993a754d92a50d4c889",
        "2efebdcc1ba9514b0deb0dfb05951341432bd26a",
        "fbe1da92b4ec1d1ab6540aca91a52c64d42a6229",
    ),
    "lighting": (
        "xxxxcb321f68x12e2x49baxa491x042208fe4615",
        "xxxx1a9211bbxdbd2x4e77xb598xc693c6b7122e",
        "xxxx07289eaex9e77x42acxaeacxd40ba4b679c4",
    ),
    "props/plants": (
        "fbeea099466186ec2d4af811c7ab24169c822092",
        "f0aefc7dcac192761cf696918086d2a21f9a5ee4",
        "dbaf471314440f77d069d09e56fd02d3ed70a99e",
    ),
    "props/trash_bins": (
        "e77fdd847315d42f536995b42866e90cc305a79b",
        "e1d968f387df48e1f1f3c5df126e497f4884e924",
        "da0efbb3c1d8989b23d18681cdb04e49da9ef9d2",
    ),
    "props/books": (
        "edf5a667080ed733fda55514ee18a8081f9dcf9c",
        "e6a24c5d82ac274c29caa24dc71c18c5c5481a56",
        "d344e96d26c7f5ca8ffcad826d39617635f81def",
    ),
    "props/drinkware": (
        "fcae3d93545969544b5aec155f56fda69f1a2f9a",
        "ed95f32cc00075ceed5bd4b593d2a06a444c0eec",
        "e3479ce73271171bdaa163968c50500719ed4d72",
    ),
    "decoration/rugs": (
        "ff3f501d99144ddb7e9aca99702f609ae6754a4c",
        "e4d32d37fb98def0c95c79a2f337785c63829074",
        "e1e746a367ad18a84242bc1f4a7c1196b251c7a8",
    ),
    "decoration/curtains": (
        "ffd5d856642f1a686d533ad41029edd975271a7e",
        "fc71f5447c513b19a3c7218beb81ffdd47423173",
        "f68d3d5d97fec3128fb897500b9428d0abe7f6a6",
    ),
    "decoration/wall_art": (
        "da909ae4a0f2411ddfe5f499f11bee8cec508197",
        "b645ac0b787d641c9a28d4edee3fd99704d63654",
        "8c73109a6020fbf91d2491766b22e7167f13ae21",
    ),
}


def _display_names(hssd_root: Path) -> dict[str, str]:
    semantics = hssd_root / "semantics" / "objects.csv"
    with semantics.open(newline="", encoding="utf-8") as stream:
        return {row["id"]: row.get("name") or row["id"] for row in csv.DictReader(stream)}


def _source_glb(hssd_root: Path, asset_id: str) -> Path:
    config_path = hssd_root / "objects" / asset_id[0] / f"{asset_id}.object_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return (config_path.parent / config["render_asset"]).resolve()


def legacy_gallery_assets(hssd_root: Path) -> tuple[GalleryAsset, ...]:
    names = _display_names(hssd_root)
    return tuple(
        GalleryAsset(
            asset_id=asset_id,
            display_name=names.get(asset_id, asset_id),
            category=category,
            source_glb=_source_glb(hssd_root, asset_id),
            converted_dir=ASSET_ROOT / category / asset_id,
        )
        for category, asset_ids in LEGACY_ASSETS.items()
        for asset_id in asset_ids
    )


@click.command()
@click.option(
    "--hssd-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=DEFAULT_HSSD_ROOT,
    help="HSSD dataset root (or set STRETCH_MUJOCO_HSSD_ROOT).",
)
@click.option("--rebuild", is_flag=True, help="Reconvert assets that are already present.")
@click.option(
    "--ktx",
    "ktx_command",
    type=click.Path(path_type=Path),
    default=DEFAULT_KTX,
    show_default=True,
)
def main(hssd_root: Path | None, rebuild: bool, ktx_command: Path) -> None:
    """Recreate the legacy textured catalog at its original office_assets paths."""
    try:
        hssd_root = require_external_directory(
            hssd_root,
            environment_variable="STRETCH_MUJOCO_HSSD_ROOT",
            description="HSSD dataset root",
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    assets = legacy_gallery_assets(hssd_root)
    converted = convert_gallery_assets(
        assets,
        rebuild=rebuild,
        ktx_command=ktx_command if ktx_command.is_file() else None,
    )
    registry = ASSET_ROOT / "mjcf" / "office_assets.xml"
    write_office_asset_registry(converted, registry)
    manifest = {
        "asset_count": len(assets),
        "source": {"dataset": "HSSD", "root_env": "STRETCH_MUJOCO_HSSD_ROOT"},
        "registry": str(registry.relative_to(PROJECT_ROOT)),
        "assets": [
            {
                "asset_id": asset.asset_id,
                "name": asset.display_name,
                "category": asset.category,
                "directory": str(asset.converted_dir.relative_to(PROJECT_ROOT)),
            }
            for asset in assets
        ],
    }
    manifest_path = ASSET_ROOT / "catalog" / "legacy_restored.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    click.echo(f"Restored {len(assets)} legacy assets and wrote {registry}")


if __name__ == "__main__":
    main()
