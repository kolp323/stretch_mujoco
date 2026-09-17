"""Config-only XML/manifest semantic inventory for the twenty active scenes.

The report intentionally exits non-zero whenever the current policies leave
entities unresolved.  It is an audit artifact, not a permissive fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile

from stretch_mujoco.npc.provenance import validate_active_catalog
from stretch_mujoco.npc.scene_compiler import (
    compile_scene_npc_config,
    config_only_inventory,
)
try:
    from tools.validate_interaction_site_plan import validate_config
except ModuleNotFoundError:
    from validate_interaction_site_plan import validate_config


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco" / "models"


def require_production_acceptance(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("acceptance_level") != "production_physical"
        or payload.get("production_evidence") is not True
        or payload.get("runtime_physical_validation") != "passed"
        or payload.get("passed") is not True
    ):
        raise RuntimeError("active_publish_requires_production_physical_acceptance")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _compile_active_catalog(config_paths: list[Path], output_root: Path) -> dict:
    """Publish a content-addressed build only after every scene passes."""
    output_root = output_root.resolve()
    build_parent = output_root / "builds"
    build_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scene-npc-active-", dir=build_parent) as temporary:
        staging = Path(temporary)
        scenes: dict[str, dict[str, str]] = {}
        receipt_bytes: list[bytes] = []
        for index, config_path in enumerate(sorted(config_paths, key=lambda item: item.stem), start=1):
            try:
                outputs = compile_scene_npc_config(
                    config_path, staging / config_path.stem / "out.xml"
                )
            except Exception as error:
                raise RuntimeError(
                    f"active_scene_compile_failed:{config_path.stem}:{error}"
                ) from error
            receipt_bytes.append(outputs["receipt"].read_bytes())
            scenes[config_path.stem] = {
                name: str(path.relative_to(staging)) for name, path in outputs.items()
            }
            print(
                json.dumps(
                    {
                        "compiled_scene": config_path.stem,
                        "index": index,
                        "total": len(config_paths),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        build_id = hashlib.sha256(b"".join(receipt_bytes)).hexdigest()
        build_root = build_parent / build_id
        if build_root.exists():
            # A prior interrupted publish may have left the same content hash
            # pointing at stale receipts. Replace only this generated build.
            shutil.rmtree(build_root)
        os.replace(staging, build_root)
        catalog = {
            "schema_version": 1,
            "build_id": build_id,
            "build_root": str(build_root.relative_to(output_root)),
            "scene_count": len(scenes),
            "scenes": scenes,
        }
        _atomic_write_json(output_root / "active_catalog.json", catalog)
        return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compile-output-root", type=Path)
    parser.add_argument("--acceptance-summary", type=Path)
    parser.add_argument(
        "--interaction-sites",
        action="store_true",
        help="audit candidate plans without publishing the active catalog",
    )
    args = parser.parse_args()
    catalog = json.loads((MODELS / "scene_npc_configs" / "catalog.json").read_text())
    config_paths = [MODELS / "scene_npc_configs" / relative for relative in catalog["configs"]]
    if args.interaction_sites:
        interaction_records = [validate_config(path) for path in config_paths]
        report = {
            "schema_version": 2,
            "scope": "npc_interactions_candidate",
            "robot_capabilities": "deferred_out_of_scope",
            "production_evidence": False,
            "scene_count": len(interaction_records),
            "scenes": interaction_records,
        }
        _write_json(args.report, report)
        print(json.dumps({"scene_count": len(interaction_records), "candidate_only": True}, sort_keys=True))
        return
    records = [config_only_inventory(path) for path in config_paths]
    report = {
        "schema_version": 1,
        "scenes": records,
        "scene_count": len(records),
        "unresolved_scene_count": sum(record["status"] != "closed" for record in records),
    }
    if report["unresolved_scene_count"]:
        _write_json(args.report, report)
        print(
            json.dumps(
                {
                    "scene_count": report["scene_count"],
                    "unresolved_scene_count": report["unresolved_scene_count"],
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2)
    if args.compile_output_root is not None:
        if args.acceptance_summary is None:
            raise RuntimeError("active_publish_requires_acceptance_summary")
        require_production_acceptance(args.acceptance_summary)
        compiled = _compile_active_catalog(config_paths, args.compile_output_root)
        report["compiled_build_id"] = compiled["build_id"]
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "scene_count": report["scene_count"],
                "unresolved_scene_count": report["unresolved_scene_count"],
                "compiled_build_id": report.get("compiled_build_id"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
