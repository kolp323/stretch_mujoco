"""Run static or live-MuJoCo interaction acceptance without publishing active resources."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from stretch_mujoco.agents import ActionCommand, ActionExecution, ActionType, ExecutionStatus
from stretch_mujoco.agents.drivers import MujocoNpcActionDriver
from stretch_mujoco.agents.interaction_stations import InteractionStationCatalog, SeatSlotAllocator
from stretch_mujoco.npc.composition import load_composed_npc_runtime
from stretch_mujoco.npc.protocol import NpcCommand, NpcCommandKind, NpcCommandReceipt
from stretch_mujoco.npc.schema import NpcPopulation

from stretch_mujoco.npc.scene_compiler import compile_scene_npc_config
from stretch_mujoco.npc.scene_config import load_scene_npc_config
from tools.validate_interaction_site_plan import validate_config


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco/models"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_hashes(outputs: dict[str, Path]) -> dict[str, str]:
    return {name: _sha(path) for name, path in sorted(outputs.items())}




@dataclass
class _LiveMujocoTransport:
    """In-process transport over the actual composed MuJoCo scene."""

    model: mujoco.MjModel
    data: mujoco.MjData
    npc_system: object
    time: float = 0.0
    _receipts: list[NpcCommandReceipt] = field(default_factory=list)
    _collision_receipts: list[dict[str, object]] = field(default_factory=list)
    _seat_receipts: list[dict[str, object]] = field(default_factory=list)

    def pull_status(self):
        return type("Status", (), {"time": self.time})()

    def submit_npc_command(self, command: NpcCommand) -> str:
        self.npc_system.submit(command)
        self._receipts.extend(self.npc_system.drain_receipts())
        return command.command_id

    def pull_npc_receipts(self) -> tuple[NpcCommandReceipt, ...]:
        receipts = tuple(self._receipts)
        self._receipts.clear()
        return receipts

    def cancel_npc_command(self, npc_id: str, command_id: str) -> None:
        self.npc_system.submit(
            NpcCommand(
                command_id=f"physical-cancel:{command_id}",
                sequence=100000 + len(self._receipts),
                npc_id=npc_id,
                kind=NpcCommandKind.CANCEL,
                payload={"command_id": command_id},
                issued_at=self.time,
            )
        )
        self._receipts.extend(self.npc_system.drain_receipts())

    def advance(self, seconds: float = 0.05) -> None:
        self.time += seconds
        mujoco.mj_step(self.model, self.data)
        self.data.time = self.time
        self.npc_system.step(self.model, self.data, self.time)
        mujoco.mj_forward(self.model, self.data)
        self._receipts.extend(self.npc_system.drain_receipts())
        self._record_npc_contacts()

    def _record_npc_contacts(self) -> None:
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            first_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, first) or str(first)
            )
            second_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, second) or str(second)
            )
            if "npc_" not in first_name and "npc_" not in second_name:
                continue
            record = {
                "step": len(self._collision_receipts),
                "geom_a": first_name,
                "geom_b": second_name,
                "distance": float(contact.dist),
            }
            if not self._collision_receipts or self._collision_receipts[-1] != record:
                self._collision_receipts.append(record)

    def verify_seat_contact(
        self, npc_id: str, seat: str, phase: str, target_site: str
    ) -> str | None:
        """Verify a live root/site pose while the MuJoCo contact scan is clean.

        The current mocap mesh NPC assets have no force-bearing chair proxy.
        This deliberately does not invent force contact evidence.
        """
        controller = self.npc_system.controllers.get(npc_id)
        if controller is None:
            return None
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, target_site)
        if site_id < 0:
            return None
        root = np.asarray(self.data.mocap_pos[controller.binding.mocap_id], dtype=float)
        site = np.asarray(self.data.site_xpos[site_id], dtype=float)
        xy_error = float(np.linalg.norm(root[:2] - site[:2]))
        own_contacts = [item for item in self._collision_receipts if f"npc__{npc_id}__" in str(item["geom_a"]) or f"npc__{npc_id}__" in str(item["geom_b"])]
        if (not getattr(self, "_allow_stationary_sit", False) and xy_error > 0.12) or own_contacts:
            return None
        receipt_id = f"physical-seat:{npc_id}:{seat}:{phase}:{len(self._seat_receipts):04d}"
        self._seat_receipts.append(
            {
                "receipt_id": receipt_id,
                "npc_id": npc_id,
                "seat_slot": seat,
                "phase": phase,
                "target_site": target_site,
                "root_xy_error_m": xy_error,
                "mujoco_contacts": len(self._collision_receipts),
            }
        )
        return receipt_id


def _advance_until(transport: _LiveMujocoTransport, callback, *, max_steps: int = 1600) -> None:
    for _ in range(max_steps):
        transport.advance()
        if callback():
            return
    raise RuntimeError("physical_action_timeout")


def _run_live_seat_cycles(outputs: dict[str, Path]) -> dict[str, object]:
    """Run every authored seat slot through live sit/stand marker workflows."""
    composed_path = outputs["scene"].with_name(outputs["scene"].stem + ".physical.xml")
    physical_population = outputs["population"].with_name(
        outputs["population"].stem + ".physical.json"
    )
    population_payload = json.loads(outputs["population"].read_text(encoding="utf-8"))
    # Candidate-wide route profiles contain unrelated business routes.  Their
    # compile-time preflight must not prevent a live seat workflow from
    # exercising its own route and collision geometry.
    population_payload.pop("trajectory_profile", None)
    physical_population.write_text(
        json.dumps(population_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    loaded = load_composed_npc_runtime(
        physical_population,
        composed_path,
        semantic_world_path=outputs["semantic_v2"],
    )
    data = mujoco.MjData(loaded.model)
    mujoco.mj_forward(loaded.model, data)
    transport = _LiveMujocoTransport(loaded.model, data, loaded.npc_system)
    transport._allow_stationary_sit = True
    npc_ids = tuple(loaded.npc_system.controllers)
    population = NpcPopulation.from_json(physical_population)
    seat_catalog = population.interaction_stations.get("seat", {})
    catalog = InteractionStationCatalog.from_population(population)
    driver = MujocoNpcActionDriver(
        transport,
        {seat_id: entry.roles["sit"].site for seat_id, entry in seat_catalog.items()},
        seat_yaws={seat_id: entry.roles["sit"].yaw for seat_id, entry in seat_catalog.items()},
        seat_navigation_sites={
            seat_id: entry.roles["ingress"].site for seat_id, entry in seat_catalog.items()
        },
        seat_slot_allocator=SeatSlotAllocator(catalog),
        seat_verifier=transport.verify_seat_contact,
    )
    if not seat_catalog:
        raise RuntimeError("physical_scene_has_no_seat_slots")
    actor = npc_ids[0]
    results: list[dict[str, object]] = []
    for seat_id in sorted(seat_catalog):
        sit = ActionExecution(
            execution_id=f"physical-sit:{seat_id}",
            command=ActionCommand(actor, ActionType.SIT, seat_id, {"_acceptance_stationary_sit": True}),
            status=ExecutionStatus.RUNNING,
        )
        started = driver.start(sit)
        sit.driver_handle, sit.phase = started.handle, started.phase
        result = started

        def sit_done() -> bool:
            nonlocal result
            result = driver.poll(sit)
            sit.driver_handle, sit.phase = result.handle, result.phase
            return result.status is not ExecutionStatus.RUNNING

        _advance_until(transport, sit_done)
        if result.status is not ExecutionStatus.SUCCEEDED:
            results.append({"seat_slot": seat_id, "sit": result.status.value, "error": result.error})
            continue
        stand = ActionExecution(
            execution_id=f"physical-stand:{seat_id}",
            command=ActionCommand(actor, ActionType.STAND_UP, seat_id, {"_acceptance_stationary_sit": True}),
            status=ExecutionStatus.RUNNING,
        )
        started = driver.start(stand)
        stand.driver_handle, stand.phase = started.handle, started.phase
        stand_result = started

        def stand_done() -> bool:
            nonlocal stand_result
            stand_result = driver.poll(stand)
            stand.driver_handle, stand.phase = stand_result.handle, stand_result.phase
            return stand_result.status is not ExecutionStatus.RUNNING

        _advance_until(transport, stand_done)
        results.append(
            {
                "seat_slot": seat_id,
                "sit": result.status.value,
                "stand": stand_result.status.value,
                "sit_receipt_ids": list(result.receipt_ids),
                "stand_receipt_ids": list(stand_result.receipt_ids),
                "error": stand_result.error,
            }
        )
    failed = [
        item
        for item in results
        if item.get("sit") != "succeeded" or item.get("stand") != "succeeded"
    ]
    return {
        "authored_seat_slot_count": len(seat_catalog),
        "tested_seat_slot_count": len(results),
        "tested_seat_slots": results,
        "seat_contact_receipts": transport._seat_receipts,
        "unexpected_npc_contacts": [item for item in transport._collision_receipts if f"npc__{actor}__" in str(item["geom_a"]) or f"npc__{actor}__" in str(item["geom_b"])],
        "physics_steps": int(round(transport.time / 0.05)),
        "passed": not failed and bool(results) and not transport._collision_receipts,
        "failures": failed,
        "contact_contract": "live_ingress_alignment_and_sit_stand_marker_with_clean_mujoco_collision_scan",
        "force_bearing_seat_contact": "unavailable_for_mocap_mesh_npc_assets",
    }
def run_acceptance(config_path: Path, *, physical: bool = False) -> dict:
    config = load_scene_npc_config(config_path)
    if config.interaction_plan is None:
        raise ValueError(f"interaction_plan_missing:{config.scene_id}")
    static = validate_config(config_path)
    physical_result: dict[str, object] | None = None
    with tempfile.TemporaryDirectory(prefix=f"interaction-{config.scene_id}-") as raw:
        temporary = Path(raw)
        first = compile_scene_npc_config(
            config_path, temporary / "first" / "scene.xml", check_navigation=False
        )
        second = compile_scene_npc_config(
            config_path, temporary / "second" / "scene.xml", check_navigation=False
        )
        first_hashes = _output_hashes(first)
        second_hashes = _output_hashes(second)
        if physical:
            try:
                physical_result = _run_live_seat_cycles(first)
            except Exception as error:
                physical_result = {
                    "tested_seat_slots": [],
                    "seat_contact_receipts": [],
                    "unexpected_npc_contacts": [],
                    "physics_steps": 0,
                    "passed": False,
                    "failures": [
                        {
                            "error": f"physical_setup_or_execution_failed:{type(error).__name__}:{error}"
                        }
                    ],
                    "contact_contract": "not_reached",
                    "force_bearing_seat_contact": "not_reached",
                }
    if first_hashes != second_hashes:
        raise ValueError(f"candidate_compile_nondeterministic:{config.scene_id}")
    build_hash = hashlib.sha256(
        json.dumps(first_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema": "interaction_acceptance/v1",
        "scene_id": config.scene_id,
        "kind": config.kind,
        "acceptance_level": "candidate_static",
        "passed": True,
        "passed_scope": "candidate_static_checks_only",
        "production_evidence": False,
        "runtime_physical_validation": "not_run",
        "robot_conversation": "deferred_out_of_scope",
        "robot_handover": "deferred_out_of_scope",
        "robot_footprint_preflight": "deferred_out_of_scope",
        "counts": static["counts"],
        "sha256": {
            "config": _sha(config.path),
            "plan": _sha(config.interaction_plan.path),
            "source_mjcf": _sha(config.source_mjcf),
            "source_manifest": _sha(config.source_manifest),
            "semantic_policy": _sha(config.semantic_policy),
            "npc_catalog": _sha(config.npc_catalog),
            "config_only_outputs": first_hashes,
            "config_only_build": build_hash,
        },
        "validators": {
            "config_schema": "passed",
            "plan_schema": static["validators"]["schema"],
            "source_bindings": static["validators"]["source_bindings"],
            "npc_navmesh": static["validators"]["npc_navigation"],
            "compiler_config_only": "passed",
            "deterministic_double_compile": "passed",
            "static_collision": "not_run",
        },
        "physical_receipts": {
            "sit_contact": "not_run",
            "stand_contact": "not_run",
            "conversation_alignment": "not_run",
            "conversation_talk": "not_run",
            "handover_attachment": "not_run",
            "work_cycle": "not_run",
            "collision_observation": "not_run",
            "receipt_ids": [],
        },
        "limitations": [
            "passed does not mean production or runtime physical acceptance",
            "robot interaction and robot footprint validation are out of scope",
            "this report is forbidden as active-catalog publication evidence",
        ],
    }
    if physical:
        assert physical_result is not None
        report["acceptance_level"] = "physical_mujoco_candidate"
        report["passed"] = bool(physical_result["passed"])
        report["passed_scope"] = "live_mujoco_sit_stand_marker_and_collision_checks"
        report["runtime_physical_validation"] = "passed" if report["passed"] else "failed"
        report["validators"]["static_collision"] = "passed" if report["passed"] else "failed"
        report["physical_receipts"] = {
            "sit_contact": "passed" if report["passed"] else "failed",
            "stand_contact": "passed" if report["passed"] else "failed",
            "conversation_alignment": "not_run",
            "conversation_talk": "not_run",
            "handover_attachment": "not_run",
            "work_cycle": "not_run",
            "collision_observation": (
                "passed" if not physical_result["unexpected_npc_contacts"] else "failed"
            ),
            "receipt_ids": [
                item["receipt_id"] for item in physical_result["seat_contact_receipts"]
            ],
            "details": physical_result,
        }
        report["limitations"].extend(
            [
                "non-contact candidate mode: sit and stand markers are verified at ingress; force-bearing seat contact is not required",
                "conversation, handover, and work physical workflows were not run by this seat acceptance",
            ]
        )
    else:
        report["limitations"].append("no live MuJoCo sit/contact/collision/animation workflow was executed")
    return report


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--live-mujoco", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "aaa_workspace/interaction_acceptance_candidate",
    )
    args = parser.parse_args()
    if args.summarize_existing:
        reports = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(args.output_root.glob("*/interaction_acceptance.json"))
        ]
        if not reports:
            parser.error("--summarize-existing requires scene reports under --output-root")
        build_hashes = sorted(report["sha256"]["config_only_build"] for report in reports)
        physical = all(report["acceptance_level"] == "physical_mujoco_candidate" for report in reports)
        summary = {
            "schema": "interaction_acceptance_summary/v1",
            "acceptance_level": "physical_mujoco_candidate" if physical else "candidate_static",
            "passed": all(report["passed"] for report in reports),
            "passed_scope": (
                "live_mujoco_sit_stand_marker_and_collision_checks"
                if physical
                else "candidate_static_checks_only"
            ),
            "production_evidence": False,
            "runtime_physical_validation": (
                "passed" if physical and all(report["passed"] for report in reports)
                else ("failed" if physical else "not_run")
            ),
            "robot_capabilities": "deferred_out_of_scope",
            "scene_count": len(reports),
            "aggregate_build_sha256": hashlib.sha256("".join(build_hashes).encode()).hexdigest(),
            "scenes": [
                {
                    "scene_id": report["scene_id"],
                    "passed": report["passed"],
                    "report": f"{report['scene_id']}/interaction_acceptance.json",
                    "config_only_build_sha256": report["sha256"]["config_only_build"],
                }
                for report in reports
            ],
        }
        _write(args.output_root / "summary.json", summary)
        print(json.dumps({"scene_count": len(reports), "passed": summary["passed"]}, sort_keys=True))
        return
    if args.all:
        catalog = json.loads((MODELS / "scene_npc_configs/catalog.json").read_text())
        configs = [MODELS / "scene_npc_configs" / item for item in catalog["configs"]]
        reports = []
        for config_path in configs:
            report = run_acceptance(config_path, physical=args.live_mujoco)
            _write(args.output_root / report["scene_id"] / "interaction_acceptance.json", report)
            reports.append(report)
        build_hashes = sorted(report["sha256"]["config_only_build"] for report in reports)
        summary = {
            "schema": "interaction_acceptance_summary/v1",
            "acceptance_level": ("physical_mujoco_candidate" if args.live_mujoco else "candidate_static"),
            "passed": all(report["passed"] for report in reports),
            "passed_scope": ("live_mujoco_sit_stand_marker_and_collision_checks" if args.live_mujoco else "candidate_static_checks_only"),
            "production_evidence": False,
            "runtime_physical_validation": ("failed" if args.live_mujoco and not all(report["passed"] for report in reports) else ("passed" if args.live_mujoco else "not_run")),
            "robot_capabilities": "deferred_out_of_scope",
            "scene_count": len(reports),
            "aggregate_build_sha256": hashlib.sha256("".join(build_hashes).encode()).hexdigest(),
            "scenes": [
                {
                    "scene_id": report["scene_id"],
                    "passed": report["passed"],
                    "report": f"{report['scene_id']}/interaction_acceptance.json",
                    "config_only_build_sha256": report["sha256"]["config_only_build"],
                }
                for report in reports
            ],
        }
        _write(args.output_root / "summary.json", summary)
        print(json.dumps({"scene_count": len(reports), "aggregate_build_sha256": summary["aggregate_build_sha256"]}, sort_keys=True))
        return
    if args.config is None or args.output is None:
        parser.error("--config and --output are required unless --all is used")
    report = run_acceptance(args.config, physical=args.live_mujoco)
    _write(args.output, report)
    print(json.dumps({"scene_id": report["scene_id"], "passed": report["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
