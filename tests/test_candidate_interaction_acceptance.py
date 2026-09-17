import json
from pathlib import Path

import pytest

from tools.audit_active_scene_semantics import require_production_acceptance
from tools.run_scene_interaction_acceptance import run_acceptance


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "stretch_mujoco/models/scene_npc_configs/office/office_01_linear_bench.json"


def test_single_scene_report_is_explicitly_candidate_static() -> None:
    report = run_acceptance(CONFIG)
    assert report["acceptance_level"] == "candidate_static"
    assert report["passed"] is True
    assert report["passed_scope"] == "candidate_static_checks_only"
    assert report["production_evidence"] is False
    assert report["runtime_physical_validation"] == "not_run"
    assert report["robot_handover"] == "deferred_out_of_scope"
    assert report["physical_receipts"]["receipt_ids"] == []
    assert set(report["sha256"]["config_only_outputs"])


def test_candidate_static_summary_cannot_publish_active_catalog(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "acceptance_level": "candidate_static",
                "passed": True,
                "production_evidence": False,
                "runtime_physical_validation": "not_run",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError, match="^active_publish_requires_production_physical_acceptance$"
    ):
        require_production_acceptance(summary)
