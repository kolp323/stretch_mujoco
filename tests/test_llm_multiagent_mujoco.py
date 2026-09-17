import math
from pathlib import Path

from examples.llm_multiagent_mujoco_benchmark import run_benchmark


def test_active_office_three_npc_async_llm_benchmark(tmp_path: Path) -> None:
    report = run_benchmark(tmp_path / "report.json", provider_latency=0.02)

    assert report["passed"], report
    assert report["clock"]["start"]["minute_of_day"] == 9 * 60
    assert math.isclose(report["clock"]["end"]["minute_of_day"], 18 * 60, abs_tol=1e-6)
    assert report["clock"]["end"]["day"] == report["clock"]["start"]["day"]
    assert len(report["agents"]) == 3
    sessions = {item["agent_id"]: item["session_id"] for item in report["llm"]}
    assert len(sessions) == 3 and len(set(sessions.values())) == 3

    actions = {item["execution_id"]: item for item in report["actions"]}
    successful = [item for item in report["llm"] if item["status"] == "succeeded"
                  and "action" in item.get("response", {})]
    assert successful
    for decision in successful:
        assert decision["request_id"] and decision["correlation_id"] and decision["session_id"]
        assert decision["execution_id"] in actions
        action = next(
            item for item in report["actions"]
            if item["action"] == decision["payload_action"]
            and decision["request_id"] in item["causal_request_ids"]
        )
        assert decision["request_id"] in action["causal_request_ids"]
        assert decision["correlation_id"] in action["causal_correlations"]
        assert decision["session_id"] in action["causal_sessions"]
        assert decision["payload_action"] == action["action"]

    names = {item["action"] for item in report["actions"]}
    assert "work" in names
    assert "drink" in names and report["activities"]["drink"]["consumed"]
    assert report["activities"]["rest"]["logical_activity"] == "rest"
    assert set(report["activities"]["rest"]["physical_actions"]) == {"sit", "idle", "stand_up"}
    assert {"sit", "idle", "stand_up"} <= names
    assert {item.get("recovery") for item in report["llm"]} >= {"replan", "safe_idle"}
    assert report["physics_steps_during_llm_wait"] > 0

    policy = report["collision_policy"]
    assert policy["source"]
    assert policy["minimum_center_separation_m"] == 2 * (
        policy["agent_radius_m"] + policy["clearance_m"]
    )
    assert report["collision_free"]
    assert report["min_npc_separation_m"] >= policy["minimum_center_separation_m"]

    for stage in ("conversation_approach", "conversation_align", "talk"):
        receipts = report["conversation_receipts"][stage]
        assert receipts and all(item["status"] == "succeeded" for item in receipts)

    receipt_status = {item["command_id"]: item["status"]
                      for item in report["terminal_command_receipts"]}
    assert not report["orphans"]
    assert report["semantic_to_receipt"]
    for trace in report["semantic_to_receipt"]:
        assert trace["receipt_ids"]
        assert all(receipt_status[item] == "succeeded" for item in trace["receipt_ids"])

    assert report["active_build_id"]
    assert report["scene_id"] == "office_01_linear_bench"
    assert report["fixture"]["used"]
    assert report["fixture"]["active_source_conversation_capability"]["status"] == "unsupported"
    assert report["fixture"]["model_validated_sites"]
    assert {"events", "llm", "actions", "collisions", "failures"} <= report.keys()
