import json
import zipfile
from pathlib import Path

from stretch_mujoco.humanoid.babel_intake import (
    babel_feature_to_bmlmovi_stageii,
    find_babel_candidates,
    write_babel_candidates,
)


def _write_index(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source_id": "BMLmovi/Subject_1_F_MoSh/Subject_1_F_4_stageii.npz",
                "frame_count": 100,
                "mocap_frame_rate": 10.0,
                "usable": True,
            }
        )
        + "\n"
    )


def _write_babel_archive(path: Path) -> None:
    record = {
        "1": {
            "feat_p": "BMLmovi/BMLmovi/Subject_1_F_MoSh/Subject_1_F_4_poses.npz",
            "frame_ann": {
                "labels": [
                    {
                        "raw_label": "walking forward",
                        "proc_label": "walk forward",
                        "act_cat": ["walk", "forward movement"],
                        "start_t": 0.15,
                        "end_t": 1.15,
                    },
                    {
                        "raw_label": "unclear movement",
                        "proc_label": "gesture",
                        "start_t": 2.0,
                        "end_t": 3.0,
                    },
                ]
            },
        }
    }
    with zipfile.ZipFile(path, "w") as archive:
        for split in ("train", "val", "test"):
            archive.writestr(
                f"babel_v1.0_release/{split}.json", json.dumps(record if split == "train" else {})
            )


def test_babel_feature_mapping_requires_expected_bmlmovi_layout() -> None:
    assert (
        babel_feature_to_bmlmovi_stageii("BMLmovi/BMLmovi/Subject_1_F_MoSh/Subject_1_F_4_poses.npz")
        == "BMLmovi/Subject_1_F_MoSh/Subject_1_F_4_stageii.npz"
    )
    assert babel_feature_to_bmlmovi_stageii("CMU/CMU/1/foo_poses.npz") is None


def test_find_babel_candidates_requires_exact_labels_and_local_motion(tmp_path: Path) -> None:
    index = tmp_path / "candidates.jsonl"
    archive = tmp_path / "babel.zip"
    _write_index(index)
    _write_babel_archive(archive)

    (candidate,) = find_babel_candidates(archive, index, per_clip=3, min_duration_seconds=0.5)

    assert candidate.target_clip == "walk"
    assert candidate.source_frame_range == (1, 12)
    assert candidate.babel_categories == ("walk", "forward movement")


def test_write_babel_candidates_records_missing_contract_clips(tmp_path: Path) -> None:
    index = tmp_path / "candidates.jsonl"
    archive = tmp_path / "babel.zip"
    _write_index(index)
    _write_babel_archive(archive)
    candidates = find_babel_candidates(archive, index, per_clip=1, min_duration_seconds=0.5)

    output = write_babel_candidates(tmp_path / "pending.json", archive, index, candidates)

    document = json.loads(output.read_text())
    assert document["review_status"] == "pending_visual_review"
    assert document["selection_policy"] == "exact_babel_frame_label_only"
    assert "walk" not in document["missing_office_clips"]
    assert "use_computer" in document["missing_office_clips"]
