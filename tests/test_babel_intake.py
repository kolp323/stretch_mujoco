import json
import zipfile
from pathlib import Path

from stretch_mujoco.humanoid.babel_intake import (
    babel_feature_to_amass_stageii,
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
    assert babel_feature_to_amass_stageii("CMU/CMU/1/foo_poses.npz") == "CMU/1/foo_stageii.npz"
    assert (
        babel_feature_to_amass_stageii(
            "Transitionsmocap/Transitions_mocap/mazen_c3d/sit_stand_poses.npz"
        )
        == "Transitions/mazen_c3d/sit_stand_stageii.npz"
    )
    assert babel_feature_to_amass_stageii("CMU/Other/1/foo_poses.npz") is None
    assert babel_feature_to_amass_stageii("CMU/CMU/../foo_poses.npz") is None


def test_find_babel_candidates_requires_exact_labels_and_local_motion(tmp_path: Path) -> None:
    index = tmp_path / "candidates.jsonl"
    archive = tmp_path / "babel.zip"
    _write_index(index)
    _write_babel_archive(archive)

    (candidate,) = find_babel_candidates(archive, index, per_clip=3, min_duration_seconds=0.5)

    assert candidate.target_clip == "walk"
    assert candidate.source_frame_range == (1, 12)
    assert candidate.babel_categories == ("walk", "forward movement")


def test_find_babel_candidates_maps_cmu_and_normalizes_annotation_spacing(tmp_path: Path) -> None:
    index = tmp_path / "candidates.jsonl"
    archive = tmp_path / "babel.zip"
    index.write_text(
        json.dumps(
            {
                "source_id": "CMU/15/15_12_stageii.npz",
                "frame_count": 2000,
                "mocap_frame_rate": 30.0,
                "usable": True,
            }
        )
        + "\n"
    )
    record = {
        "9": {
            "feat_p": "CMU/CMU/15/15_12_poses.npz",
            "frame_ann": {
                "labels": [
                    {
                        "raw_label": "give cards",
                        "proc_label": "give deck cards   with right hand",
                        "act_cat": ["give"],
                        "start_t": 1.0,
                        "end_t": 2.0,
                    }
                ]
            },
        }
    }
    with zipfile.ZipFile(archive, "w") as zipped:
        for split in ("train", "val", "test"):
            zipped.writestr(
                f"babel_v1.0_release/{split}.json", json.dumps(record if split == "train" else {})
            )

    (candidate,) = find_babel_candidates(archive, index, per_clip=1, min_duration_seconds=0.5)

    assert candidate.target_clip == "give"
    assert candidate.source_id == "CMU/15/15_12_stageii.npz"
    assert candidate.babel_proc_label == "give deck cards with right hand"


def test_find_babel_candidates_prioritizes_explicit_chair_sit_down(tmp_path: Path) -> None:
    index = tmp_path / "candidates.jsonl"
    archive = tmp_path / "babel.zip"
    index.write_text(
        "\n".join(
            json.dumps(
                {
                    "source_id": f"BMLmovi/Subject_{subject}_F_MoSh/Subject_{subject}_F_4_stageii.npz",
                    "frame_count": 200,
                    "mocap_frame_rate": 10.0,
                    "usable": True,
                }
            )
            for subject in (1, 2)
        )
        + "\n"
    )
    records = {
        "generic": {
            "feat_p": "BMLmovi/BMLmovi/Subject_1_F_MoSh/Subject_1_F_4_poses.npz",
            "frame_ann": {
                "labels": [
                    {"proc_label": "sit down", "start_t": 0.0, "end_t": 10.0},
                ]
            },
        },
        "chair": {
            "feat_p": "BMLmovi/BMLmovi/Subject_2_F_MoSh/Subject_2_F_4_poses.npz",
            "frame_ann": {
                "labels": [
                    {"proc_label": "sit down in chair", "start_t": 0.0, "end_t": 2.0},
                ]
            },
        },
    }
    with zipfile.ZipFile(archive, "w") as zipped:
        for split in ("train", "val", "test"):
            zipped.writestr(
                f"babel_v1.0_release/{split}.json", json.dumps(records if split == "train" else {})
            )

    (candidate,) = find_babel_candidates(archive, index, per_clip=1, min_duration_seconds=0.5)

    assert candidate.target_clip == "sit_down"
    assert candidate.babel_proc_label == "sit down in chair"


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
