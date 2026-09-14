from pathlib import Path


def test_office_llm_template_has_no_real_key():
    content = (
        Path(__file__).resolve().parents[1] / "stretch_mujoco/models/office_llm.example.json"
    ).read_text()
    assert '"api_key": "PASTE_API_KEY_HERE"' in content
    assert '"api_key": "sk-' not in content
