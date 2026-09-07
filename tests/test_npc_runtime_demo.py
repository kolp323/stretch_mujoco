import mujoco

from stretch_mujoco.npc.system import NpcSystem
from tools.render_npc_runtime_demo import run_move_trace


def _model() -> mujoco.MjModel:
    frames = "\n".join(
        f'<geom name="npc__employee_01__clip__walk__frame__{index:03d}__slot__body" '
        'type="sphere" size=".1" rgba="1 1 1 0"/>'
        for index in range(8)
    )
    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <body name="npc__employee_01" mocap="true">
              <geom name="npc__employee_01__clip__idle__frame__000__slot__body"
                    type="sphere" size=".1"/>
              {frames}
            </body>
            <site name="destination" pos=".2 0 0"/>
          </worldbody>
        </mujoco>
        """
    )


def test_move_trace_records_marker_gated_root_motion() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    trace = run_move_trace(
        NpcSystem.from_model(model),
        model,
        data,
        npc_id="employee_01",
        target_site="destination",
        seconds=4.0,
        fps=16,
    )

    assert trace.first_foot_marker in {"left_foot", "right_foot"}
    assert trace.marker_gated is True
    assert trace.stop_marker in {"left_foot", "right_foot"}
    assert trace.terminal_status == "succeeded"
