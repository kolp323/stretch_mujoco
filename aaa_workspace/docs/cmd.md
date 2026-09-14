uv run examples/office_multi_npc_day.py \
  --end 18 \
  --step 0.25 \
  --mp4-fps 10 \
  --mp4 outputs/npc_week1_visual_api.mp4 \
  --output outputs/npc_week1_visual_report.json

uv run examples/office_multi_npc_day.py \
  --end 18 --snapshot-dir outputs/day_3d

MUJOCO_GL=egl uv run tools/render_mujoco_video.py \
  --input outputs/day_3d/office_snapshots.jsonl \
  --output outputs/day_3d.mp4

| Week 1 项目 | 当前状态 | 你应如何理解 |
|---|---|---|
| 拉取项目、配置环境 | 已具备可运行环境 | 当前已在 `feat/npc-system`，`uv` 环境可运行；我没有重拉代码或改分支。 |
| 熟悉代码 | 已完成 | 已明确 Agent 负责决策，Runtime 负责校验/状态，Semantics 是世界真相源，动画和机器人是执行层。 |
| 加入多个不同外观任务模型 | 部分完成 | Scene 2 有 2 个实际 NPC 实体与不同材质；本地有 3 套纹理变体，但“10 个可复现、可提交外观”尚未完成。 |
| 单 NPC 与环境交互、全天工作 | 已完成逻辑验收 | 单 NPC 可按日程、需求和效用决策执行工作、移动、休息、拿取、请求机器人等；新增了忙碌、关注目标、失败原因和动画状态。 |
| NPC 与 mock 机器人交互 | 已完成 | 已有 mock robot 的任务生命周期：请求 → pending → running → 成功/失败；成功更新语义位置，失败释放资源预留。 |


```bash
cd <appearance-optimization-worktree>

# 可视化调整
.venv/bin/python tools/tune_npc_obj_accessory.py \
  --runtime-config stretch_mujoco/models/accessories/beautiful_hair_v1.runtime.json

# 烘焙融合 obj
.venv/bin/python tools/build_npc_scene.py \
  --accessory-runtime-config \
  stretch_mujoco/models/accessories/beautiful_hair_v1.runtime.json \
  --output stretch_mujoco/models/.npc_accessory_tuned_office.xml \
  --include-base-scene

# 在特定动作下调整
.venv/bin/python tools/tune_npc_obj_accessory.py \
  --runtime-config stretch_mujoco/models/accessories/beautiful_hair_v1.runtime.json \
  --reference-clip walk \
  --reference-frame 7 \
  --full-body

```
