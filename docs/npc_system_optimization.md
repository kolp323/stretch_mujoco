# NPC 系统优化说明（历史记录）

> 此文记录了 NPC 改造开始时的目标和差距，部分“当前版本”描述已经被后续的
> population、asset manifest、动画回执和会话实现取代。使用和配置请以
> [NPC 系统指南](npc_system.md) 为准；集成迁移的证据记录见
> `aaa_workspace/docs/current.md`。

## 1. 当前版本效果

当前 NPC 功能分为“可视化 NPC”和“逻辑员工 Agent”两层：

- humanoid 提供 SMPL-X 网格、预烘焙动作和导航，已有 idle、walk、sit、work、eat 等动作帧，坐下前会规划无碰撞接近点。
- agents 提供确定性的员工运行时。每个员工拥有 profile、needs、schedule、memory、perception、planner、executor 和 EmployeeState。
- semantics 维护对象、位置、所有权、权限和请求关系；MuJoCo 负责几何与真实位姿，语义图负责身份和任务状态。
- OfficeAgentRuntime 使用封闭动作集合（move_to、work、request_robot、handover、attend_meeting 等），执行前检查目标、位置、感知、资源冲突和访问权限。
- LLM 是可选高层建议器，响应必须经过 JSON 校验和运行时验证；已有事件触发的 dialogue 请求及机器人取放物品链路。
- Scene 2 的 office_scene2_multi_npc.xml 已示范多个员工实体和 3 套 NPC 纹理材质；tools/generate_npc_textures.py 可从基础 UV 贴图生成外观变体。

## 2. 当前版本使用方式

### 2.1 查看单个 NPC 动作

    uv run examples/office_scene.py
    uv run examples/office_scene.py --animation-demo
    uv run examples/office_scene.py --npc-animation sit --npc-chair chair_left

--headless 可只编译场景而不打开窗口。入口为 examples/office_scene.py，场景为 stretch_mujoco/models/office_scene.xml。

### 2.2 运行员工 Agent

    uv run examples/office_agents.py --complete-task
    uv run examples/office_autonomy.py --seconds 60
    uv run examples/office_multi_npc_day.py --end 18 --step 0.25 --chat-log

配置文件为 stretch_mujoco/models/office_agents.json、office_scene2_agents.json；语义配置为对应的 office_semantics.json 或 office_scene2_semantics.json。需要 LLM 时从 office_llm.example.json 复制为本地忽略文件并配置凭据。

### 2.3 生成外观与动画

    uv run tools/generate_npc_textures.py
    uv run tools/bake_smplx_animations.py

输出位于 stretch_mujoco/models/assets/humanoid/generated/animations/，由 smplx_humanoid_assets.xml 或场景 XML 引用。SMPL/SMPL-X 模型和 UV 文件属于本地授权资产，不应提交公共仓库。

## 3. 主要问题

1. 动作由离散 OBJ 帧切换，缺少速度曲线、过渡、转身、手臂指向、注视和说话口型；多人同时行动时容易僵硬。
2. 纹理变体主要是全局颜色变换，不能稳定表达发型、服装款式、配饰和体型；NPC 增加后需手工编辑大量 XML geom。
3. 对话目前是事件/LLM 结果字段，缺少参与者、距离、轮次、打断、超时、字幕和语音接口；NPC-NPC 与 NPC-机器人没有统一协议。
4. EmployeeState 缺少社交关系、注意力、当前会话、可用性、受阻原因和动作失败恢复状态。
5. 逻辑位置和渲染/物理位姿存在双重来源风险；导航或交互失败时状态、语义关系和动画可能不同步。
6. 缺少多人并发、对话、外观加载、状态迁移和长时间稳定性的自动化测试与可观测性。

## 4. 优化任务

### A. 动作与动画

- 定义 AnimationState（clip、phase、speed、loop、blend、upper_body_overlay）和动作生命周期：requested → navigating → aligning → playing → completed/failed。
- 在 mesh_animator.py 增加交叉淡化、播放速度、随机起始相位和打断策略；导航朝向与脚步保持一致。
- 增加 talk、gesture_point、gesture_wave、use_computer、pick_up、handover 片段；说话叠加头部注视和上半身手势。
- 所有动作必须有目标 site、完成条件和超时；失败时回到 idle 或重规划。

### B. 外观与皮肤

- 设计 NpcAppearance 配置（skin、hair、top、bottom、shoes、accessories、scale），由 JSON 绑定 agent_id，渲染层按配置加载材质，避免复制 XML。
- 保留基础 UV 兼容性，同时支持分区材质或贴图图层；纹理脚本使用可复现 seed，并输出缩略图/元数据清单。
- 每个 NPC 做正面/侧面/坐姿验收，检查遮挡、颜色稳定、透明帧无闪烁和缺失资产报错。

### C. NPC、机器人和 NPC-NPC 交互

- 新增 ConversationSession：session_id、participants、topic、turn、started_at、timeout、status、transcript、interrupt_policy。
- 用语义距离和朝向判定能否开口；会话期间锁定参与者注意力，结束或超时后恢复原计划。
- 机器人对话支持 request、clarify、acknowledge、handover_confirm；NPC-NPC 支持问候、进度询问、会议邀请和冲突解决。
- LLM 只生成候选文本/意图，运行时负责敏感信息过滤、长度限制、冷却和回退模板；对话写入事件日志与 AgentMemory。

### D. 状态与调度

- 扩展 EmployeeState：availability、attention_target、conversation_id、social_energy、stress、blocked_reason、last_failure、animation_state。
- 将需求、日程、社交事件和任务优先级统一为 utility 输入，支持计划抢占、恢复和防重复决策。
- 语义关系更新与动作完成使用同一事件 ID；物理控制器必须回报成功、失败或超时，禁止逻辑层默认成功。

### E. 测试与交付

- 单元测试覆盖状态迁移、会话超时、权限、记忆容量、动作校验和确定性 seed。
- 集成测试至少 3 个 NPC + Stretch 同时导航、对话、请求物品和 handover，验证无死锁、重复占用和事件乱序。
- 回归测试 headless 编译 NPC 场景，检查材质/网格、透明帧、碰撞体和站立高度。
- 运行时输出每 NPC 的状态、动作、会话、失败原因和最近事件。

## 5. 分工与完成标准

同事负责 stretch_mujoco/humanoid/、stretch_mujoco/agents/、stretch_mujoco/semantics/ 及 NPC 配置/测试；场景 XML 只通过稳定接口或 fixture 协作。

完成标准：至少 10 个外观不同的 NPC；动作切换无明显跳帧；NPC-NPC 与 NPC-机器人对话均能开始、轮流、超时和恢复；失败可回退；headless 集成测试和一段带字幕的多 NPC 录屏通过。
