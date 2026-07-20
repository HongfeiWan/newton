# Newton Nero + L10 Diffusion Policy 环境

这套环境用于在 Newton/MJWarp 中并行执行瓶子抓取、搬运和释放任务。机械手必须抓稳瓶子、保持接触将其抬高至少 0.1 m、在 XY 平面移出初始位置、放回接近初始高度并保持初始姿态，最后释放并等待瓶子稳定。瓶子的初始位置目前固定；冻结 DP 仍只使用与 LeRobot 数据集对齐的图像、26 维状态和 19 维绝对动作。Residual PPO contract v9 另外读取五路实时指根电机负载，并仅向 privileged critic 提供 reward-v13 guidance；它不改变 DP checkpoint、数据集或 actor 输入协议。

## 数据协议

环境的 policy observation 使用与 `local_data/groot/smooth/meta/info.json` 相同的字段名：

| 字段 | 单帧形状 | dtype | 内容 |
| --- | ---: | --- | --- |
| `observation.state` | `[26]` | `float32` | `arm_joint_pos[7] + arm_eef_pos[3] + arm_eef_rot6d[6] + hand_joint_pos[10]` |
| `observation.images.ego_view` | `[180, 320, 3]` | `uint8` | node0 ROI 后的头部 RGB |
| `observation.images.wrist_view` | `[480, 640, 3]` | `uint8` | 腕部 RGB |
| `observation.finger_root_load` | `[5]` | `float32` | 可选；thumb/index/middle/ring/pinky 指根主动屈曲关节的归一化闭合方向负载，仅供 residual actor |
| `action` | `[19]` | `float32` | 绝对 EEF xyz + 绝对 EEF rotation-6D + 绝对手指目标 |

26 维 state 的严格顺序是：

```text
[arm_joint_pos(7), eef_xyz(3), eef_rotation_6d(6), hand_joint_pos(10)]
```

19 维 action 的后 10 维只包含 L10 主动关节。L10 URDF 中的 thumb MCP/IP 与四指 PIP/DIP 是带非单位倍率的 mimic follower，同时也拥有位置执行器；若只更新主动关节，follower 的旧 PD target 会与 leader 对抗。环境现在从公开 hand spec 建立所有 lane 的 GPU leader/follower 索引，每个 control step 在主动目标完成对应控制模式的解码和限位后，按 `follower = offset + multiplier * leader` 更新 follower target，并按 follower 自身限位裁剪、把 follower target velocity 置零。`pd_eef_pose_abs`、`pd_joint_pos`、`pd_joint_delta_pos` 以及 partial reset 都使用同一同步契约；数据集和 DP 仍保持原来的 10 维主动关节协议。

19 维 action 的严格顺序是：

```text
[eef_xyz_target(3), eef_rotation_6d_target(6), hand_joint_target(10)]
```

Rotation-6D 与 GR00T 核心协议一致，按旋转矩阵前两行保存：

```text
[r00, r01, r02, r10, r11, r12]
```

state EEF 和 action EEF 都使用 Nero/CAN 基座下的 flange pose。数据集加载时会强制检查 row-first
分量名、`rot6d_convention` 和 state-copy action metadata，不符合契约的数据会在训练开始前报错。修复后的行为克隆数据把每一帧的
`state[7:16]` 作为该帧绝对 EEF action target，因此环境把反归一化后的 XYZ 和旋转矩阵直接交给
GPU IK，不再重复应用旧 node0 command frame 的 `A·T·B` 变换：

```text
T_ik_target = T_action = T_state_target
```

`hold_action()` 同样直接输出当前 FK flange pose，保证 observation、action 和 IK 三端使用同一坐标系和
row-first Rot6D 协议。

DP 的 `state[7:10]` 和 `action[0:3]` 都是 right Nero/CAN 基座帧，而 residual actor 的前三维固定解释为
world XYZ。当前 URDF 的固定旋转满足 `p_world = R_world_from_action @ p_action`：

```text
R_world_from_action = [[0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0]]
```

因此 CAN X 是 world +Z。contract v9 先在 world frame 对 residual 施加水平/竖直尺度，再用
`R_world_from_action.T` 转回 CAN frame 与 DP base XYZ 相加，最后按 CAN action bounds 裁剪。这个矩阵完整写入
run config 和 checkpoint，resume 必须逐项匹配；未来 URDF 使用非轴置换固定旋转时不依赖某个硬编码 CAN 分量。

`env.step()` 接收的是已经反归一化的物理动作，不是 `[-1, 1]` 关节动作。DP 网络使用训练 split 的
min/max；训练输出会保存统计值及其 SHA-256，同时 split 指纹覆盖 Parquet 中实际的 state/action 数值，
修复数据后不能误用旧 checkpoint resume。

当前修复版按 `raw_episode_id` 审计每个 clip，并以同一行 `state[:7]` 的 Nero SDK/CAN MDH flange FK
作为物理旋转真值。只有 episode 20–42 的 `state[10:16]` 和 `action[3:9]` 做了矩阵转置后重新
row-first 编码；其他数值分量保持逐位不变。`info.json` 和每条 episode metadata 都带
`teleop_stack.rot6d_physical_truth_migration.v1` 证据。加载器还会扫描所有 Parquet，拒绝 NaN/Inf，
并验证每帧 `action == concat(state[7:16], state[16:26])`（`atol=1e-6`）。

`numeric_data_sha256` 对所有 episode 的逻辑 float32 state/action 内容计算，与 Parquet codec 或文件字节
无关；它不覆盖视频字节或图像预处理代码。视频缓存未因数值旋转修复而失效，因此继续复用。

## 图像链路

现有 GR00T 推理将 `ego_view` 和 `wrist_view` 作为两个独立 video modality，并没有先拼接图像。DP 因此也使用两个不共享权重的 ResNet-18 encoder：

```text
ego_view   1280x800 dataset frame
  -> node0 ROI: zoom=2.0, center=(0.50, 0.65)
  -> 320x180
  -> ego_encoder

wrist_view 640x480
  -> wrist_encoder
```

仿真相机直接用 ROI ray 生成 320×180 的 `ego_view`，所以 rollout 不再做 CPU crop/resize。训练数据通过 `teleop_stack.camera_preprocessing.preprocess_ego_rgb()` 使用与当前 GR00T simulator inference 相同的处理函数。两路 encoder 内部再在 GPU 上 resize 到网络输入尺寸。

## 绝对 EEF 动作和 GPU IK

默认控制模式为 `pd_eef_pose_abs`。每个控制周期执行：

1. 从 19 维 action 读取绝对命令帧 EEF 9D 和绝对手指 10D 目标。
2. 将同一 Nero/CAN 基座帧下的 flange XYZ 和 row-first Rot6D 直接作为 GPU IK 目标。
3. 在 CUDA Torch 上批量计算 Nero MDH FK 和 `[N, 6, 7]` 空间 Jacobian。
4. 对所有 world 批量执行 damped least-squares。
5. 将每周期臂关节步长限制为默认 `0.045 rad`，手指步长限制为 `0.08 rad`。
6. 直接写入 GPU 上的 Newton joint position/velocity target。

默认 IK 参数与现有 checkpoint-200000 推理控制器保持一致：位置步长 `0.03 m`、旋转步长 `5 deg`、position/orientation weight 为 `3/1`、damping 为 `0.02`。环境不对每个 world 做 `.numpy()` 或 CPU IK。

旧的 `pd_joint_pos` 和 `pd_joint_delta_pos` 仍可用于兼容测试，但它们是 17 维关节动作，不属于第一版 DP 协议。

## 环境接口

```python
from teleop_stack.envs import GrootDiffusionPolicyEnv, GrootNewtonEnv, GrootNewtonEnvConfig

base_env = GrootNewtonEnv(
    GrootNewtonEnvConfig(
        num_envs=64,
        device="cuda:0",
        obs_mode="policy",
        control_mode="pd_eef_pose_abs",
        bottle_lift_height=0.1,
        bottle_min_xy_displacement=0.1,
    )
)
env = GrootDiffusionPolicyEnv(base_env, obs_horizon=2, action_horizon=8)

observation, info = env.reset()
# observation["observation.state"]: [64, 2, 26]
# observation["observation.images.ego_view"]: [64, 2, 180, 320, 3]
# observation["observation.images.wrist_view"]: [64, 2, 480, 640, 3]

# physical_action_chunk: [64, 8, 19], CUDA float32
observation, reward, terminated, truncated, info = env.step(physical_action_chunk)
```

`reset(options={"env_idx": cuda_indices})` 和 `reset(world_mask=cuda_bool_mask)` 支持局部 reset。状态、reward、终止标志、图像和 history buffer 都留在 GPU。

## 任务阶段、成功条件和 Reward

任务使用每个 world 独立的 GPU 状态机：

```text
APPROACH → CARRYING → RELEASED → SUCCESS / FAIL
```

环境构造时先在所有 lane 上并行模拟瓶子自由落体 60 个 60 Hz 帧，然后只把瓶子 free joint 的 7 个稳定位姿坐标写入 reset default，并清零它的 6 个速度。机械臂和手的 default 不会被 settle 过程中的瞬时误差覆盖。因此 `initial_obj_pose` 、full reset 和 partial reset 都以落桌后的稳定瓶子姿态为基准，不再把初始约 20 mm 自由下落误计为负 lift，也不会抵消之后的真实离桌高度。这一过程使用已捕获的 CUDA graph，只发生在构造阶段，不增加 `step()` 热路径开销；`bottle_settle_metadata` 记录帧数、时长和 backend。

- `APPROACH`：实时抓持必须同时有拇指接触、至少一根非拇指接触，并达到 `grasp_finger_count`；该对向抓持默认连续保持 6 个 60 Hz 仿真帧后才进入 `CARRYING`，任一中断都会重新计数。
- `CARRYING`：进入该阶段后不再要求每帧维持严格对向抓持；只要任意手指仍接触瓶子，就累计 gated persistent 最大抬升、允许开始 transport/确认抬升，并可在有效放置姿态下授权 release。完全失去全部手部接触才开始独立的 6 帧 release debounce；无接触帧不会更新最大抬升或重新授权 release，因而投掷产生的物理高度不能推进任务。
- `RELEASED`：释放前必须在有效接触下曾达到 `initial_z + bottle_lift_height`（默认 `0.1 m`）；释放后重新接触会失败。
- `SUCCESS`：释放后 XY 相对初始位置的位移至少为 `0.1 m`、Z 误差不超过 `0.01 m`、相对初始姿态的四元数测地角不超过 `15°`，并且瓶子线速度和角速度连续 12 个 60 Hz 帧低于阈值。
- `FAIL`：搬运已经开始但在达到抬升高度或允许释放位姿前释放，释放后重新接触，或者瓶子稳定后偏离最终位姿范围。

目前没有指定单一目标方向；`bottle_min_xy_displacement` 表示从初始 XY 径向移开至少指定距离。以后增加固定目标区时，应把它替换为目标中心与区域边界，而不是复用抬升目标 `_goal_pos`。

接触仍是 privileged GPU task signal，不进入 26 维 policy state。接触候选会进一步检查几何 separation，默认只接受表面间距不超过 `0.2 mm` 的手指—瓶子 contact；它不是以 N 为单位的 `SensorContact` 力读数。`info["finger_contact_counts"]` / `evaluate()["finger_contact_counts"]` 在 GPU 上暴露按 `thumb,index,middle,ring,pinky` 排列的五路当前帧接触点计数。为避免 10 Hz 控制边界漏掉 60 Hz 瞬态，`finger_contact_any_frame_this_control_step` 对六个仿真帧逐指做 sticky-OR；`opposed_grasp_any_frame_this_control_step` 只在拇指与非拇指于同一帧满足严格对向抓持时置位，不能由两个不同时刻的逐指 OR 推导；`opposed_grasp_max_consecutive_physics_frames_this_control_step` 记录该控制周期内最长连续对向帧数。每次 `step()` 开始时清零所有 lane，partial reset 只清被 reset lane。Reward contract v13 还在六个 physics frame 上累计接触占空比和“已接触一侧引导缺失一侧”的连续几何进度；这些数组始终留在 GPU，不改变阶段推进、严格抓持确认或成功判定。

Dense reward 采用 ManiSkill `StackCube` 风格的阶段覆盖，而不是把所有阶段相加：

```text
APPROACH, 无接触:         R = r0 = min(1.35, r_reach + 0.35 s_pregrasp)
APPROACH, 仅非拇指接触:   R = R_N = r0 + (1.50 - r0) (0.10 c_N + 0.90 G_N)
APPROACH, 仅拇指接触:     R = R_T = r0 + (1.50 - r0) (0.10 c_T + 0.90 G_T)
APPROACH, 异步双侧接触:   R = max(R_N, R_T)
APPROACH, 当步瞬时对向:   R = min(1.60, 1.51 + 0.07 p_hold + 0.02 n_non_thumb / 4)
                                                                           # [1.529, 1.60]
APPROACH, 严格对向抓持:   R = 1 + 0.125 n_non_thumb + 0.5          # [1.625, 2]
CARRYING, 尚未达到高度:   R = 2 + 0.5 r_takeoff + 1.5 r_lift      # [2, 4]
CARRYING, 已达到高度:     R = 4 + r_place                         # [4, 5]
RELEASE_READY/RELEASED:   R = 6 + r_static                        # [6, 7]
SUCCESS:                  R = 8
FAIL:                     R = -8
normalized_dense:         R / 8
```

令一个 10 Hz control step 含 `F=6` 个 60 Hz physics frame，`p_j = 1 - tanh(d_j / 0.08 m)` 是第 `j` 根指尖到瓶子有限圆柱表面的 proximity，`o_i` 是 thumb 与非拇指 partner `i` 的径向对向度，`z_i` 是二者的轴向高度一致性。每个 physics frame 中，若非拇指 partner `i` 真实接触，缺失 thumb 的引导分数为 `u_Ni = p_thumb (0.60 + 0.25 o_i + 0.15 z_i)`，并在已接触 partners 中取最大值；若 thumb 真实接触，缺失非拇指一侧的引导分数为 `u_T = max_i p_i (0.60 + 0.25 o_i + 0.15 z_i)`。`c_N/c_T` 是对应 anchor 接触帧数除以 `F`，`G_N/G_T` 是逐帧最大 `u_N/u_T` 之和除以 `F`，因此 `0 <= G <= c <= 1`。几何项占单边增益的 90%，接触占空比只占 10%；固定的单边接触高平台被移除，策略必须在保持已接触一侧的同时把缺失一侧移向可形成对向抓持的几何位置。

`n_non_thumb` 是当步 sticky 的不同非拇指数量，`p_hold = min(k, 5) / 5`，`k` 是六个 physics frame 内最长的连续同帧对向接触。阶段严格覆盖而不累加：控制步最后一帧仍满足 `is_grasped` 时保留最低 `1.625` 的实时严格分支；否则，同帧对向 sticky 覆盖异步双侧或任一单侧分支，再回退到无接触 `r0`。拇指和非拇指只在不同帧分别接触时不能合成为对向接触，只能取得 `max(R_N,R_T)`。所有未确认分支最多为 `1.60`，严格低于 `1.625`；第六个连续帧会按原状态机进入 `CARRYING`，reward 最低为 `2.0`。

Reward contract v13 沿用 v11 的真实指尖和瓶子局部圆柱几何：`r_reach` 使用五个真实指尖点的中心；`s_pregrasp` 对四个 thumb--non-thumb 指尖对组合双方 proximity、径向对向度和 `0.04 m` 尺度的轴向高度一致性后取最大值。8 cm 距离尺度使典型约 12 cm 的拇指表面间隙仍保留约 `0.095` proximity；径向同向、正交和反向的对向度分别为 `0`、`0.5` 和 `1`。环境直接暴露 GPU 上的 `c_N/c_T/G_N/G_T`、partner 对向/高度分数、`finger_surface_gap`、`opposed_pregrasp_score` 以及单边 guidance gain/reward，用于 privileged critic 和边界日志；它们不进入 frozen DP 的 26 维 state 或 residual actor 输入。

进入 `CARRYING` 后，令 `h` 为确认严格抓持后、任意手指保持接触期间累计的最大正向高度；`r_takeoff = clamp(h / 0.01, 0, 1)`，`r_lift = clamp(h / bottle_lift_height, 0, 1)`。0.5/1.5 的权重同时保留最初 1 cm 的分辨率和整个配置抬升高度上的连续梯度，不再在 1 cm 形成平台，也不再把完整高度硬编码为 0.1 m。`r_place` 是 XY 位移、最终 Z 和姿态三个 `[0, 1]` shaping 项的平均值。Sparse reward 在成功时为 `+1`，失败时为 `-1`。确认抓持后，合法 release-ready 之前连续 6 个仿真帧失去全部手部接触会直接失败，不能退回 approach 反复领取接触奖励。

高度诊断分为三种：`current_lift_height` 是当前物理正向高度，`physical_max_lift_height` 是 episode 内无条件物理最大高度，`max_contacted_carry_lift_height` 只累计初始严格抓持确认后且仍有任意手指接触时的最大高度。最后一个 gated persistent 量同时参与 reward 和成功契约，避免投掷后利用无接触物理高度；训练日志同时报告物理最大高度和接触搬运最大高度，以区分“瓶子没有抬动”和“抬动但抓持丢失”。根级兼容字段 `lift_height` 与 `current_lift_height` 指向同一 GPU 数组；`max_lift_height` 与 `max_contacted_carry_lift_height` 也指向同一 GPU 数组。

训练默认使用固定 horizon：`terminate_on_success=False`，进入 `SUCCESS` 后每个剩余 step 都保持最高阶段奖励 8，直到 episode 截断；失败仍立即终止。这样越早完成任务回报越高，不会出现一直抓住瓶子领取 4–5 分反而优于释放的激励错配。独立成功率评估如果需要成功后立即 reset，可传 `--terminate-on-success`。

在线 RL rollout 第一版应使用 `action_horizon=1`。当前 action-chunk 接口会完整执行整个 chunk；某个 world 在 chunk 中途失败时，后续动作虽不再计入汇总 reward，仍会推进底层状态，因此不适合作为精确的 terminal transition。离线 DP 推理仍可使用多步 chunk；在线训练若要增大 horizon，需要先实现逐 world 冻结和首次 terminal snapshot。

基础环境 smoke test：

```bash
docker/run_groot_rl_env.sh \
  --num-envs 64 \
  --obs-mode policy \
  --control-mode pd_eef_pose_abs \
  --bottle-min-xy-displacement 0.1 \
  --steps 100
```

关闭图像只适合检查 physics/IK 吞吐量：

```bash
docker/run_groot_rl_env.sh --num-envs 256 --obs-mode state --no-images --steps 1000
```

## 数据窗口和训练器

`GrootLeRobotWindowDataset` 直接读取现有 Parquet 和 MP4：

```python
from teleop_stack.datasets import GrootLeRobotWindowDataset, create_groot_lerobot_bc_split

split = create_groot_lerobot_bc_split(
    "local_data/groot/smooth",
    validation_fraction=0.1,
    split_seed=0,
)
train_dataset = GrootLeRobotWindowDataset(
    "local_data/groot/smooth",
    obs_horizon=2,
    pred_horizon=16,
    episode_indices=split.train_episode_indices,
)
validation_dataset = GrootLeRobotWindowDataset(
    "local_data/groot/smooth",
    obs_horizon=2,
    pred_horizon=16,
    episode_indices=split.validation_episode_indices,
    stats=train_dataset.stats,
)
sample = train_dataset[0]
```

窗口语义为：

- observation：当前帧以及前一帧，episode 起点使用第一帧 padding；
- action：从当前帧开始的未来 16 个绝对动作，episode 末尾使用最后动作 padding；
- `action_is_pad`：使 diffusion loss 忽略末尾 padding；
- 数值数据：启动时读取所选 episode 的全部状态和动作，体积相对视频很小；
- 视频：若相邻的 `.mp4.frames.npy` RGB 缓存存在，DataLoader worker 会优先用只读 mmap 随机取帧；缺少缓存时才回退到 H.264 解码；
- 每个 worker 默认最多保留 8 个 mmap 或 `VideoCapture`，且每个回退用的 FFmpeg decoder 只使用 1 个线程，避免多日期数据随机采样时耗尽主机线程和文件句柄；
- `--require-frame-cache` 可让训练在任何所选相机缓存缺失时立即失败，避免正式任务静默回退到低吞吐的 MP4 随机 seek。

BC split 会先排除 `success != true` 或 `outcome != "success"` 的轨迹，再按
`raw_episode_id + source_start_frame + source_end_frame` 删除精确重复 clip。不同范围的 clip 会保留，
但同一 `raw_episode_id` 的所有 clip 一定进入同一个 train 或 validation split，避免相邻真机帧泄漏到两侧。
Isaac-GR00T 原始目录保持不变，不会物理删除或重编号 episode。

启动离线 DP 行为克隆：

```bash
conda_envs/newton/bin/python tools/train_newton_groot_diffusion_policy.py \
  --dataset local_data/groot/smooth \
  --output-dir checkpoints/dp/groot_l10_pick \
  --batch-size 32 \
  --num-workers 4 \
  --validation-workers 2 \
  --video-cache-size 8 \
  --video-decode-threads 1 \
  --require-frame-cache \
  --prefetch-factor 1 \
  --validation-fraction 0.1 \
  --split-seed 0 \
  --validate-every 5000 \
  --steps 100000
```

网络包含两个独立 camera encoder、一个 26 维 state encoder 和一个预测 16×19 action noise 的 temporal denoiser。训练、归一化、resize、noise scheduler 和 loss 都在 GPU。
训练目录会写入可审计的 `dataset_split.json`，定期保存可恢复训练的 step checkpoint，并把 validation loss
最低的模型保存为推理用 `best.pt`。归一化统计只从 train split 计算，validation 复用 train 统计；训练前
会把独立重算值与数据集的 `meta/dp_train_stats.json` 逐元素核对。checkpoint 同时保存 numeric fingerprint、
统计 payload 和统计 SHA-256；resume 会核对格式、网络配置、split、数值指纹和统计，推理也会核对统计
payload 与模型 normalization buffer。

运行 checkpoint：

```bash
conda_envs/newton/bin/python tools/run_newton_groot_dp.py \
  checkpoints/dp/groot_l10_pick/checkpoint_00100000.pt \
  --num-envs 32 \
  --action-horizon 8
```

## CPU/GPU 边界

| 路径 | 位置 | 原因 |
| --- | --- | --- |
| URDF/GLB/JSON/Parquet metadata 加载 | CPU，初始化阶段 | 文件 I/O 和场景构建 |
| `.frames.npy` 随机取帧 | CPU DataLoader worker + OS page cache | 只读 mmap，仅复制当前 batch |
| H.264 MP4 回退解码 | CPU DataLoader worker | 当前依赖中没有 NVDEC/DALI 解码链路 |
| pinned batch 到 CUDA | 异步传输 | `non_blocking=True` |
| DP encoder、归一化、denoiser、loss | GPU | 训练主路径 |
| Newton physics、camera render、reward、reset | GPU | batched Warp/MJWarp |
| EEF FK/Jacobian/DLS IK | GPU | batched Torch |
| 五路 `qfrc_actuator` 提取、归一化和 history | GPU | contract v9 residual actor 的实时指根负载；不进入冻结 DP 的 26 维 state |
| log/export | CPU，仅评估或 checkpoint 边界 | 不在逐 physics-step 主路径执行 |

不能把视频文件读取宣称为“全 GPU”。现有远端 smooth 数据已经包含逐帧 mmap cache，训练器不会把约 153 GB 缓存整体载入内存或显存，只依赖 OS page cache 并复制当前 batch。若仍需提高上限，可再接入 NVDEC/DALI 或生成降采样后的训练缓存；不建议把全部原始视频长期常驻显存。

## Residual PPO 在线微调

`train_newton_groot_residual_ppo.py` 使用已有 DP checkpoint 作为冻结的视觉动作先验，只训练 residual actor 和 critic：

- contract v9 保留每个 lane 独立缓存一次 DP plan 的冻结 condition 和前八行 action；正常情况下双相机/state encoder 与 denoiser 每八个 10 Hz control step 运行一次，而不是每步重跑；
- 每个 lane 依次执行缓存的 rows 0--7。row 7 执行后才失效并重新规划；partial reset 只丢弃对应 lane 的剩余 rows，其他 lane 的 row 和 condition 不变；checkpoint 的 prediction horizon 必须至少为 8；
- actor 输入由该 chunk 的 plan-time condition、当步实际 row 的 normalized base action、每步刷新的 normalized 26 维当前状态、当前与上一帧的 normalized state delta、五路实时指根负载和 row 0--7 one-hot 组成；state delta 与负载不进入 chunk cache，reset lane 的两者归零；residual 合成、动作诊断和 rollout 保存复用同一个 physical base tensor，避免 normalized base 与执行动作错行；
- actor 输出 16 维 residual latent：世界坐标位置 3 维、局部轴角 3 维和手指 10 维；
- 旋转 residual 在 SO(3) 上组合后重新编码为 row-first Rot6D，不直接加减六个矩阵元素；
- critic 额外使用 phase、接触、连续抬升进度、释放和目标误差等 GPU privileged task state，并在 contract v9 末尾加入 `r0,c_N,c_T,G_N,G_T`；actor 输入保持不变，不看到这些 reward guidance；
- rollout 只保存冻结后的 plan condition、当步基础动作、row、latent、value 和 reward，不保存历史 RGB 帧或完整 action chunk；
- `terminated` 和固定任务 horizon 的 `truncated` 默认都不 bootstrap，且 GAE 不跨 episode；若将时间限制视为非任务终态，可显式传 `--bootstrap-time-limit`，终态 value 查询使用独立 DP RNG，不能扰乱 behavior plan 的随机流；
- evaluation 按 lane 分 wave 收集首次 terminal，严格返回请求的 episode 数，不让快速失败 lane 重复污染统计；
- `best.pt` 按成功率、低失败率和高回报依次排序，`best_return.pt` 独立保留最高回报模型；
- 日志包含阶段占比/事件率、partial non-thumb/thumb/bilateral-unconfirmed 的训练 step occupancy 与 eval episode ever rate、`r0/c_N/c_T/G_N/G_T` 的均值/非零率、非拇指接触条件下的 thumb gap/proximity 与 guidance opposition/Z、单边 guidance gain/reward、物理与接触搬运最大抬升、1/10/50 mm 的 rollout lane/step 与 eval episode 达成率、当前帧和 control-step 接触、最长连续对向 physics-frame、`opposed_pregrasp_score`、contact→grasp 转化、EEF ΔZ、每指负载、rows 0--7 cache、动作饱和/裁剪、梯度范数和最终 KL。接触与 guidance 逐 physics/control step 都在 GPU 上累计，只在 train/eval 日志边界读取聚合标量；
- checkpoint 保存 actor/critic、optimizer、更新计数、DP 文件 SHA-256、contract v9 的 cache/actor/critic/坐标帧语义、有效 10 维 hand residual scale 及拇指索引、active+mimic hand target metadata、reward contract v13、五路负载的 joint 顺序/归一化/采样时序和独立 CUDA generator 状态。恢复训练会 full reset 环境并让全部 cache 失效，不恢复与物理状态不同步的 chunk；contract v8、reward contract v12 或更旧 residual checkpoint 会被明确拒绝，避免不同 critic 输入或奖励语义被误 resume。

示例：

```bash
conda_envs/newton/bin/python tools/train_newton_groot_residual_ppo.py \
  checkpoints/dp/groot_l10_pick/best.pt \
  --output-dir checkpoints/residual_ppo/groot_l10_transfer \
  --num-envs 32 \
  --rollout-steps 64 \
  --minibatch-size 512 \
  --inference-steps 6 \
  --max-episode-steps 300 \
  --thumb-residual-scales-normalized 0.25 0.75 0.90 \
  --initial-log-std -2.2 \
  --total-timesteps 2000000
```

默认 residual 上限为 world XY `0.015 m`、world Z `0.05 m`、旋转 `5°`、手指归一化范围 `0.1`。`--hand-residual-scale-normalized` 仍控制其余手指并提供兼容默认；需要扩大拇指可达域时，可显式传 `--thumb-residual-scales-normalized PITCH YAW ROLL`，只覆盖 hand indices `0/1/9`（latent `6/7/15`）。启动时会把最终 10 维 scale 一次上传并复用于动作合成和诊断，不在 rollout step 内重新建立 CPU tensor。日志分别给出三个拇指坐标的 signed/absolute normalized residual、tanh 饱和、绝对动作边界裁剪，以及相对当前 hand state 的 `0.08 rad` 动态限速率。独立的 world Z 范围让 residual 能补足示教中常见的约 60--80 mm 抬升，同时仍保留 95 mm 的最终达标门槛；扩大范围前应检查 world-Z residual、IK/动作裁剪率和接触稳定性。`vertical_action_clamp_fraction` 是保守的“任一 CAN XYZ bound 被裁剪”指标，因为一般固定旋转会把 world-up 命令分布到多个 CAN 轴；其余 Z gap 和实际 EEF ΔZ 指标都在 world frame 计算。Residual PPO 默认 episode horizon 为 300 个 10 Hz step，以覆盖现有成功示教中抓取、搬运和释放的主要时长分布。训练使用上面的阶段覆盖 `normalized_dense` reward，成功阶段在固定 horizon 内保持最高奖励，使越早完成任务的回报越高。

当前 bottle 任务的成功轨迹对照中，non-thumb 已接触而 thumb 未接触的样本需要的 pitch/yaw/roll residual normalized P95 约为 `0.173/0.586/0.680`；统一 `0.1` 对三轴的覆盖率只有约 `6%--9%`。上例显式使用 `0.25/0.75/0.90`，分别提供约 `0.047/0.500/0.464 rad` 的最大物理修正并保留余量。它不是库级默认值；换手型、动作统计或数据集后应重新做轨迹对照。环境仍以每 control step `0.08 rad` 限速并记录逐轴限速/饱和/裁剪率，因此应根据这些指标而不是只看 PPO loss 决定是否继续扩大范围。

从零启动 residual PPO 前，可用相同初态比较 frozen DP 的三种在线执行语义：

```bash
conda_envs/newton/bin/python tools/compare_newton_groot_dp_rollouts.py \
  checkpoints/dp/groot_l10_pick/best.pt \
  --output-dir runs/dp_index_compare \
  --modes index0 index1 chunk8 \
  --num-envs 32 \
  --max-episode-steps 300 \
  --inference-steps 6 \
  --seed 0
```

`index0` 和 `index1` 都每个 control step 重规划，分别执行对应 chunk row；`chunk8` 每次规划后逐步执行 rows 0--7，是 contract v9 在无异步 reset 时的 frozen-base 对照。v9 trainer 不能使用全局 `control_step % 8`：每个 lane 有独立 row cursor，reset lane 从新 plan 的 row 0 开始。对照使用相同 episode 配额和最大 horizon，而不是按 DP call 数统计；实际 active control steps 会随提前终止而变化，并与六个 60 Hz simulation frame 上累计的接触/缓冲区 high-water 与 overflow 诊断一起写入 `summary.json`。三维 action-chunk adapter 在 terminal 后仍可能推进已结束 world，因此该诊断工具始终通过二维 action 逐步执行、保存每个 lane 的首次 terminal，并 reset 所有 terminal lane，避免已计数 lane 污染共享碰撞诊断。

`qfrc_actuator` 是 PD actuator 输出，不是纯接触反力。开始 v7 长训前必须运行负载探针：

```bash
conda_envs/newton/bin/python tools/probe_newton_l10_finger_root_load.py \
  checkpoints/dp/groot_l10_pick/best.pt \
  --output-dir runs/finger_root_load_probe \
  --num-envs 32 \
  --episodes 32 \
  --max-episode-steps 300 \
  --inference-steps 6
```

探针按 frozen DP chunk8 语义执行，并检查 reset 后五路为零、数值 finite、无接触饱和率低于 10%、抓持时第二大指根负载相对无接触至少增加 0.1，以及该第二大负载区分几何抓持的 AUC 大于 0.75。任一门槛失败时不得直接长训，应先校准空载 bias/scale，或改用真机可复现的五路外部接触 gate；不能把只识别瓶子的仿真 contact flag 直接泄露给 actor。

环境通用默认的网格窄相位 triangle-pair buffer 仍按每环境 `65,536` 个候选对扩展，并保留至少 `1,000,000` 的容量。Residual PPO 的 ±15 mm 平移与手部闭合动作可产生更高候选峰值，因此 v7 训练入口通过 `--triangle-pairs-per-env` 显式默认使用 `131,072`；32 个并行环境对应 `4,194,304`。该值同时写入 args、env config、run config 和 checkpoint resume contract。若仍出现 `Triangle pair buffer overflowed`，该 rollout 的接触与 reward 可能已被截断，不应继续用于训练。

离线 BC trainer 仍只优化 diffusion noise loss；Residual PPO 不修改 DP 权重。若后续需要直接更新 denoiser，仍需单独实现和验证 DPPO 或其他 diffusion RL 目标。
