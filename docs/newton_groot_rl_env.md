# Newton Nero + L10 Diffusion Policy 环境

这套环境用于在 Newton/MJWarp 中并行执行瓶子抓取、搬运和释放任务。机械手必须抓稳瓶子、保持接触将其抬高至少 0.1 m、在 XY 平面移出初始位置、放回接近初始高度并保持初始姿态，最后释放并等待瓶子稳定。瓶子的初始位置目前固定；第一版策略不把接触力或 `qfrc_actuator` 放入 observation，只对齐现有 LeRobot 数据集的图像、26 维状态和 19 维绝对动作。

## 数据协议

环境的 policy observation 使用与 `local_data/groot/smooth/meta/info.json` 相同的字段名：

| 字段 | 单帧形状 | dtype | 内容 |
| --- | ---: | --- | --- |
| `observation.state` | `[26]` | `float32` | `arm_joint_pos[7] + arm_eef_pos[3] + arm_eef_rot6d[6] + hand_joint_pos[10]` |
| `observation.images.ego_view` | `[180, 320, 3]` | `uint8` | node0 ROI 后的头部 RGB |
| `observation.images.wrist_view` | `[480, 640, 3]` | `uint8` | 腕部 RGB |
| `action` | `[19]` | `float32` | 绝对 EEF xyz + 绝对 EEF rotation-6D + 绝对手指目标 |

26 维 state 的严格顺序是：

```text
[arm_joint_pos(7), eef_xyz(3), eef_rotation_6d(6), hand_joint_pos(10)]
```

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

- `APPROACH`：至少两根不同手指连续接触瓶子 2 个 60 Hz 仿真帧后确认抓取。
- `CARRYING`：每个 60 Hz 仿真帧检查五指所有指节与瓶子的接触；允许单帧数值抖动，连续 2 帧完全无接触才确认释放。只有最后一个仍有接触的帧已经满足抬升和最终位姿约束，释放才有效；瓶子脱手后靠惯性进入目标区会失败。
- `RELEASED`：释放前必须在有效接触下曾达到 `initial_z + 0.1 m`；释放后重新接触会失败。
- `SUCCESS`：释放后 XY 相对初始位置的位移至少为 `0.1 m`、Z 误差不超过 `0.01 m`、相对初始姿态的四元数测地角不超过 `15°`，并且瓶子线速度和角速度连续 12 个 60 Hz 帧低于阈值。
- `FAIL`：搬运已经开始但在达到抬升高度或允许释放位姿前释放，释放后重新接触，或者瓶子稳定后偏离最终位姿范围。

目前没有指定单一目标方向；`bottle_min_xy_displacement` 表示从初始 XY 径向移开至少指定距离。以后增加固定目标区时，应把它替换为目标中心与区域边界，而不是复用抬升目标 `_goal_pos`。

接触仍是 privileged GPU task signal，不进入 26 维 policy state。接触候选会进一步检查几何 separation，默认只接受表面间距不超过 `0.2 mm` 的手指—瓶子 contact；它不是以 N 为单位的 `SensorContact` 力读数。

Dense reward 采用 ManiSkill `StackCube` 风格的阶段覆盖，而不是把所有阶段相加：

```text
APPROACH:                 R = 2 * r_reach                         # [0, 2]
CARRYING, 尚未达到高度:   R = 3 + r_lift                          # [3, 4]
CARRYING, 已达到高度:     R = 4 + r_place                         # [4, 5]
RELEASED, 最终位姿合格:   R = 6 + r_static                        # [6, 7]
SUCCESS:                  R = 8
FAIL:                     R = 0
normalized_dense:         R / 8
```

其中 `r_place` 是 XY 位移、最终 Z 和姿态三个 `[0, 1]` shaping 项的平均值。Sparse reward 在成功时为 `+1`，失败时为 `-1`。阶段基线严格递增，使后续阶段覆盖前一阶段；连续接触作为状态机硬条件，不提供可被策略通过永不释放而长期刷取的独立正奖励。

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
| qfrc/log/export | 不启用 | 第一版协议明确排除 |

不能把视频文件读取宣称为“全 GPU”。现有远端 smooth 数据已经包含逐帧 mmap cache，训练器不会把约 153 GB 缓存整体载入内存或显存，只依赖 OS page cache 并复制当前 batch。若仍需提高上限，可再接入 NVDEC/DALI 或生成降采样后的训练缓存；不建议把全部原始视频长期常驻显存。

## Residual PPO 在线微调

`train_newton_groot_residual_ppo.py` 使用已有 DP checkpoint 作为冻结的视觉动作先验，只训练 residual actor 和 critic：

- 每步只运行一次双相机/state encoder，冻结特征同时供 DP denoiser、actor 和 critic 使用；
- DP 仍预测完整的 `16×19` action chunk，但在线 rollout 固定 `action_horizon=1`；
- actor 输入冻结特征和本次 DP 基础动作，输出 16 维 residual latent：世界坐标位置 3 维、局部轴角 3 维和手指 10 维；
- 旋转 residual 在 SO(3) 上组合后重新编码为 row-first Rot6D，不直接加减六个矩阵元素；
- critic 额外使用 phase、接触、抬升、释放和目标误差等 GPU privileged task state，这些信息不进入 actor；
- rollout 只保存冻结后的特征、基础动作、latent、value 和 reward，不保存历史 RGB 帧；
- `terminated` 和固定任务 horizon 的 `truncated` 默认都不 bootstrap，且 GAE 不跨 episode；若将时间限制视为非任务终态，可显式传 `--bootstrap-time-limit` 使用 reset 前的最终 observation value；
- checkpoint 保存 actor/critic、optimizer、更新计数、DP 文件 SHA-256 和独立 CUDA generator 状态。恢复训练会重新 reset 环境，不承诺物理轨迹逐 bit 延续。

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
  --total-timesteps 2000000
```

默认 residual 上限为位置 `0.015 m`、旋转 `5°`、手指归一化范围 `0.1`。这些限制比并行环境数量更影响训练安全性；扩大 residual 前应先检查 IK 饱和率和瓶子接触稳定性。Residual PPO 默认 episode horizon 为 300 个 10 Hz step，以覆盖现有成功示教中抓取、搬运和释放的主要时长分布。训练使用上面的阶段覆盖 `normalized_dense` reward，成功阶段在固定 horizon 内保持最高奖励，使越早完成任务的回报越高。

网格窄相位的 triangle-pair buffer 默认按每环境 `65,536` 个候选对扩展，并保留至少 `1,000,000` 的容量；因此 32 个并行环境使用 `2,097,152`。一旦日志出现 `Triangle pair buffer overflowed`，该 rollout 的接触与 reward 可能已被截断，不应继续用于训练。

离线 BC trainer 仍只优化 diffusion noise loss；Residual PPO 不修改 DP 权重。若后续需要直接更新 denoiser，仍需单独实现和验证 DPPO 或其他 diffusion RL 目标。
