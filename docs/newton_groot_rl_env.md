# Newton Nero + L10 Diffusion Policy 环境

这套环境用于在 Newton/MJWarp 中并行执行“抓住瓶子并向上抬高 0.1 m”。瓶子的初始位置目前固定；第一版策略不使用接触力或 `qfrc_actuator`，只对齐现有 LeRobot 数据集的图像、26 维状态和 19 维绝对动作。

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

基础环境 smoke test：

```bash
docker/run_groot_rl_env.sh \
  --num-envs 64 \
  --obs-mode policy \
  --control-mode pd_eef_pose_abs \
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

## 当前可训练范围

现有 trainer 是标准离线 Diffusion Policy behavior cloning。它会使用数据集中所有去重后的成功示教建立
episode 级 train/validation split，训练一个比 GR00T 小得多的初始化策略，并直接接入 Newton rollout。
它还不是奖励驱动的在线 RL 算法。

要进行真正的在线 RL 微调，还需要明确选择 DPPO、Diffusion-QL 或“GR00T/成功 rollout 写入 replay 后继续 diffusion BC”的路线，并补齐 replay buffer、advantage/Q loss、策略版本与评估循环。环境、19 维动作、26 维 state、双相机 observation history 和 GPU action-chunk 接口已经为这些训练器准备好；第一版应先验证离线 DP 能稳定完成抬瓶，再增加在线 RL，避免同时排查视觉域差异、IK 和 RL 优化三个问题。
