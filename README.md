# Newton Quest Teleop 启动流程

## Docker 启动

推荐使用 direct-gpu 模式，固定使用物理 GPU0，并使用 `DISPLAY=:1`：

```bash
cd ~/project/newton
NEWTON_VR_GPU=0 DISPLAY=:1 docker/run_vr_stack.sh --vr-output-mode direct-gpu
```

## GR00T 模型推理

在 Docker 中使用 GPU0 启动 Newton 仿真和 GR00T RTC 推理：

```bash
cd ~/project/newton
NEWTON_GROOT_GPU=0 docker/run_groot_rtc.sh \
  --viewer gl \
  --image-source sim \
  --state-source sim \
  --start-policy
```

该命令使用仿真实时图像和仿真状态作为模型输入，并在场景启动后立即开始策略推理。更多参数见 [Newton GR00T RTC control](docs/newton_groot_rtc_control.md)。

桌面、瓶子和 L10 手的默认物理参数集中在
[`configs/scene_physics/groot_rtc.json`](configs/scene_physics/groot_rtc.json)。修改该文件后重启容器即可生效；也可以通过
`--scene-physics-config PATH` 选择另一份配置。GR00T 推理和 VR 遥操默认共享该配置，命令行或对应
`NEWTON_*` 环境变量中显式传入的物理参数优先于配置文件。

## 非 Docker 启动

需要在 `newton` conda 环境中启动：

```bash
cd ~/project/newton
conda activate newton
scripts/run_newton_vr_prereqs_object.sh --display :0
```
