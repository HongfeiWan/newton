# Newton Quest Teleop 启动流程

## Docker 启动

推荐使用 direct-gpu 模式，固定使用物理 GPU0，并使用 `DISPLAY=:1`：

```bash
cd ~/project/newton
NEWTON_VR_GPU=0 DISPLAY=:1 docker/run_vr_stack.sh --vr-output-mode direct-gpu
```

## 非 Docker 启动

需要在 `newton` conda 环境中启动：

```bash
cd ~/project/newton
conda activate newton
scripts/run_newton_vr_prereqs_object.sh --display :0
```
