# Newton Quest Teleop 启动流程

## 1. 准备一次视频设备

如果 `/dev/video44` 已存在，可以跳过。

```bash
sudo modprobe -r v4l2loopback

sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2
```

确认：

```bash
ls -l /dev/video44
```

非docker启动，需要在newton conda环境中启动

scripts/run_newton_vr_prereqs_object.sh --display :0


## 2. Docker 一条命令启动

```bash
cd ~/project/newton
newgrp docker
DISPLAY=:0 docker/run_vr_stack.sh
```

不要默认用 `sudo` 启动这条命令；如果确实用 `sudo DISPLAY=:0 docker/run_vr_stack.sh`，
脚本也会自动挂载原用户 home 下的 `.cache` 和 `.cloudxr`。

启动成功后终端应看到：

```text
CloudXR web client is serving https://127.0.0.1:8443/
Quest web page: https://192.168.8.100:8443/
OpenXR runtime found but no active Quest session yet; retrying...
```

这表示主机已经在等待 Quest 连接。

启动前可先做一次 Docker 预检查：

```bash
cd ~/project/newton
DISPLAY=:0 docker/run_vr_stack.sh --check-only
```

正常应看到：

```text
[vr-prereqs] ok: preflight passed
```

Docker 启动脚本会把宿主机的 GPU、X11 display、`/dev/video44`、Docker socket、
`~/.cloudxr`、Vosk 模型、CloudXR web cache、`IsaacTeleop` 和本项目目录挂进容器，
并在容器内执行：

```bash
scripts/run_newton_vr_prereqs.sh --display :0
```

## 3. Quest 连接

在 Quest 浏览器打开：

```text
https://192.168.8.100:8443/
```

如果证书不通过，先打开：

```text
https://192.168.8.100:48322/
```

接受证书后，再回到 `https://192.168.8.100:8443/`。

进入 XR 页面后，打开语音，然后进入沉浸模式。

## 4. 语音控制

```text
开始    开始遥操
暂停    暂停跟随
继续    恢复跟随
重置    重新锚定
停止    停止跟随并保持
退出    退出遥操
```

也可以在主机上手动发送命令测试：

```bash
scripts/send_teleop_voice_command_once.sh --command engage
scripts/send_teleop_voice_command_once.sh --command clutch
scripts/send_teleop_voice_command_once.sh --command resume
scripts/send_teleop_voice_command_once.sh --command stop
```

## 5. 常用检查

检查 8443 是否启动：

```bash
ss -ltnp | grep 8443
curl -kI https://192.168.8.100:8443/
```

正常应看到 `HTTP/1.1 200 OK`。

查看日志：

```bash
tail -n 80 logs/vr_stack/cloudxr_runtime.log
tail -n 80 logs/vr_stack/cloudxr_web_client.log
tail -n 80 logs/vr_stack/quest_voice_bridge.log
tail -n 80 logs/vr_stack/newton_vr_output.log
```

停止整套流程：在启动终端按 `Ctrl+C`。
