# Newton Quest Teleop 启动流程

## 1. 准备一次视频设备

如果 `/dev/video44` 已存在，可以跳过。

```bash
sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2
```

确认：

```bash
ls -l /dev/video44
```

## 2. 一条命令启动

```bash
conda activate newton
scripts/run_newton_vr_prereqs.sh --display :0
```

启动成功后终端应看到：

```text
CloudXR web client is serving https://127.0.0.1:8443/
Quest web page: https://192.168.8.100:8443/
OpenXR runtime found but no active Quest session yet; retrying...
```

这表示主机已经在等待 Quest 连接。

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
