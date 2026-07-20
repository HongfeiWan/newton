from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = REPO_ROOT / "tools" / "overlay_assets" / "isaac_teleop_cloudxr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply teleop_stack-owned CloudXR web client and SDK script overlays onto an IsaacTeleop checkout."
        )
    )
    parser.add_argument(
        "--isaac-teleop-root",
        required=True,
        help="Path to the upstream IsaacTeleop checkout to patch in place.",
    )
    return parser.parse_args()


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Expected snippet not found while patching {path}")
    return text.replace(old, new, 1)


def patch_download_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "\n# Colors for output\n",
        (
            "\n# Load CloudXR env defaults and local overrides when this script is run directly.\n"
            "# This keeps the script usable without requiring the caller to pre-export variables.\n"
            "# shellcheck disable=SC1091\n"
            'source "$GIT_ROOT/scripts/setup_cloudxr_env.sh"\n'
            "\n# Colors for output\n"
        ),
        path,
    )
    path.write_text(text, encoding="utf-8")


def patch_dockerfile_web_app(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    legacy_npm_install = "  npm install --ignore-scripts ../sdk.tgz && npm install --ignore-scripts\n"
    patched_npm_install = "  npm install --ignore-scripts --no-save ../sdk.tgz && npm install --ignore-scripts\n"
    if legacy_npm_install in text and patched_npm_install not in text:
        text = text.replace(legacy_npm_install, patched_npm_install, 1)

    legacy_nginx_block = """    location / { \\
        proxy_pass http://127.0.0.1:8080; \\
        proxy_http_version 1.1; \\
        proxy_set_header Upgrade $http_upgrade; \\
        proxy_set_header Connection "upgrade"; \\
        proxy_set_header Host $host; \\
        proxy_cache_bypass $http_upgrade; \\
    } \\
}' > /etc/nginx/conf.d/default.conf
"""
    patched_nginx_block = """    location / { \\
        proxy_pass http://127.0.0.1:8080; \\
        proxy_http_version 1.1; \\
        proxy_set_header Upgrade $http_upgrade; \\
        proxy_set_header Connection "upgrade"; \\
        proxy_set_header Host $host; \\
        proxy_cache_bypass $http_upgrade; \\
    } \\
    location /quest-voice { \\
        proxy_pass http://127.0.0.1:8766; \\
        proxy_http_version 1.1; \\
        proxy_set_header Upgrade $http_upgrade; \\
        proxy_set_header Connection "upgrade"; \\
        proxy_set_header Host $host; \\
        proxy_read_timeout 86400; \\
    } \\
}' > /etc/nginx/conf.d/default.conf
"""
    if legacy_nginx_block in text and patched_nginx_block not in text:
        text = text.replace(legacy_nginx_block, patched_nginx_block, 1)

    path.write_text(text, encoding="utf-8")


def patch_device_profiles(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "export type DeviceProfileId = 'custom' | 'quest2' | 'quest3' | 'quest3s' | 'pico4ultra';\n",
        """export type DeviceProfileId =
  | 'custom'
  | 'quest2'
  | 'quest3'
  | 'quest3_laptop_safe'
  | 'quest3s'
  | 'pico4ultra';
""",
        path,
    )

    text = replace_once(
        text,
        "\n// Quest 3S: same as Quest 3 pending device-specific validation.\n",
        """
// Quest 3 laptop-safe defaults prioritize stable streaming over maximum fidelity.
const QUEST3_LAPTOP_SAFE_PROFILE: DeviceProfile = {
  id: 'quest3_laptop_safe',
  label: 'Quest 3 (Laptop Safe)',
  description: 'Conservative defaults for laptop-hosted streaming.',
  connection: {
    httpsRequired: false,
  },
  web: {
    webglAntialias: false,
    xrWebGLLayerAntialias: false,
    powerPreference: 'high-performance',
    framebufferScaleFactor: 0.75,
    fixedFoveation: 1.0,
    frameBufferScaling: 0.75,
    foveation: 1.0,
  },
  cloudxr: {
    perEyeWidth: 1536,
    perEyeHeight: 1344,
    deviceFrameRate: 72,
    maxStreamingBitrateKbps: 30000,
    codec: 'h264',
    enablePoseSmoothing: true,
    posePredictionFactor: 1.0,
    enableTexSubImage2D: true,
    useQuestColorWorkaround: true,
  },
};

// Quest 3S: same as Quest 3 pending device-specific validation.
""",
        path,
    )

    text = replace_once(
        text,
        "  quest3: QUEST3_PROFILE,\n",
        "  quest3: QUEST3_PROFILE,\n  quest3_laptop_safe: QUEST3_LAPTOP_SAFE_PROFILE,\n",
        path,
    )

    text = replace_once(
        text,
        "    value === 'quest3' ||\n",
        "    value === 'quest3' ||\n    value === 'quest3_laptop_safe' ||\n",
        path,
    )

    path.write_text(text, encoding="utf-8")


def patch_index_html(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "        .validation-message-box.show {\n            display: block;\n        }\n\n        /* Error Message Box */\n",
        """        .validation-message-box.show {
            display: block;
        }

        .voice-command-card {
            margin-top: 16px;
            margin-bottom: 8px;
            padding: 12px 14px;
            border: 1px solid var(--border-color);
            background: #f8faf5;
        }

        .voice-command-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .voice-command-status {
            font-size: 0.85rem;
            color: #444;
            margin-top: 8px;
            line-height: 1.4;
        }

        .voice-command-hint {
            font-size: 0.8rem;
            color: #555;
            margin-top: 8px;
            line-height: 1.4;
        }

        .voice-command-button {
            width: auto;
            min-height: 44px;
            padding: 10px 14px;
            font-size: 0.9rem;
            color: #000;
        }

        /* Error Message Box */
""",
        path,
    )

    text = replace_once(
        text,
        '                        <option value="quest3">Quest 3</option>\n',
        (
            '                        <option value="quest3">Quest 3</option>\n'
            '                        <option value="quest3_laptop_safe">Quest 3 (Laptop Safe)</option>\n'
        ),
        path,
    )

    text = replace_once(
        text,
        '                        <option value="80">80 Mbps</option>\n',
        (
            '                        <option value="30">30 Mbps</option>\n'
            '                        <option value="80">80 Mbps</option>\n'
        ),
        path,
    )

    text = replace_once(
        text,
        '                <div id="errorMessageBox" class="error-message-box" role="alert" aria-live="polite">\n                    <span class="error-icon">⚠</span>\n                    <span id="errorMessageText"></span>\n                </div>\n\n                <h2 class="settings-title">Settings</h2>\n',
        """                <div id="errorMessageBox" class="error-message-box" role="alert" aria-live="polite">
                    <span class="error-icon">⚠</span>
                    <span id="errorMessageText"></span>
                </div>

                <div class="voice-command-card">
                    <div class="voice-command-title">Quest Voice Commands</div>
                    <button id="voiceCommandToggle" class="voice-command-button" type="button">Enable Voice</button>
                    <div id="voiceCommandStatus" class="voice-command-status">Quest microphone uplink is off</div>
                    <div class="voice-command-hint">Enable this before entering XR. Spoken commands: 开始 / 暂停 / 继续 / 重置 / 停止</div>
                </div>

                <h2 class="settings-title">Settings</h2>
""",
        path,
    )

    path.write_text(text, encoding="utf-8")


def patch_app_tsx(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "import { CloudXR2DUI } from './CloudXR2DUI';\n",
        "import { CloudXR2DUI } from './CloudXR2DUI';\nimport { QuestVoiceCommandBridge } from './QuestVoiceCommandBridge';\n",
        path,
    )

    desired_bridge_block = """  useEffect(() => {
    if (!cloudXR2DUI) {
      return;
    }

    const voiceBridge = new QuestVoiceCommandBridge({
      buttonId: 'voiceCommandToggle',
      statusId: 'voiceCommandStatus',
      uplinkSampleRate: 16000,
    });
    voiceBridge.initialize();
    return () => {
      voiceBridge.cleanup();
    };
  }, [cloudXR2DUI]);

  // Update HTML error message display when error state changes
"""
    old_bridge_block = """  useEffect(() => {
    if (!cloudXR2DUI) {
      return;
    }

    const voiceBridge = new QuestVoiceCommandBridge({
      buttonId: 'voiceCommandToggle',
      statusId: 'voiceCommandStatus',
      websocketPort: 8766,
      uplinkSampleRate: 16000,
    });
    voiceBridge.initialize();
    return () => {
      voiceBridge.cleanup();
    };
  }, [cloudXR2DUI]);

  // Update HTML error message display when error state changes
"""
    if desired_bridge_block in text:
        pass
    elif old_bridge_block in text:
        text = text.replace(old_bridge_block, desired_bridge_block, 1)
    else:
        text = replace_once(
            text,
            "  // Update HTML error message display when error state changes\n",
            desired_bridge_block,
            path,
        )

    old_status_block = """  const handleStatusChange = (connected: boolean, status: string) => {
    setIsConnected(connected);
    setSessionStatus(status);
  };
"""
    new_status_block = """  const handleStatusChange = (connected: boolean, status: string) => {
    setIsConnected(connected);
    setSessionStatus(status);
    if (connected) {
      setErrorMessage('');
      cloudXR2DUI?.hideError();
    }
  };
"""
    if new_status_block in text:
        pass
    elif old_status_block in text:
        text = text.replace(old_status_block, new_status_block, 1)
    else:
        print(f"Warning: status-change cleanup snippet not found while patching {path}; skipping optional patch.")

    path.write_text(text, encoding="utf-8")


def copy_overlay_file(relative_path: str, *, isaac_teleop_root: Path) -> None:
    source = OVERLAY_ROOT / relative_path
    target = isaac_teleop_root / "deps" / "cloudxr" / relative_path
    if not source.is_file():
        raise RuntimeError(f"Overlay asset not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    isaac_teleop_root = Path(args.isaac_teleop_root).expanduser().resolve()

    if not isaac_teleop_root.is_dir():
        raise SystemExit(f"IsaacTeleop root not found: {isaac_teleop_root}")

    patch_download_script(isaac_teleop_root / "scripts" / "download_cloudxr_sdk.sh")
    patch_download_script(isaac_teleop_root / "scripts" / "download_cloudxr_runtime_sdk.sh")
    patch_dockerfile_web_app(isaac_teleop_root / "deps" / "cloudxr" / "Dockerfile.web-app")
    patch_device_profiles(isaac_teleop_root / "deps" / "cloudxr" / "webxr_client" / "helpers" / "DeviceProfiles.ts")
    patch_index_html(isaac_teleop_root / "deps" / "cloudxr" / "webxr_client" / "src" / "index.html")
    patch_app_tsx(isaac_teleop_root / "deps" / "cloudxr" / "webxr_client" / "src" / "App.tsx")
    copy_overlay_file("webxr_client/src/QuestVoiceCommandBridge.ts", isaac_teleop_root=isaac_teleop_root)

    print(f"Applied teleop_stack CloudXR overlay to {isaac_teleop_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
