from __future__ import annotations

import inspect
from pathlib import Path


def main() -> int:
    import isaacteleop.cloudxr.wss as wss

    path = Path(inspect.getfile(wss)).resolve()
    text = path.read_text(encoding="utf-8")

    marker = '    backend_uri = f"ws://{backend_host}:{backend_port}{path}"\n'
    replacement = '''    if path.startswith("/quest-voice"):
        voice_backend_port = int(os.environ.get("TELEOP_QUEST_VOICE_BIND_PORT", "8766"))
        backend_uri = f"ws://127.0.0.1:{voice_backend_port}{path}"
    else:
        backend_uri = f"ws://{backend_host}:{backend_port}{path}"
'''
    if replacement in text:
        print(f"cloudxr_wss_voice_route: already patched {path}")
        return 0
    if marker not in text:
        raise SystemExit(f"Could not find CloudXR WSS backend_uri marker in {path}")

    text = text.replace(marker, replacement, 1)
    path.write_text(text, encoding="utf-8")
    print(f"cloudxr_wss_voice_route: patched {path}")
    print("Restart `python -m isaacteleop.cloudxr --accept-eula` for the route to take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
