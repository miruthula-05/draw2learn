import json
from pathlib import Path
import time


def load_project_state(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_project_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2)
    temp_path = path.with_suffix(".tmp")
    for _ in range(3):
        try:
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(path)
            return
        except OSError:
            time.sleep(0.1)
    try:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    except OSError:
        pass
