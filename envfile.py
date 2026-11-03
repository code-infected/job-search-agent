"""Write settings chosen on the setup page into .env and the running process."""

import os
import re
from pathlib import Path

KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*")
# Values are written unquoted, so anything that would change how a line parses is refused.
UNSAFE_VALUE_RE = re.compile(r"[\s'\"\\#=`$]")


def update_env(path: Path, updates: dict[str, str]) -> None:
    for key, value in updates.items():
        if not KEY_RE.fullmatch(key):
            raise ValueError(f"invalid setting name: {key!r}")
        if UNSAFE_VALUE_RE.search(value) or len(value) > 500:
            raise ValueError(f"{key} contains characters that are not allowed")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    for index, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            lines[index] = f"{key}={remaining.pop(key)}"
    lines += [f"{key}={value}" for key, value in remaining.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ.update(updates)
