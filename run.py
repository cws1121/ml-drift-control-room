"""One-command local setup and launch. Requires Python 3.12."""

import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info < (3, 12):
        raise SystemExit(
            "Python 3.12 or newer is required; 3.12 is the tested version."
        )
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print("Creating local Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    requirements = ROOT / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = environment / "requirements.sha256"
    if not stamp.exists() or stamp.read_text() != digest:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(requirements)], check=True
        )
        stamp.write_text(digest)
    print("Open http://127.0.0.1:4182 — stop with Ctrl+C", flush=True)
    try:
        subprocess.run(
            [
                str(python),
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "4182",
            ],
            cwd=ROOT,
            check=True,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
