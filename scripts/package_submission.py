#!/usr/bin/env python
"""Create ProjectGroup_[ID].zip per submission requirements."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "report.pdf",
    "README.md",
    "requirements.txt",
    "demo.py",
    "notebooks/",
    "src/",
    "results/",
    "config/config.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="ID", help="Group ID for filename")
    args = parser.parse_args()

    out = ROOT / f"ProjectGroup_{args.id}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            path = ROOT / item
            if not path.exists():
                print(f"WARNING: missing {item}")
                continue
            if path.is_dir():
                for f in path.rglob("*"):
                    if f.is_file() and ".venv" not in str(f):
                        zf.write(f, f.relative_to(ROOT))
            else:
                zf.write(path, path.name)
    print(f"Created: {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
