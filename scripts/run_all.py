#!/usr/bin/env python
"""Execute full ML/DL pipeline end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(script: str, quick: bool = False) -> None:
    cmd = [sys.executable, str(Path(__file__).parent / script)]
    if quick:
        cmd.append("--quick")
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run complete university ML/DL project")
    parser.add_argument("--quick", action="store_true", help="Reduced epochs and grids for smoke test")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    args = parser.parse_args()

    run("run_classical_ml.py", args.quick)
    run("run_dimensionality.py", args.quick)
    run("run_mlp.py", args.quick)
    if not args.skip_cnn:
        run("run_cnn.py", args.quick)
    if not args.skip_transfer:
        run("run_transfer.py", args.quick)

    print("\nAll pipelines completed successfully.")


if __name__ == "__main__":
    main()
