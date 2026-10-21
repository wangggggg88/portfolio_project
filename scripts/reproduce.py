"""Rerun tests and the full analysis in a second, separately installed interpreter
and check that every output file in reports/full is byte-identical.

Create the second environment first, for example:
    python -m venv .venv_repro
    .venv_repro\\Scripts\\python.exe -m pip install -r requirements-lock.txt
"""

import argparse
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/full"
SKIP = {"reproducibility.json"}


def hashes(folder):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name not in SKIP
    }


def main(interpreter):
    if not interpreter.exists():
        sys.exit(f"Second interpreter not found: {interpreter}")
    sys.path.insert(0, str(ROOT))
    tests = unittest.defaultTestLoader.discover(str(ROOT / "tests")).countTestCases()
    before = hashes(OUT)
    for cmd in (
        ["-m", "unittest", "discover", "-s", "tests"],
        ["-m", "src.run_analysis", "--config", "config/full.json"],
    ):
        subprocess.run([str(interpreter), *cmd], cwd=ROOT, check=True)
    after = hashes(OUT)
    differences = sorted(f for f in set(before) | set(after) if before.get(f) != after.get(f))
    if differences:
        raise AssertionError(differences)
    result = {
        "status": "passed",
        "second_environment": interpreter.parent.parent.name,
        "unit_test_count": tests,
        "full_analysis": "passed",
        "files_compared": len(after),
        "different_files": differences,
        "output_sha256": after,
    }
    (OUT / "reproducibility.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS: {tests} tests and the full run reproduce all {len(after)} output files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=str(ROOT / ".venv_repro/Scripts/python.exe"))
    main(Path(p.parse_args().python))
