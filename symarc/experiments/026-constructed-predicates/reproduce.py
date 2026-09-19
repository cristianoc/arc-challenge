"""Rebuild and independently audit 026 in a fresh output directory.

Usage: python3 reproduce.py --out /tmp/symarc026-replay
Only the Python standard library is required. Existing output is not overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCES = {
    "language.py": "27541e70aa1a0b3585e13be9ff247e88e8d114772ce87abf5bb5e979fb46354d",
    "run.py": "0819beb49a0925ecc4e5cc03d7e28f72d8fcb8362837142b157e8bc6215328f2",
    "test_run.py": "686f8c8ca0ad15552cfec64ebf3a07cc5dfa526e95e8466121519b5bb9517ebd",
    "audit.py": "8606afbc856a2088ca66e25459e244af7025522d20ac2472aff1c099a477e942",
}
OUTPUTS = {
    "library.json": "7ea82c7a4157ab85e0440f0319ef7f48339b8e4ab1704bf84d76fa0deb1c3195",
    "construction.json": "5075efd376f1bb2900b46022244013223883e9db0f1e896afae4f167b84721e9",
    "predictions.json": "e3b3a78942378b6de83a10b89ad9d296e1706e2327c42d1574619d954bd5c372",
    "scores.json": "c781d53d64e4f5c17a1b8bd34d922638193ad1627c02c11f6227ac60b1b4b4e3",
    "audit.json": "41abb70c03e3cc25b2b01119ecb5f7695ecf90a9051c483c929d2edb9526f140",
}


def verify(root: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"SHA-256 mismatch for {root / name}: {actual}")


def execute(script: str, *args: str | Path) -> None:
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a fresh directory")
    verify(HERE, SOURCES)
    out.mkdir(parents=True)
    execute("test_run.py")
    execute("run.py", "prepare", "--out", out / "input")
    execute("run.py", "construct", "--problem", out / "input/problem.json",
            "--frame", out / "input/frame-control.json", "--out", out / "run")
    execute("run.py", "score", "--out", out / "run", "--answers", out / "input/answers.json")
    execute("audit.py", "--input", out / "input", "--run", out / "run")
    verify(out / "run", OUTPUTS)
    (out / "VERIFIED.txt").write_text(
        "Frozen source hashes verified. Twelve synthetic tests passed. "
        "All five scientific output files match the retained SHA-256 values.\n")
    print(f"Verified all scientific outputs in {out}")


if __name__ == "__main__":
    main()
