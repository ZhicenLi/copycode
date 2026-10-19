"""Build the delivery zip and refuse to write it if anything is missing.

The previous packaging step used `robocopy /XD data`, which excluded *every*
directory called `data` -- including the `fusion/data` package. The archive
looked fine (66 entries) but could not be imported. This script verifies
coverage before writing and re-checks the finished archive afterwards.

The archive name carries the snapshot version (`fusion_delivery_v3.zip`) and an
existing archive is never overwritten, so old versions stay downloadable.

Usage:
    python tools/package.py [output.zip] [--force]
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def latest_version() -> int:
    versions = ROOT.parent / "versions"
    numbers = []
    if versions.exists():
        for path in versions.glob("v*"):
            head = path.name.split("_", 1)[0]
            if head[1:].isdigit():
                numbers.append(int(head[1:]))
    return max(numbers, default=0)


DEFAULT_OUT = ROOT.parent / f"fusion_delivery_v{latest_version()}.zip"

EXCLUDED_DIRS = {"__pycache__", "outputs", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
# dataset cache only ever sits at the project root; `fusion/data` is code
EXCLUDED_TOP_LEVEL = {"data"}


def collect(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        if rel.parts and rel.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        yield path, rel.as_posix()


def required_files(root: Path) -> list[str]:
    required = ["run.py", "requirements.txt", "README.md"]
    for package in ("fusion", "tests", "configs", "docs", "tools"):
        folder = root / package
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES:
                    if "__pycache__" not in path.parts:
                        required.append(path.relative_to(root).as_posix())
    return sorted(required)


def main() -> int:
    force = "--force" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    output = Path(positional[0]).resolve() if positional else DEFAULT_OUT
    files = list(collect(ROOT))
    packed = {name for _, name in files}

    missing = [name for name in required_files(ROOT) if name not in packed]
    if missing:
        print("REFUSING to write an incomplete archive; missing:", file=sys.stderr)
        for name in missing:
            print("  -", name, file=sys.stderr)
        return 1

    if output.exists():
        if not force:
            print(
                f"refusing to overwrite {output.name}: bump the version or pass --force",
                file=sys.stderr,
            )
            return 1
        output.unlink()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        broken = [name for name in names if "\\" in name]
        if broken:
            print("archive contains backslash paths:", broken[:5], file=sys.stderr)
            return 1
        missing = [name for name in required_files(ROOT) if name not in set(names)]
        if missing:
            print("archive lost files:", missing[:5], file=sys.stderr)
            return 1

    size_kb = output.stat().st_size / 1024
    print(f"wrote {output} ({size_kb:.0f} KB, {len(names)} entries)")
    data_entries = [n for n in names if n.startswith("fusion/data/")]
    print(f"fusion/data entries: {len(data_entries)} -> {data_entries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
