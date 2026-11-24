"""Update SHA256SUMS.txt from the Git index after changing public files.

Run ``git add .`` first. Ignored/generated files and this manifest are omitted.
"""

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    paths = sorted(Path(name.decode()) for name in names if name and name != b"SHA256SUMS.txt")
    if not paths:
        raise RuntimeError("No staged/tracked files; run git add . first")
    lines = []
    for relative in paths:
        file = ROOT / relative
        if not file.is_file():
            raise FileNotFoundError(file)
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {relative.as_posix()}\n")
    (ROOT / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {len(paths)} checksums to SHA256SUMS.txt")


if __name__ == "__main__":
    main()
