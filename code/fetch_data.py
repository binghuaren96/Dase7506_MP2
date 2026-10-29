"""Install the unchanged course data from the official ZIP, with SHA256 checks.

Network access is used only by this explicit setup command. The submitted
prediction method never imports this module or downloads anything.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "code" / "data"
OFFICIAL_ZIP = "https://github.com/xudongwu-0/DASE7506-MP2/archive/bb801f4f2e7629d270df765ab12c333b2dc616bb.zip"
PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
MAX_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
DATA_FILES = tuple(
    name
    for split in ("train", "validation", "test")
    for name in (
        f"candidates/gsm8k_{split}.jsonl",
        f"candidates/gsm8k_{split}.npz",
        f"gsm8k_{split}.jsonl",
    )
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected(data):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected course data protocol in manifest.json")
    expected = {name: manifest.get("sha256", {}).get(name) for name in DATA_FILES}
    if any(not isinstance(value, str) or len(value) != 64
           or any(c not in "0123456789abcdef" for c in value) for value in expected.values()):
        raise ValueError("The supplied manifest must include SHA256 hashes for all nine data files")
    return expected


def _missing(data, expected):
    missing = []
    for name, digest in expected.items():
        target = data / name
        if not target.resolve().is_relative_to(data):
            raise ValueError(f"Data destination escapes its directory: {name}")
        if target.is_symlink():
            raise ValueError(f"Refusing to overwrite conflicting data: {name}")
        if target.exists():
            if not target.is_file() or _sha256(target) != digest:
                raise ValueError(f"Refusing to overwrite conflicting data: {name}")
        else:
            missing.append(name)
    return missing


def _download(destination):
    request = urllib.request.Request(OFFICIAL_ZIP, headers={"User-Agent": "DASE7506-MP2-data-setup"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as output:
        total = 0
        while block := response.read(CHUNK_BYTES):
            total += len(block)
            if total > MAX_BYTES:
                raise ValueError("Course ZIP exceeds the 256 MiB download limit")
            output.write(block)


def _members(archive):
    """Find only whitelisted files; never use extract/extractall or ZIP paths."""
    selected = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (member.is_dir() or path.is_absolute() or "\\" in member.filename
                or any(part in (".", "..", "") for part in member.filename.split("/"))
                or stat.S_ISLNK(member.external_attr >> 16)):
            continue
        for name in DATA_FILES:
            suffix = ("code", "data", *PurePosixPath(name).parts)
            if tuple(path.parts[-len(suffix):]) == suffix:
                if name in selected:
                    raise ValueError(f"Ambiguous duplicate file in course ZIP: {name}")
                selected[name] = member
                break
    if set(selected) != set(DATA_FILES):
        absent = sorted(set(DATA_FILES) - set(selected))
        raise ValueError("Course ZIP is missing expected files: " + ", ".join(absent))
    if sum(member.file_size for member in selected.values()) > MAX_BYTES:
        raise ValueError("Course data exceeds the 256 MiB uncompressed limit")
    return selected


def fetch_data(data_dir=DATA, archive_path=None):
    """Verify every ZIP payload before copying any file into the data directory.

    Existing matching files are left untouched. A conflicting file is an error,
    not an invitation to replace it. Manifest, LICENSE and NOTICE are never
    copied from the archive or modified by this function.
    """
    data = Path(data_dir).expanduser().resolve(strict=True)
    expected = _expected(data)
    missing = _missing(data, expected)
    if not missing:
        return {"status": "already_complete", "files_written": 0, "data_dir": str(data)}
    source = Path(archive_path).expanduser().resolve(strict=True) if archive_path is not None else None
    if source is not None and source.stat().st_size > MAX_BYTES:
        raise ValueError("Course ZIP exceeds the 256 MiB archive limit")
    project = ROOT.resolve()
    runs = (project / "runs").resolve()
    if runs != project / "runs":
        raise ValueError("Project runs/ must not redirect temporary staging elsewhere")
    runs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fetch_data_", dir=runs) as temporary:
        staging = Path(temporary).resolve()
        if staging.parent != runs:
            raise ValueError("Temporary data staging must remain directly inside project runs/")
        if source is None:
            source = staging / "course.zip"
            _download(source)
        with zipfile.ZipFile(source) as archive:
            selected = _members(archive)
            for name, digest in expected.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                hasher = hashlib.sha256()
                total = 0
                with archive.open(selected[name]) as incoming, target.open("xb") as output:
                    while block := incoming.read(CHUNK_BYTES):
                        total += len(block)
                        if total > MAX_BYTES:
                            raise ValueError(f"Oversized course data file: {name}")
                        hasher.update(block)
                        output.write(block)
                if hasher.hexdigest() != digest:
                    raise ValueError(f"Course ZIP SHA256 mismatch: {name}")
        # Check all destinations again before the first installation write.
        missing = _missing(data, expected)
        for name in missing:
            target = data / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with (staging / name).open("rb") as incoming, target.open("xb") as output:
                shutil.copyfileobj(incoming, output, CHUNK_BYTES)
    return {"status": "complete", "files_written": len(missing), "data_dir": str(data)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded official course ZIP")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args(argv)
    try:
        result = fetch_data(args.data_dir, args.archive)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Data setup failed: {error}\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
