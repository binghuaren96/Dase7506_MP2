"""Data setup verifies the fixed allowlist before touching its destination."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_data


class FetchDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.payload = {name: ("synthetic " + name).encode() for name in fetch_data.DATA_FILES}
        self.manifest = {"protocol": fetch_data.PROTOCOL,
                         "sha256": {name: hashlib.sha256(value).hexdigest()
                                    for name, value in self.payload.items()}}
        self.manifest_path = self.data / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.data / "LICENSE.txt").write_text("retained license", encoding="utf-8")
        (self.data / "NOTICE.txt").write_text("retained notice", encoding="utf-8")
        self.root_patch = patch.object(fetch_data, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def archive(self, *, corrupt=None, extra=(), prefix="DASE7506-MP2-main/"):
        path = self.root / "course.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in self.payload.items():
                archive.writestr(prefix + "code/data/" + name, b"bad bytes" if name == corrupt else value)
            for name, value in extra:
                archive.writestr(name, value)
        return path

    def preserved_metadata(self):
        return {name: (self.data / name).read_bytes()
                for name in ("manifest.json", "LICENSE.txt", "NOTICE.txt")}

    def test_valid_archive_install_and_existing_matching_noop(self):
        before = self.preserved_metadata()
        with patch.object(fetch_data, "_download", side_effect=AssertionError("No download permitted")):
            result = fetch_data.fetch_data(self.data, self.archive())
            self.assertEqual(result["files_written"], 9)
            original_times = {name: (self.data / name).stat().st_mtime_ns for name in self.payload}
            # A complete matching directory needs neither an archive nor network.
            result = fetch_data.fetch_data(self.data)
        self.assertEqual(result["status"], "already_complete")
        for name, value in self.payload.items():
            self.assertEqual((self.data / name).read_bytes(), value)
            self.assertEqual((self.data / name).stat().st_mtime_ns, original_times[name])
        self.assertEqual(self.preserved_metadata(), before)
        self.assertEqual(list((self.root / "runs").iterdir()), [])

    def test_last_hash_failure_creates_no_destination_files(self):
        before = self.preserved_metadata()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            fetch_data.fetch_data(self.data, self.archive(corrupt=fetch_data.DATA_FILES[-1]))
        self.assertFalse((self.data / "candidates").exists())
        self.assertEqual({p.name for p in self.data.iterdir()}, set(before))
        self.assertEqual(self.preserved_metadata(), before)
        self.assertEqual(list((self.root / "runs").iterdir()), [])

    def test_existing_conflict_is_not_overwritten_or_partially_installed(self):
        name = fetch_data.DATA_FILES[-1]
        (self.data / name).write_bytes(b"keep my file")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            fetch_data.fetch_data(self.data, self.archive())
        self.assertEqual((self.data / name).read_bytes(), b"keep my file")
        self.assertFalse((self.data / "candidates").exists())

    def test_partial_matching_install_keeps_existing_file_untouched(self):
        name = fetch_data.DATA_FILES[-1]
        existing = self.data / name
        existing.write_bytes(self.payload[name])
        before = existing.stat().st_mtime_ns
        result = fetch_data.fetch_data(self.data, self.archive())
        self.assertEqual(result["files_written"], 8)
        self.assertEqual(existing.stat().st_mtime_ns, before)

    def test_duplicate_or_missing_zip_payload_refused_before_install(self):
        path = self.archive(extra=(("other/code/data/" + fetch_data.DATA_FILES[0], b"duplicate"),))
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate"):
            fetch_data.fetch_data(self.data, path)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("code/data/" + fetch_data.DATA_FILES[0], self.payload[fetch_data.DATA_FILES[0]])
        with self.assertRaisesRegex(ValueError, "missing expected files"):
            fetch_data.fetch_data(self.data, path)
        self.assertFalse((self.data / "candidates").exists())

    def test_only_known_payloads_copied_and_traversal_members_ignored(self):
        archive = self.archive(prefix="", extra=(
            ("../code/data/gsm8k_train.jsonl", b"traversal"),
            ("elsewhere/code/data/../../outside.txt", b"traversal"),
            ("code/data/NOTICE.txt", b"replace notice"),
            ("code/data/unexpected.py", b"not a dataset"),
        ))
        before = self.preserved_metadata()
        fetch_data.fetch_data(self.data, archive)
        self.assertEqual(self.preserved_metadata(), before)
        self.assertFalse((self.data / "unexpected.py").exists())
        self.assertFalse((self.root / "outside.txt").exists())
        self.assertEqual({p.relative_to(self.data).as_posix() for p in self.data.rglob("*") if p.is_file()},
                         set(fetch_data.DATA_FILES) | set(before))

    def test_wrong_protocol_or_missing_expected_hash_rejected(self):
        for change in ({"protocol": "different"}, {"sha256": {}}):
            with self.subTest(change=change):
                self.manifest_path.write_text(json.dumps(dict(self.manifest, **change)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    fetch_data.fetch_data(self.data, self.archive())
                self.assertFalse((self.data / "candidates").exists())


if __name__ == "__main__":
    unittest.main()
