"""Keeps secrets and oversized files out of the repository."""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS = [
    re.compile(r"(?i)api[_-]?key\s*=\s*[\"'][A-Za-z0-9_\-]{16,}[\"']"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY-----"),
]


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / line for line in out.splitlines() if line]
    except Exception:  # not a git checkout (for example an exported archive)
        files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    return [p for p in files if p.is_file()]


class HygieneTests(unittest.TestCase):
    def test_no_hardcoded_secrets(self):
        for path in tracked_files():
            if path.suffix in {".pt", ".png", ".jpg", ".csv"} or path.name == "test_repo_hygiene.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRETS:
                self.assertIsNone(pattern.search(text), f"possible secret in {path.relative_to(ROOT)}")

    def test_no_file_exceeds_github_limit(self):
        for path in tracked_files():
            self.assertLess(path.stat().st_size, 50 * 1024 * 1024, f"{path.relative_to(ROOT)} is too large to commit")

    def test_no_machine_specific_paths(self):
        for path in tracked_files():
            if path.suffix != ".py" or path.name == "test_repo_hygiene.py":
                continue
            self.assertNotRegex(path.read_text(encoding="utf-8"), r"C:[/\\]Users", path.name)


if __name__ == "__main__":
    unittest.main()
