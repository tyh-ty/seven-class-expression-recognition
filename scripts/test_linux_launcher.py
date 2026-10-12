"""Exercise the offline Bash bootstrap with a fake Python, without Linux."""
import hashlib
import io
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = (r"C:\Program Files\Git\usr\bin\bash.exe" if os.name == "nt" else shutil.which("bash"))


def bash_path(path):
    value = str(path).replace("\\", "/")
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


class LinuxBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "情绪识别 space 'quote' $cash %25"
        self.root.mkdir()
        self.bin = self.root / "mock-bin"
        self.bin.mkdir()
        (self.root / "deployment").mkdir()
        (self.root / "scripts").mkdir()
        shutil.copyfile(ROOT / "start-linux.sh", self.root / "start.sh")
        (self.root / "requirements-runtime-linux.txt").write_text("fake-package==1\n")
        self.log = self.root / "calls.log"
        self.env = {k: v for k, v in os.environ.items() if k.upper() != "PATH"}
        self.env["MOCK_LOG"] = bash_path(self.log)
        self.env["PATH"] = (
            str(self.bin) + r";C:\Program Files\Git\usr\bin;C:\Program Files\Git\mingw64\bin"
            if os.name == "nt" else str(self.bin) + ":/usr/bin:/bin"
        )
        self.write_command("uname", 'if [[ "$1" == -s ]]; then echo Linux; else echo x86_64; fi')
        # Flock is not available in Git Bash; concurrency itself needs Linux QA.
        self.write_command("flock", "exit 0")
        self.write_command("python3", 'echo unexpected-system-python >> "$MOCK_LOG"; exit 99')
        for name in ("curl", "wget"):
            self.write_command(name, 'echo unexpected-download >> "$MOCK_LOG"; exit 99')
        fake_python = b'''#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$MOCK_LOG"
if [[ "$*" == *' -m venv '* ]]; then
  mkdir -p "$4/bin"
  cp -- "$0" "$4/bin/python"
  chmod +x "$4/bin/python"
elif [[ "$*" == *' -m pip '* ]]; then
  [[ "$*" == *'--no-index'* && "$*" == *'--no-cache-dir'* && "$*" == *'--isolated'* ]] || exit 24
  [[ "${MOCK_FAIL_PIP:-0}" == 0 ]] || exit 23
fi
'''
        self.cache = self.root / "deployment/linux-cache"
        (self.cache / "wheels").mkdir(parents=True)
        archive_path = self.cache / "python.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            entry = tarfile.TarInfo("python/bin/python3")
            entry.mode = 0o755
            entry.size = len(fake_python)
            archive.addfile(entry, io.BytesIO(fake_python))
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        wheel = self.cache / "wheels/fake.whl"
        wheel.write_bytes(b"fake-wheel")
        (self.cache / "SHA256SUMS").write_text(
            f"{digest}  python.tar.gz\n"
            f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  wheels/fake.whl\n",
            encoding="utf-8", newline="\n",
        )
        (self.root / "deployment/linux-python.env").write_text(
            "LINUX_PYTHON_VERSION='test'\nLINUX_PYTHON_ARCHIVE='python.tar.gz'\n"
            "LINUX_PYTHON_URL='https://example.invalid/python.tar.gz'\n"
            f"LINUX_PYTHON_SHA256='{digest}'\n", encoding="utf-8", newline="\n",
        )

    def write_command(self, name, code):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + code + "\n", encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def launch(self, **environment):
        return subprocess.run(
            [BASH, bash_path(self.root / "start.sh"), "--no-browser", "--port", "8875"],
            env={**self.env, **environment}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", timeout=30,
        )

    def test_first_install_and_reuse_without_network_or_system_python(self):
        first = self.launch()
        self.assertEqual(first.returncode, 0, first.stdout)
        calls = self.log.read_text(encoding="utf-8")
        self.assertNotIn("download", calls)
        self.assertIn("-m pip --isolated install", calls)
        self.assertIn("--find-links", calls)
        self.assertIn("--check", calls)
        self.assertIn("--no-browser --port 8875", calls)
        self.assertNotIn("unexpected-system-python", calls)
        self.log.write_text("")
        second = self.launch(MOCK_FAIL_PIP="1")
        self.assertEqual(second.returncode, 0, second.stdout)
        calls = self.log.read_text(encoding="utf-8")
        self.assertNotIn("download", calls)
        self.assertNotIn("-m pip", calls)

    def test_missing_or_corrupt_python_never_executes_or_downloads(self):
        archive = self.cache / "python.tar.gz"
        original = archive.read_bytes()
        archive.unlink()
        for damaged in (False, True):
            if damaged:
                archive.write_bytes(b"corrupt archive")
            result = self.launch()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("离线资源缺失或校验失败", result.stdout)
            self.assertFalse((self.root / ".venv").exists())
            self.assertFalse((self.root / ".runtime-linux/python").exists())
            self.assertFalse(self.log.exists())
        archive.write_bytes(original)
        retry = self.launch()
        self.assertEqual(retry.returncode, 0, retry.stdout)

    def test_failed_dependency_install_can_retry(self):
        failed = self.launch(MOCK_FAIL_PIP="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((self.root / ".venv/.runtime-requirements-installed").exists())
        self.assertNotIn("--no-browser", self.log.read_text(encoding="utf-8"))
        retried = self.launch()
        self.assertEqual(retried.returncode, 0, retried.stdout)

    def test_changed_requirements_and_broken_environment_are_repaired(self):
        first = self.launch()
        self.assertEqual(first.returncode, 0, first.stdout)
        requirements = self.root / "requirements-runtime-linux.txt"
        requirements.write_text("fake-package==2\n")
        self.log.write_text("")
        changed = self.launch()
        self.assertEqual(changed.returncode, 0, changed.stdout)
        self.assertIn("-m pip --isolated install", self.log.read_text(encoding="utf-8"))
        (self.root / ".venv/bin/python").unlink()
        self.log.write_text("")
        repaired = self.launch()
        self.assertEqual(repaired.returncode, 0, repaired.stdout)
        self.assertNotIn("download", self.log.read_text(encoding="utf-8"))
        self.assertTrue(list(self.root.glob(".venv.previous.*")))

    def test_missing_or_corrupt_wheel_stops_before_pip_and_service(self):
        wheel = self.cache / "wheels/fake.whl"
        wheel.unlink()
        for damaged in (False, True):
            if damaged:
                wheel.write_bytes(b"damaged")
            self.log.write_text("")
            result = self.launch()
            self.assertNotEqual(result.returncode, 0)
            calls = self.log.read_text(encoding="utf-8")
            self.assertNotIn("-m pip", calls)
            self.assertNotIn("--no-browser", calls)
            self.assertNotIn("download", calls)
            self.assertFalse((self.root / ".venv/.runtime-requirements-installed").exists())

    def test_desktop_starts_without_python_and_handles_file_uri(self):
        from urllib.parse import quote
        desktop = (ROOT / "deployment/linux/启动情绪识别.desktop").read_text(encoding="utf-8")
        value = next(line[5:] for line in desktop.splitlines() if line.startswith("Exec="))
        # Unescape the Desktop Entry string and argument layers. Run the resulting
        # Bash code from a file: Windows argv parsing changes backslashes in -c.
        replacements = {"\\": "\\", "s": " ", "n": "\n", "t": "\t", "r": "\r"}
        value = re.sub(r"\\([\\sntr])", lambda m: replacements[m[1]], value)
        self.assertTrue(value.startswith("bash -c "))
        arguments = shlex.split(value.replace("%%", "%"))
        self.assertEqual(arguments[-1], "%k")
        command = arguments[2].replace("\\$", "$").replace("\\`", "`")
        decoded = self.root / "desktop-command.sh"
        decoded.write_text(command + "\n", encoding="utf-8", newline="\n")
        (self.root / "start.sh").write_text('printf success > "$(dirname -- "$0")/clicked"\n', encoding="utf-8", newline="\n")
        location = bash_path(self.root / "启动情绪识别.desktop")
        for target in [location, "file://" + quote(location)]:
            result = subprocess.run([BASH, bash_path(decoded), target], env=self.env, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
            self.assertEqual((self.root / "clicked").read_text(), "success")
            (self.root / "clicked").unlink()


if __name__ == "__main__":
    unittest.main()
