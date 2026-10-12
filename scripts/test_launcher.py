"""Deployment integration checks; no camera access and no feedback collection."""
from __future__ import annotations

import json
import random
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from http.client import HTTPException
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
OPENER = build_opener(ProxyHandler({}))


def free_port() -> int:
    while True:
        port = random.randrange(19000, 29000)
        try:
            with socket.socket() as connection, socket.socket() as adjacent:
                connection.bind(("127.0.0.1", port))
                adjacent.bind(("127.0.0.1", port + 1))
            return port
        except OSError:
            continue


class LauncherIntegrationTest(unittest.TestCase):
    def test_occupied_port_repeated_launch_and_http_inference(self) -> None:
        port = free_port()
        with socket.socket() as occupied, tempfile.TemporaryFile() as log:
            occupied.bind(("127.0.0.1", port))
            occupied.listen()
            command = [
                sys.executable, "-I", "-X", "utf8", "-u", str(ROOT / "scripts/launcher.py"),
                "--no-browser", "--port", str(port),
            ]
            service = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
            try:
                url = f"http://127.0.0.1:{port + 1}"
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if service.poll() is not None:
                        log.seek(0)
                        self.fail(log.read().decode("utf-8", errors="replace"))
                    try:
                        with OPENER.open(url + "/api/health", timeout=0.3) as response:
                            health = json.load(response)
                        break
                    except (OSError, HTTPException):
                        time.sleep(0.1)
                else:
                    self.fail("Service did not become ready after port collision")
                self.assertEqual(health["application"], "seven-class-facial-expression-system")
                self.assertTrue(health["ready"])
                self.assertEqual({m["id"] for m in health["models"]}, {"transformer"})
                self.assertEqual(health["models"][0]["outputShape"], ["batch_size", 7])
                with OPENER.open(url, timeout=5) as response:
                    self.assertIn("摄像头", response.read().decode("utf-8"))

                duplicate = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=15)
                self.assertEqual(duplicate.returncode, 0, duplicate.stderr.decode("utf-8", errors="replace"))
                self.assertIn("服务已经运行", duplicate.stdout.decode("utf-8"))
                self.assertIn(url, duplicate.stdout.decode("utf-8"))
                self.assertIsNone(service.poll())

                import cv2
                import numpy as np
                from urllib.request import Request

                _, encoded = cv2.imencode(".jpg", np.zeros((224, 224, 3), dtype=np.uint8))
                request = Request(url + "/api/analyze?model=transformer", encoded.tobytes(), {"Content-Type": "image/jpeg"})
                with OPENER.open(request, timeout=10) as response:
                    result = json.load(response)
                self.assertEqual(result["model"]["id"], "transformer")
                self.assertEqual(result["faces"], [])
            finally:
                if service.poll() is None:
                    service.terminate()
                service.wait(timeout=10)

    @unittest.skipUnless(sys.platform == "win32", "Windows bootstrap")
    def test_incomplete_package_stops_before_environment_install(self) -> None:
        import shutil

        with tempfile.TemporaryDirectory(prefix="emotion-incomplete-") as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            (root / "deployment").mkdir()
            shutil.copy2(ROOT / "scripts/start-demo.ps1", root / "scripts/start-demo.ps1")
            shutil.copy2(ROOT / "deployment/runtime-lock.json", root / "deployment/runtime-lock.json")
            result = subprocess.run([
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(root / "scripts/start-demo.ps1"), "-PrepareOnly", "-Offline",
            ], capture_output=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("模型缺失", result.stdout.decode("utf-8", errors="replace"))
            self.assertFalse((root / ".runtime").exists())


if __name__ == "__main__":
    unittest.main()
