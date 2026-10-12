"""Start a checked local service, reusing this installation when already running."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import webbrowser
from http.client import HTTPException
from http.server import HTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def running_installation(port: int, instance: str) -> bool:
    # A machine-wide HTTP proxy must never intercept the localhost readiness probe.
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=0.3) as response:
            health = json.load(response)
        return (
            health.get("application") == "seven-class-facial-expression-system"
            and health.get("instance") == instance
            and health.get("ready") is True
        )
    except (OSError, URLError, HTTPException, ValueError, AttributeError):
        return False


class LocalHTTPServer(HTTPServer):
    # Windows SO_REUSEADDR can allow two listeners on one port. Prefer exclusivity.
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def open_page(url: str, enabled: bool) -> None:
    if enabled:
        try:
            if not webbrowser.open(url):
                print(f"请手动在浏览器打开：{url}", flush=True)
        except Exception as error:
            print(f"浏览器未能自动打开：{error}。请访问 {url}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65515:
        parser.error("port must be between 1024 and 65515")

    import mimetypes
    import numpy as np
    from server import ApplicationHandler, EmotionRecognizer, INSTANCE_ID

    if args.check:
        recognizer = EmotionRecognizer()
        # Exercise the actual Transformer graph, not just its imports.
        face = np.full((224, 224, 3), 128, dtype=np.uint8)
        for runtime in recognizer.classifiers.values():
            spec = runtime["spec"]
            tensor = recognizer._emotion_tensor(face)
            scores = runtime["session"].run(None, {runtime["input_name"]: tensor})[0]
            expected_shape = (1, spec.emotion_count)
            if scores.shape != expected_shape or not np.isfinite(scores).all():
                raise RuntimeError(f"Invalid classifier output: {spec.name}")
        recognizer.analyze(face)
        print("环境与模型自检通过（CPU 推理）。", flush=True)
        return

    ports = range(args.port, args.port + 21)
    for port in ports:
        if running_installation(port, INSTANCE_ID):
            url = f"http://127.0.0.1:{port}"
            print(f"本项目服务已经运行：{url}", flush=True)
            open_page(url, not args.no_browser)
            return

    httpd = None
    for port in ports:
        try:
            httpd = LocalHTTPServer(("127.0.0.1", port), ApplicationHandler)
            break
        except OSError as error:
            if error.errno not in (13, 98, 10013, 10048) and getattr(error, "winerror", None) not in (10013, 10048):
                raise
    if httpd is None:
        raise RuntimeError("没有可用端口，请关闭占用端口的程序或用 -Port 指定其他端口。")
    try:
        print("正在加载识别模型…", flush=True)
        ApplicationHandler.recognizer = EmotionRecognizer()
        mimetypes.add_type("text/javascript", ".js")
        url = f"http://127.0.0.1:{httpd.server_port}"
        print(f"服务已就绪：{url}\n保留此窗口；关闭窗口或按 Ctrl+C 停止服务。", flush=True)
        open_page(url, not args.no_browser)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
