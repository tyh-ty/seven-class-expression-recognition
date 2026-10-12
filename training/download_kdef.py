from __future__ import annotations

import argparse
import subprocess
import time
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://www.kdef.se/download/KDEF_and_AKDEF.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official KDEF archive in parallel")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path(r"D:\KDEF\KDEF_and_AKDEF.zip"))
    parser.add_argument("--connections", type=int, default=8)
    return parser.parse_args()


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers["Content-Length"])


def download_parts(
    url: str, jobs: list[tuple[Path, int, int]], total_size: int
) -> None:
    processes: list[tuple[Path, int, subprocess.Popen[bytes]]] = []
    for path, start, end in jobs:
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            continue
        process = subprocess.Popen(
            [
                "curl.exe",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--silent",
                "--show-error",
                "--range",
                f"{start}-{end}",
                "--output",
                str(path),
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        processes.append((path, expected, process))

    while any(process.poll() is None for _, _, process in processes):
        downloaded = sum(path.stat().st_size for path, _, _ in processes if path.exists())
        print(f"downloaded {downloaded:,}/{total_size:,} bytes", flush=True)
        time.sleep(10)

    for path, expected, process in processes:
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"curl failed for {path.name}: {stderr.decode(errors='replace').strip()}"
            )
        actual = path.stat().st_size
        if actual != expected:
            raise RuntimeError(f"Incomplete part {path.name}: {actual}/{expected} bytes")


def main() -> None:
    args = parse_args()
    if args.connections < 1:
        raise ValueError("--connections must be positive")
    size = remote_size(args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_root = args.output.with_suffix(args.output.suffix + ".parts")
    part_root.mkdir(exist_ok=True)
    chunk_size = (size + args.connections - 1) // args.connections
    jobs: list[tuple[Path, int, int]] = []
    for index in range(args.connections):
        start = index * chunk_size
        end = min(size - 1, start + chunk_size - 1)
        if start <= end:
            jobs.append((part_root / f"part-{index:02d}", start, end))

    download_parts(args.url, jobs, size)

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for path, _, _ in jobs:
            with path.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    destination.write(chunk)
    if temporary.stat().st_size != size:
        raise RuntimeError(f"Combined file has wrong size: {temporary.stat().st_size}/{size}")
    temporary.replace(args.output)
    for path, _, _ in jobs:
        path.unlink()
    part_root.rmdir()
    print(f"download complete: {args.output} ({size:,} bytes)")


if __name__ == "__main__":
    main()
