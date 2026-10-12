"""Resumable ranged download of the authors' published MELD raw video archive."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SOURCE = "https://hf-mirror.com/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
SIZE = 10_878_146_150
SHA256 = "a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e"
CHUNK = 16 * 1024 * 1024


def fetch(index: int, path: Path) -> int:
    first = index * CHUNK
    last = min(SIZE - 1, first + CHUNK - 1)
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                SOURCE, headers={"User-Agent": "Codex-Research", "Range": f"bytes={first}-{last}"}
            )
            with urllib.request.urlopen(request, timeout=100) as response:
                expected = f"bytes {first}-{last}/{SIZE}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"Unexpected response for {index}: {response.status} {response.headers.get('Content-Range')}")
                payload = response.read()
            if len(payload) != last - first + 1:
                raise RuntimeError(f"Truncated range {index}: {len(payload)}")
            with path.open("r+b", buffering=0) as output:
                output.seek(first)
                output.write(payload)
            return index
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(r"D:\MELD-source\MELD.Raw.tar.gz"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise ValueError("--workers must be between 1 and 24")
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = path.with_suffix(path.suffix + ".ranges")
    if path.exists() and path.stat().st_size != SIZE:
        raise RuntimeError(f"Existing archive has unexpected size: {path.stat().st_size}")
    if not path.exists():
        with path.open("wb") as target:
            target.truncate(SIZE)
    count = math.ceil(SIZE / CHUNK)
    completed = set()
    if progress.exists():
        for line in progress.read_text(encoding="ascii").splitlines():
            if not re.fullmatch(r"\d+", line) or int(line) >= count:
                raise RuntimeError("Invalid progress file")
            completed.add(int(line))
    pending = [index for index in range(count) if index not in completed]
    print(f"archive={path} chunks={count} done={len(completed)} pending={len(pending)}", flush=True)
    with progress.open("a", encoding="ascii") as log:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, index, path) for index in pending]
            for future in as_completed(futures):
                index = future.result()
                log.write(f"{index}\n")
                log.flush()
                completed.add(index)
                if len(completed) % 20 == 0 or len(completed) == count:
                    print(f"downloaded={len(completed)}/{count} ({len(completed)/count:.1%})", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    print(f"sha256={digest.hexdigest()}", flush=True)
    if digest.hexdigest() != SHA256:
        raise RuntimeError("MELD archive checksum mismatch; do not extract")
    progress.unlink()
    print("download_complete_verified", flush=True)


if __name__ == "__main__":
    main()
