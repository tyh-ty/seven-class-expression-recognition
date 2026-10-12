"""Download the seven-class RAF-DB mirror for local research only."""
from __future__ import annotations
import argparse, hashlib, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FILES = {
    "train-00000-of-00005.parquet": 8984087,
    "train-00001-of-00005.parquet": 8978340,
    "train-00002-of-00005.parquet": 9002062,
    "train-00003-of-00005.parquet": 420171119,
    "train-00004-of-00005.parquet": 1611405396,
}
SHA256 = {
    "train-00000-of-00005.parquet": "7133ca6d8581d87582a64efbed8c062136441258472a93806c07f7e2b4f6877e",
    "train-00001-of-00005.parquet": "f0f20e0c88ee46716aa2cacefa1c60734753825b9e82610c80957c541c8c5e84",
    "train-00002-of-00005.parquet": "56d011c10abd49b1be73e4c7a91e748b29ab626cd9363024c9cbdba326d13b1e",
    "train-00003-of-00005.parquet": "0732442448f0b2e08e062d544616c3834b3dfa54a38d844db28f50fe8b362a1c",
    "train-00004-of-00005.parquet": "31689d57345a78ebf2a30be0f21a5341b1edbcfea8b44d70e17a650a2225a0da",
}
BASE = "https://hf-mirror.com/datasets/deanngkl/raf-db-7emotions/resolve/main/data/"
CHUNK = 16 * 1024 * 1024

def fetch(url: str, path: Path, start: int, end: int) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "Codex-Research", "Range": f"bytes={start}-{end}"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"short range {start}: {len(data)}")
            return start, data
        except Exception:
            if attempt == 5: raise
            time.sleep(2 ** attempt)
    raise AssertionError

def download(name: str, out: Path, workers: int) -> None:
    size = FILES[name]; path = out / name; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size != size: path.unlink()
    if not path.exists():
        with path.open("wb") as f: f.truncate(size)
    ranges = [(start, min(size - 1, start + CHUNK - 1)) for start in range(0, size, CHUNK)]
    done = path.with_suffix(path.suffix + ".ranges")
    complete = {int(x) for x in done.read_text().splitlines()} if done.exists() else set()
    pending = [(a,b) for a,b in ranges if a not in complete]
    url = BASE + name
    print(name, "chunks", len(ranges), "pending", len(pending), flush=True)
    with done.open("a") as log, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, url, path, a, b) for a,b in pending]
        for future in as_completed(futures):
            start, data = future.result()
            with path.open("r+b", buffering=0) as f: f.seek(start); f.write(data)
            log.write(f"{start}\n"); log.flush(); complete.add(start)
            if len(complete) % 10 == 0 or len(complete) == len(ranges): print(name, len(complete), "/", len(ranges), flush=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != SHA256[name]:
        raise RuntimeError(f"SHA256 mismatch for {name}; do not use this data")
    done.unlink(missing_ok=True)
    print(name, 'sha256_verified', digest, flush=True)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=Path(r"D:\RAFDB-source")); p.add_argument("--workers", type=int, default=8); args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in FILES: download(name, args.output, args.workers)
    print("download_complete", args.output, flush=True)

if __name__ == "__main__": main()
