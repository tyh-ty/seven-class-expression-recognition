"""Fetch a deterministic CAER video pilot via verified ZIP byte ranges.

The original research-use dataset is described at https://caer-dataset.github.io/.
This third-party processed copy is a candidate source, not verified facial truth.
Preserve the source train/validation split and never download test data here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://www.kaggle.com/api/v1/datasets/download/bearmn/processed-caer-video-dataset"
TOTAL_BYTES = 7_668_725_556
LABELS = {"Anger": "anger", "Disgust": "disgust", "Fear": "fear", "Happy": "happiness",
          "Neutral": "neutral", "Sad": "sadness", "Surprise": "surprise"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(r"D:\CAER-source\pilot-v1"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--target-train", type=int, default=100)
    parser.add_argument("--other-train", type=int, default=30)
    parser.add_argument("--target-val", type=int, default=20)
    parser.add_argument("--other-val", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    entries = json.loads((ROOT / "output/dataset-research-processed-caer-video-dataset-entries.json").read_text())
    url_file = ROOT / "output/caer-video-range-url.json"
    url = json.loads(url_file.read_text())["url"] if url_file.exists() else SOURCE
    selected = []
    for split in ("train", "validation"):
        for source_label, label in LABELS.items():
            candidates = [item for item in entries if item["name"].startswith(f"{split}/{source_label}/")
                          and item["name"].endswith(".avi")]
            candidates.sort(key=lambda item: hashlib.sha256(("caer-pilot-v1:" + item["name"]).encode()).hexdigest())
            target = label in ("anger", "fear")
            cap = (args.target_train if target else args.other_train) if split == "train" else (
                args.target_val if target else args.other_val)
            selected.extend(dict(item, split="val" if split == "validation" else "train", label=label)
                            for item in candidates[:cap])
    args.output.mkdir(parents=True, exist_ok=True)

    def download(item: dict) -> dict:
        name = item["name"]
        identifier = Path(name).stem
        if not identifier.isdigit() or not 0 < item["size"] < 50 * 1024 * 1024:
            raise ValueError(f"Unexpected entry: {name}")
        destination = args.output / item["split"] / item["label"] / (identifier + ".avi")
        payload = None
        if destination.exists():
            existing = destination.read_bytes()
            if len(existing) == item["size"] and zlib.crc32(existing) == item["crc"]:
                payload = existing
        for attempt in range(3):
            if payload is not None:
                break
            start = item["offset"]
            end = min(TOTAL_BYTES - 1, start + item["compressed"] + 131100)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                    "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=25) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{TOTAL_BYTES}":
                        raise ValueError("Server did not honor the exact byte range")
                    raw = response.read(end - start + 2)
                if len(raw) != end - start + 1 or raw[:4] != b"PK\x03\x04":
                    raise ValueError("Truncated or invalid ZIP entry")
                _, _, flags, method, _, _, _, _, _, name_length, extra_length = struct.unpack_from("<I5H3I2H", raw)
                archived_name = raw[30:30 + name_length].decode("utf-8")
                if flags & 1 or method != item["method"] or archived_name != name:
                    raise ValueError("ZIP entry does not match the audited index")
                offset = 30 + name_length + extra_length
                compressed = raw[offset:offset + item["compressed"]]
                payload = zlib.decompress(compressed, -15) if method == 8 else compressed if method == 0 else None
                if payload is None or len(payload) != item["size"] or zlib.crc32(payload) != item["crc"]:
                    payload = None
                    raise ValueError("ZIP length/CRC mismatch")
            except Exception as error:
                if attempt == 2:
                    detail = str(error).replace(url, "[download URL]")[:160]
                    raise RuntimeError(f"Download failed: {name}: {type(error).__name__}: {detail}") from None
                time.sleep(attempt + 1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(payload)
        elif destination.read_bytes() != payload:
            destination.write_bytes(payload)
        return {"source_entry": name, "split": item["split"], "label": item["label"],
                "file": str(destination.relative_to(args.output)), "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(), "archive_crc32_verified": True}

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item): item for item in selected}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"entry": futures[future]["name"], "error": str(error)})
            completed = len(results) + len(failures)
            if completed % 20 == 0 or completed == len(selected):
                print(f"downloaded={len(results)}/{len(selected)} failed={len(failures)}", flush=True)
    report = {
        "official_source": "https://caer-dataset.github.io/",
        "download_source": SOURCE, "mirror_archive_bytes": TOTAL_BYTES,
        "terms": "Original CAER public research-use permission; source video copyrights remain with owners. "
                 "Third-party processed mirror, original-file equivalence not independently established.",
        "scope": "Movie/TV scene-emotion video pilot for facial-label screening. Not webcam or microexpression ground truth.",
        "split": "Source train and validation retained. No test clips selected. Identity-disjointness not established.",
        "counts": dict(sorted(Counter(f"{r['split']}/{r['label']}" for r in results).items())),
        "total_bytes": sum(r["bytes"] for r in results), "failures": failures,
        "videos": sorted(results, key=lambda r: r["source_entry"]),
    }
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "total_bytes", "failures")}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
