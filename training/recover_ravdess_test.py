from __future__ import annotations

import argparse
import concurrent.futures
import json
import struct
import subprocess
import zlib
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


ARCHIVE_SIZES = {22: 560907466, 24: 595799165}


def entries_from_tail(tail: bytes, total: int) -> list[dict]:
    end_offset = tail.rfind(b"PK\x05\x06")
    if end_offset < 0:
        raise ValueError("ZIP end record is missing")
    end = struct.unpack("<4s4H2IH", tail[end_offset:end_offset + 22])
    offset = end[6] - (total - len(tail))
    result = []
    while tail[offset:offset + 4] == b"PK\x01\x02":
        values = struct.unpack("<4s6H3I5H2I", tail[offset:offset + 46])
        name = tail[offset + 46:offset + 46 + values[10]].decode("utf-8")
        result.append({"name": name, "method": values[4], "crc32": values[7],
                       "compressed_size": values[8], "size": values[9], "offset": values[-1]})
        offset += 46 + values[10] + values[11] + values[12]
    if len(result) != end[4]:
        raise ValueError("ZIP central directory is incomplete")
    return result


def decompress_entry(payload: bytes, entry: dict) -> bytes:
    header = struct.unpack("<4s5H3I2H", payload[:30])
    if header[0] != b"PK\x03\x04":
        raise ValueError(f"Missing local header: {entry['name']}")
    offset = 30 + header[-2] + header[-1]
    data = payload[offset:offset + entry["compressed_size"]]
    if len(data) != entry["compressed_size"]:
        raise ValueError(f"Truncated entry: {entry['name']}")
    decoded = zlib.decompress(data, -15) if entry["method"] == 8 else data
    if len(decoded) != entry["size"] or zlib.crc32(decoded) & 0xFFFFFFFF != entry["crc32"]:
        raise ValueError(f"CRC/length verification failed: {entry['name']}")
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover full audiovisual test speech clips from official RAVDESS ZIP ranges")
    parser.add_argument("--root", type=Path, default=Path(r"D:\RAVDESS"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for actor, total in ARCHIVE_SIZES.items():
        original = args.root / f"Video_Speech_Actor_{actor:02d}.zip"
        tail = (args.root / f"actor{actor}-zip-tail.bin").read_bytes()
        entries = [entry for entry in entries_from_tail(tail, total)
                   if Path(entry["name"]).name.startswith("01-01-")
                   and Path(entry["name"]).name.split("-")[2] != "02"]
        if len(entries) != 52:
            raise ValueError(f"Unexpected full audiovisual seven-class count: {len(entries)}")
        output = args.root / "recovered-av-speech" / f"Actor_{actor:02d}"
        output.mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/records/1188976/files/Video_Speech_Actor_{actor:02d}.zip?download=1"

        def recover(entry: dict) -> dict:
            destination = output / Path(entry["name"]).name
            if destination.is_file():
                data = destination.read_bytes()
                if len(data) == entry["size"] and zlib.crc32(data) & 0xFFFFFFFF == entry["crc32"]:
                    return {**entry, "source": "verified-recovery-cache"}
            start = entry["offset"]
            # Include enough bytes for the local header and its small extra field.
            end = min(total - 1, start + entry["compressed_size"] + 2047)
            if start + entry["compressed_size"] + 30 + len(entry["name"]) <= original.stat().st_size:
                with original.open("rb") as stream:
                    stream.seek(start)
                    payload = stream.read(end - start + 1)
                source = "original-complete-entry"
            else:
                part = destination.with_suffix(".compressed-part")
                result = subprocess.run(
                    ["curl.exe", "-L", "--fail", "--retry", "3", "--retry-all-errors",
                     "--max-time", "180", "--silent", "--show-error", "--range", f"{start}-{end}",
                     "--output", str(part), url], capture_output=True,
                )
                if result.returncode:
                    raise RuntimeError(result.stderr.decode(errors="replace"))
                payload = part.read_bytes()
                source = "official-zenodo-byte-range"
            decoded = decompress_entry(payload, entry)
            destination.write_bytes(decoded)
            part = destination.with_suffix(".compressed-part")
            if part.exists():
                part.unlink()
            return {**entry, "source": source}

        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(recover, entry) for entry in entries]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                records.append(record)
                print(f"actor={actor} recovered={len(records)}/{len(entries)} crc_verified {Path(record['name']).name}", flush=True)
        archive_path = args.root / f"Video_Speech_Actor_{actor:02d}.av-seven-recovered.zip"
        with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
            for entry in entries:
                archive.write(output / Path(entry["name"]).name, entry["name"])
        report = {"actor": actor, "official_url": url, "scope": "52 full audiovisual speech videos, seven emotions; calm excluded",
                  "original_zip_complete": False, "recovered_videos": sorted(records, key=lambda item: item["name"])}
        (output / "recovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"verified recovered archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
