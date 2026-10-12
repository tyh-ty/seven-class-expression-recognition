"""Build the portable Linux archive with Unix executable modes, even on Windows."""
from __future__ import annotations

import hashlib
import io
import json
import argparse
from pathlib import Path
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "start.sh": "start-linux.sh",
    "启动情绪识别.desktop": "deployment/linux/启动情绪识别.desktop",
    "使用说明.txt": "deployment/linux/使用说明.txt",
    "requirements-runtime-linux.txt": "requirements-runtime-linux.txt",
    "deployment/linux-python.env": "deployment/linux-python.env",
    "deployment/linux-runtime-lock.json": "deployment/linux-runtime-lock.json",
    "server.py": "server.py",
    "scripts/launcher.py": "scripts/launcher.py",
    "training/emotion_preprocessing.py": "training/emotion_preprocessing.py",
    "models/face_detection_yunet_2023mar.onnx": "models/face_detection_yunet_2023mar.onnx",
    "models/vit_emotion_seven_fp32.onnx": "models/vit_emotion_seven_fp32.onnx",
    **{f"web/{name}": f"web/{name}" for name in (
        "app.js", "emotion-speech.js", "index.html", "styles.css",
    )},
}


def payload(source: str) -> bytes:
    path = ROOT / source
    if path.suffix in {".onnx", ".whl", ".gz"}:
        return path.read_bytes()
    return path.read_text(encoding="utf-8-sig").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing build assets from their pinned URLs")
    args = parser.parse_args()
    lock = json.loads((ROOT / "deployment/linux-runtime-lock.json").read_text(encoding="utf-8"))
    files = dict(FILES)
    checksums = []
    for asset in lock["assets"]:
        relative = "deployment/linux-cache/" + asset["file"]
        path = ROOT / relative
        valid = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        if not valid and args.download:
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(asset["url"], timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != asset["sha256"]:
                raise RuntimeError(f"Download checksum mismatch: {asset['file']}")
            path.write_bytes(data)
            valid = True
        if not valid:
            raise RuntimeError(f"Offline asset missing or corrupt: {path}. Re-run with --download to restore it.")
        files[relative] = relative
        checksums.append(f"{asset['sha256']}  {asset['file']}\n")
    checksum_path = ROOT / "deployment/linux-cache/SHA256SUMS"
    checksum_path.write_text("".join(checksums), encoding="utf-8", newline="\n")
    files["deployment/linux-cache/SHA256SUMS"] = "deployment/linux-cache/SHA256SUMS"

    destination = ROOT / "dist/情绪识别.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    expected = {}
    with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, source in files.items():
            data = payload(source)
            member = tarfile.TarInfo(f"emotion/{name}")
            member.size = len(data)
            member.mode = 0o755 if name.endswith((".sh", ".desktop")) else 0o644
            member.mtime = int((ROOT / source).stat().st_mtime)
            archive.addfile(member, io.BytesIO(data))
            expected[member.name] = hashlib.sha256(data).hexdigest()

    # Verify content, startup targets, and permissions before replacing the package.
    with tarfile.open(temporary, "r:gz") as archive:
        assert set(archive.getnames()) == set(expected)
        for member in archive:
            assert member.isfile()
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected[member.name]
        for name in ("start.sh", "启动情绪识别.desktop"):
            assert archive.getmember(f"emotion/{name}").mode == 0o755
        start = archive.extractfile("emotion/start.sh").read()
        assert b'"$ROOT/requirements-runtime-linux.txt"' in start
        assert "emotion/requirements-runtime-linux.txt" in expected
        assert "emotion/deployment/linux-python.env" in expected
        assert b"--no-index --no-cache-dir" in start
        for asset in lock["assets"]:
            assert expected["emotion/deployment/linux-cache/" + asset["file"]] == asset["sha256"]

    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_name(destination.name + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8", newline="\n",
    )
    manifest = {
        "status": "built-and-verified", "package": str(destination),
        "sha256": digest, "bytes": destination.stat().st_size,
        "top_level_directory": "emotion", "files": expected,
        "offline_first_install": True,
        "bundled_python": lock["python_version"],
        "offline_asset_count": len(lock["assets"]),
        "linux_desktop_tested": False,
    }
    report = ROOT / "output/linux-package-manifest.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
