from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "emotion-dataset/v2"
CLASSES = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)
SPLITS = ("train", "val", "test")
TARGET_VALIDATION_DIR = "target_val"
TARGET_VALIDATION_ALIASES = ("target_val", "target-val")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
EXTRA_SOURCE_PREFIXES = (
    "web_",
    "ravdess_",
    "cremad_",
    "kdef_",
    "ckplus_",
    "rafdb_",
    "glasses_",
)
STORAGE_MODES = ("auto", "copy", "hardlink")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_class_layout(path: Path, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    found = tuple(sorted(child.name for child in path.iterdir() if child.is_dir()))
    if found != CLASSES:
        if "contempt" in found:
            raise ValueError(
                f"Expected the current seven classes {CLASSES}, found legacy eight-class "
                f"layout including contempt in {path}. Generate a fresh seven-class "
                "FERPlus directory instead of reusing the legacy output."
            )
        raise ValueError(
            f"Expected class folders {CLASSES}, found {found} in {path}. "
            "Check that this is a seven-class ImageFolder split."
        )
    if not allow_empty:
        empty = [label for label in CLASSES if not image_files(path / label)]
        if empty:
            raise ValueError(f"Dataset split contains empty classes in {path}: {empty}")
    return found


def resolve_target_validation_path(path: Path) -> tuple[Path, str]:
    candidates: list[Path] = []
    if path.exists():
        candidates.append(path)
    if path.name in TARGET_VALIDATION_ALIASES:
        for name in TARGET_VALIDATION_ALIASES:
            candidate = path.parent / name
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_dir():
            canonical_name = TARGET_VALIDATION_DIR if candidate.name in TARGET_VALIDATION_ALIASES else candidate.name
            return candidate, canonical_name
    aliases = ", ".join(TARGET_VALIDATION_ALIASES)
    raise FileNotFoundError(
        f"Target validation directory not found: {path}. Supported names are {aliases}."
    )


def link_or_copy(source: Path, destination: Path, mode: str = "auto") -> str:
    if mode not in STORAGE_MODES:
        raise ValueError(f"Unknown storage mode {mode!r}; choose one of {STORAGE_MODES}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(source, destination)
        return "copy"


def storage_warning(methods: dict[str, int]) -> str | None:
    if methods.get("hardlink", 0):
        return (
            "This dataset contains hard links. Editing one linked file can change the "
            "same underlying content in another dataset; use --storage-mode copy for "
            "an independent snapshot."
        )
    return None


def manifest_rows(roots: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, root in roots.items():
        if not root.is_dir():
            continue
        resolved_root = root.resolve()
        for path in image_files(root):
            rows.append(
                {
                    "role": role,
                    "root": str(resolved_root),
                    "path": str(path.resolve().relative_to(resolved_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def write_manifest(path: Path, roots: dict[str, Path]) -> dict[str, object]:
    rows = manifest_rows(roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows]
    encoded = "".join(lines).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": str(path),
        "sha256": sha256_bytes(encoded),
        "files": len(rows),
        "roles": sorted(roots),
    }


def content_hash_index(roots: Iterable[Path]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for root in roots:
        for path in image_files(root):
            digest = sha256_file(path)
            index.setdefault(digest, []).append(str(path.resolve()))
    return index
