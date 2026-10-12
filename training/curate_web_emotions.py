from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps

from emotion_preprocessing import emotion_tensor_path


CLASSES = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)
LABEL_ALIASES = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happiness",
    "happiness": "happiness",
    "neutral": "neutral",
    "neutrality": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass
class Sample:
    path: Path
    relative_path: str
    source_label: str
    final_label: str
    sha256: str
    dhash: int
    width: int
    height: int
    sharpness: float
    model_label: str = ""
    model_confidence: float = 0.0
    source_probability: float = 0.0
    status: str = "pending"
    reasons: list[str] = field(default_factory=list)
    output_path: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and pre-label public facial-expression images"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--decisions",
        type=Path,
        help="CSV with id, decision (accept/relabel/reject), and optional new_label.",
    )
    parser.add_argument(
        "--require-complete-decisions",
        action="store_true",
        help="Fail if any review sample has no manual decision.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    parser.add_argument("--review-confidence", type=float, default=0.45)
    parser.add_argument("--minimum-short-edge", type=int, default=40)
    parser.add_argument("--minimum-sharpness", type=float, default=18.0)
    parser.add_argument("--contact-sheet-columns", type=int, default=5)
    parser.add_argument("--contact-sheet-rows", type=int, default=5)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    gray = ImageOps.exif_transpose(image).convert("L").resize(
        (9, 8), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for index, bit in enumerate(bits.flat):
        if bit:
            value |= 1 << index
    return value


def image_sharpness(image: Image.Image) -> float:
    gray = np.asarray(ImageOps.exif_transpose(image).convert("L"), dtype=np.uint8)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def discover_samples(root: Path) -> tuple[list[Sample], list[dict[str, str]]]:
    samples: list[Sample] = []
    invalid: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        raw_label = relative.parts[0].lower()
        label = LABEL_ALIASES.get(raw_label)
        if label is None:
            invalid.append(
                {
                    "path": relative.as_posix(),
                    "reason": f"unknown_label:{raw_label}",
                }
            )
            continue
        try:
            with Image.open(path) as image:
                image.load()
                samples.append(
                    Sample(
                        path=path,
                        relative_path=relative.as_posix(),
                        source_label=label,
                        final_label=label,
                        sha256=file_sha256(path),
                        dhash=difference_hash(image),
                        width=image.width,
                        height=image.height,
                        sharpness=image_sharpness(image),
                    )
                )
        except (OSError, ValueError) as error:
            invalid.append(
                {
                    "path": relative.as_posix(),
                    "reason": f"unreadable:{type(error).__name__}",
                }
            )
    return samples, invalid


def reject_exact_duplicates(samples: list[Sample]) -> None:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.sha256].append(sample)
    for group in groups.values():
        if len(group) < 2:
            continue
        labels = {sample.source_label for sample in group}
        if len(labels) > 1:
            for sample in group:
                sample.status = "rejected"
                sample.reasons.append("cross_label_exact_duplicate")
            continue
        keeper = max(group, key=lambda sample: (sample.width * sample.height, sample.sharpness))
        for sample in group:
            if sample is not keeper:
                sample.status = "rejected"
                sample.reasons.append("within_label_exact_duplicate")


def reject_near_duplicates(samples: list[Sample], max_distance: int) -> None:
    candidates = [sample for sample in samples if sample.status == "pending"]
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if (candidates[left].dhash ^ candidates[right].dhash).bit_count() <= max_distance:
                union(left, right)

    groups: dict[int, list[Sample]] = defaultdict(list)
    for index, sample in enumerate(candidates):
        groups[find(index)].append(sample)
    for group in groups.values():
        if len(group) < 2:
            continue
        labels = {sample.source_label for sample in group}
        if len(labels) > 1:
            for sample in group:
                sample.status = "rejected"
                sample.reasons.append("cross_label_near_duplicate")
            continue
        keeper = max(group, key=lambda sample: (sample.width * sample.height, sample.sharpness))
        for sample in group:
            if sample is not keeper:
                sample.status = "rejected"
                sample.reasons.append("within_label_near_duplicate")


def model_input(path: Path, image_size: int = 224) -> np.ndarray:
    return emotion_tensor_path(path, image_size=image_size)


def predict(samples: list[Sample], model: Path, batch_size: int, threads: int) -> None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch = np.stack([model_input(sample.path) for sample in batch_samples])
        logits = np.asarray(session.run(None, {input_name: batch})[0], dtype=np.float32)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for sample, row in zip(batch_samples, probabilities, strict=True):
            prediction = int(row.argmax())
            sample.model_label = CLASSES[prediction]
            sample.model_confidence = float(row[prediction])
            sample.source_probability = float(row[CLASSES.index(sample.source_label)])


def assign_statuses(
    samples: list[Sample],
    review_confidence: float,
    minimum_short_edge: int,
    minimum_sharpness: float,
) -> None:
    for sample in samples:
        if sample.status == "rejected":
            continue
        if min(sample.width, sample.height) < minimum_short_edge:
            sample.reasons.append("low_resolution")
        if sample.sharpness < minimum_sharpness:
            sample.reasons.append("low_sharpness")
        if (
            sample.model_label != sample.source_label
            and sample.model_confidence >= review_confidence
        ):
            sample.status = "review"
            sample.reasons.append("model_label_disagreement")
        elif "low_resolution" in sample.reasons or "low_sharpness" in sample.reasons:
            sample.status = "review"
        else:
            sample.status = "accepted"


def apply_decisions(
    samples: list[Sample], decisions_path: Path, require_complete: bool
) -> Counter[str]:
    with decisions_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    decisions: dict[int, dict[str, str]] = {}
    for row in rows:
        raw_id = (row.get("id") or "").strip()
        decision = (row.get("decision") or "").strip().lower()
        if not raw_id or not decision:
            continue
        sample_id = int(raw_id)
        if not 0 <= sample_id < len(samples):
            raise ValueError(f"Manual decision id is out of range: {sample_id}")
        if sample_id in decisions:
            raise ValueError(f"Duplicate manual decision for sample id {sample_id}")
        decisions[sample_id] = row

    counts: Counter[str] = Counter()
    for sample_id, sample in enumerate(samples):
        row = decisions.get(sample_id)
        if row is None:
            continue
        expected_path = (row.get("source_path") or "").strip()
        if expected_path and expected_path != sample.relative_path:
            raise ValueError(
                f"Decision path mismatch for id {sample_id}: "
                f"expected {sample.relative_path}, got {expected_path}"
            )
        if sample.status != "review":
            raise ValueError(
                f"Manual decision targets non-review sample {sample_id}: {sample.status}"
            )
        decision = (row.get("decision") or "").strip().lower()
        if decision == "accept":
            sample.status = "accepted"
            sample.final_label = sample.source_label
            sample.reasons.append("manual_accept")
        elif decision == "relabel":
            new_label = LABEL_ALIASES.get((row.get("new_label") or "").strip().lower())
            if new_label not in CLASSES:
                raise ValueError(
                    f"Invalid new_label for sample {sample_id}: {row.get('new_label')!r}"
                )
            sample.status = "accepted"
            sample.final_label = new_label
            sample.reasons.append("manual_relabel")
        elif decision == "reject":
            sample.status = "rejected"
            sample.reasons.append("manual_reject")
        else:
            raise ValueError(f"Unknown decision for sample {sample_id}: {decision!r}")
        counts[decision] += 1

    undecided = [
        sample.relative_path for sample in samples if sample.status == "review"
    ]
    if require_complete and undecided:
        preview = ", ".join(undecided[:5])
        raise ValueError(
            f"Missing manual decisions for {len(undecided)} review samples: {preview}"
        )
    counts["remaining_review"] = len(undecided)
    return counts


def copy_samples(samples: list[Sample], output: Path) -> None:
    for index, sample in enumerate(samples):
        if sample.status == "accepted":
            destination_root = output / "train" / sample.final_label
        elif sample.status == "review":
            destination_root = output / "review" / sample.source_label
        else:
            destination_root = output / "rejected" / sample.source_label
        destination_root.mkdir(parents=True, exist_ok=True)
        name = f"web_{index:04d}_{sample.sha256[:10]}{sample.path.suffix.lower()}"
        destination = destination_root / name
        shutil.copy2(sample.path, destination)
        sample.output_path = destination.relative_to(output).as_posix()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_contact_sheets(
    samples: list[Sample], output: Path, columns: int, rows: int
) -> list[str]:
    sheet_paths: list[str] = []
    cell_width = 220
    cell_height = 240
    margin = 20
    header_height = 52
    items_per_sheet = columns * rows
    title_font = load_font(24)
    label_font = load_font(15)
    detail_font = load_font(12)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        group_label = sample.final_label if sample.status == "accepted" else sample.source_label
        groups[f"{sample.status}_{group_label}"].append(sample)

    sheet_root = output / "contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    for group_name, group in sorted(groups.items()):
        for page_index in range(math.ceil(len(group) / items_per_sheet)):
            page_samples = group[
                page_index * items_per_sheet : (page_index + 1) * items_per_sheet
            ]
            canvas = Image.new(
                "RGB",
                (
                    margin * 2 + columns * cell_width,
                    margin * 2 + header_height + rows * cell_height,
                ),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (margin, margin),
                f"{group_name}  page {page_index + 1}",
                fill="black",
                font=title_font,
            )
            for item_index, sample in enumerate(page_samples):
                row = item_index // columns
                column = item_index % columns
                x = margin + column * cell_width
                y = margin + header_height + row * cell_height
                with Image.open(sample.path) as image:
                    preview = ImageOps.exif_transpose(image).convert("RGB")
                    preview.thumbnail((190, 170), Image.Resampling.LANCZOS)
                preview_x = x + (cell_width - preview.width) // 2
                canvas.paste(preview, (preview_x, y))
                draw.text(
                    (x + 8, y + 176),
                    f"#{samples.index(sample):04d} src={sample.source_label} final={sample.final_label}",
                    fill="black",
                    font=label_font,
                )
                reason = ",".join(sample.reasons) or "clean"
                draw.text(
                    (x + 8, y + 198),
                    f"model={sample.model_label} {sample.model_confidence:.2f} {reason[:18]}",
                    fill=(110, 20, 20) if sample.status != "accepted" else (30, 80, 35),
                    font=detail_font,
                )
            sheet_path = sheet_root / f"{group_name}_{page_index + 1:02d}.jpg"
            canvas.save(sheet_path, quality=92)
            sheet_paths.append(sheet_path.relative_to(output).as_posix())
    return sheet_paths


def write_manifest(samples: list[Sample], output: Path) -> None:
    fields = (
        "id",
        "source_path",
        "source_label",
        "final_label",
        "model_label",
        "model_confidence",
        "source_probability",
        "status",
        "reasons",
        "width",
        "height",
        "sharpness",
        "sha256",
        "dhash",
        "output_path",
    )
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples):
            writer.writerow(
                {
                    "id": index,
                    "source_path": sample.relative_path,
                    "source_label": sample.source_label,
                    "final_label": sample.final_label,
                    "model_label": sample.model_label,
                    "model_confidence": f"{sample.model_confidence:.6f}",
                    "source_probability": f"{sample.source_probability:.6f}",
                    "status": sample.status,
                    "reasons": ";".join(sample.reasons),
                    "width": sample.width,
                    "height": sample.height,
                    "sharpness": f"{sample.sharpness:.3f}",
                    "sha256": sample.sha256,
                    "dhash": f"{sample.dhash:016x}",
                    "output_path": sample.output_path,
                }
            )
    with (output / "review_decisions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("id", "source_path", "current_label", "decision", "new_label"),
        )
        writer.writeheader()
        for index, sample in enumerate(samples):
            if sample.status == "review":
                writer.writerow(
                    {
                        "id": index,
                        "source_path": sample.relative_path,
                        "current_label": sample.source_label,
                        "decision": "",
                        "new_label": "",
                    }
                )


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(
            f"Output is not empty: {args.output}. Use a new output path to avoid data loss."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f"{args.output.name}-building-", dir=args.output.parent)
    )
    try:
        samples, invalid = discover_samples(args.input)
        if not samples:
            raise ValueError(f"No supported images found under {args.input}")
        reject_exact_duplicates(samples)
        reject_near_duplicates(samples, args.near_duplicate_distance)
        predict(samples, args.model, args.batch_size, args.threads)
        assign_statuses(
            samples,
            args.review_confidence,
            args.minimum_short_edge,
            args.minimum_sharpness,
        )
        manual_decisions: Counter[str] = Counter()
        if args.decisions:
            manual_decisions = apply_decisions(
                samples, args.decisions, args.require_complete_decisions
            )
        copy_samples(samples, temporary)
        sheet_paths = make_contact_sheets(
            samples,
            temporary,
            args.contact_sheet_columns,
            args.contact_sheet_rows,
        )
        write_manifest(samples, temporary)

        status_counts = Counter(sample.status for sample in samples)
        class_counts: dict[str, dict[str, int]] = {}
        for label in CLASSES:
            class_counts[label] = {
                status: sum(
                    sample.final_label == label and sample.status == status
                    for sample in samples
                )
                for status in ("accepted", "review", "rejected")
            }
        reason_counts = Counter(
            reason for sample in samples for reason in sample.reasons
        )
        agreement = sum(
            sample.model_label == sample.source_label
            for sample in samples
            if sample.status != "rejected"
        )
        final_agreement = sum(
            sample.model_label == sample.final_label
            for sample in samples
            if sample.status == "accepted"
        )
        evaluated = sum(sample.status != "rejected" for sample in samples)
        accepted = sum(sample.status == "accepted" for sample in samples)
        report = {
            "dataset": "Human-Face_Images_for_Emotion_Recognition",
            "source": "Prasanna18/Human-Face_Images_for_Emotion_Recognition",
            "declared_license": "MIT",
            "input": str(args.input),
            "model": str(args.model),
            "total_images": len(samples),
            "invalid_images": invalid,
            "status_counts": dict(status_counts),
            "class_counts": class_counts,
            "reason_counts": dict(reason_counts),
            "manual_decisions": dict(manual_decisions),
            "model_source_label_agreement": round(agreement / max(evaluated, 1), 6),
            "model_final_label_agreement": round(
                final_agreement / max(accepted, 1), 6
            ),
            "settings": {
                "near_duplicate_distance": args.near_duplicate_distance,
                "review_confidence": args.review_confidence,
                "minimum_short_edge": args.minimum_short_edge,
                "minimum_sharpness": args.minimum_sharpness,
                "grayscale_model_input": True,
            },
            "contact_sheets": sheet_paths,
            "training_policy": "Only train/ is eligible for training; review/ and rejected/ are excluded.",
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        source_readme = args.input.parent / "README.md"
        if source_readme.is_file():
            shutil.copy2(source_readme, temporary / "SOURCE_README.md")
        if args.output.exists():
            args.output.rmdir()
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
