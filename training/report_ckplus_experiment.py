from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "training/runs/vit-small-seven-ckplus-v1"
BASE = ROOT / "training/runs/vit-small-seven-glasses-v1"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(report: dict, kind: str) -> dict:
    if kind == "image":
        values = report["metrics"]
        return {
            "examples": report["examples"],
            "accuracy": values["accuracy"],
            "macro_f1": report.get("supported_metrics", {}).get(
                "macro_f1_supported", values["macro_f1"]
            ),
            "anger_recall": values["per_class"]["anger"]["recall"],
            "fear_recall": values["per_class"]["fear"]["recall"],
        }
    if kind == "video":
        values = report["video_level_mean_logits"]
        return {
            "examples": values["examples"],
            "accuracy": values["accuracy"],
            "macro_f1": values.get("macro_f1", values.get("macro_f1_supported")),
            "anger_recall": values["per_class"]["anger"]["recall"],
            "fear_recall": values["per_class"]["fear"]["recall"],
        }
    scenario = report["scenarios"]["glasses_updated"]
    rows = scenario["records"]
    recall = {
        label: sum(row["predicted"] == label for row in rows if row["label"] == label)
        / sum(row["label"] == label for row in rows)
        for label in ("anger", "fear")
    }
    return {
        "examples": scenario["examples"],
        "accuracy": scenario["accuracy_including_misses"],
        "anger_recall": recall["anger"],
        "fear_recall": recall["fear"],
    }


def main() -> None:
    baseline_files = {
        "FERPlus": (BASE / "ferplus_test_int8.json", "image"),
        "CK+": (BASE / "ckplus_test_int8.json", "image"),
        "KDEF": (BASE / "kdef_test_int8.json", "image"),
        "RAVDESS video": (BASE / "ravdess_complete_test.json", "video"),
        "CREMA-D video": (BASE / "cremad_test_breakdown.json", "video"),
        "Synthetic glasses product": (
            ROOT / "output/glasses-product-candidate-benchmark.json", "glasses"
        ),
    }
    baseline = {label: metrics(read(path), kind) for label, (path, kind) in baseline_files.items()}
    suffixes = {
        "FERPlus": ("ferplus", "image"),
        "CK+": ("ckplus", "image"),
        "KDEF": ("kdef", "image"),
        "RAVDESS video": ("ravdess", "video"),
        "CREMA-D video": ("cremad", "video"),
        "Synthetic glasses product": ("glasses-product", "glasses"),
    }
    candidates = {
        name: {
            label: metrics(read(RUN / f"{name}_{suffix}.json"), kind)
            for label, (suffix, kind) in suffixes.items()
        }
        for name in ("e1-a025", "e1-a05", "e5-a075", "e5-last2-full")
    }
    dataset = read(ROOT / "data/seven-class/ckplus-adaptation-v1/dataset.json")
    history = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    audit_path = RUN / "extended-duplicate-audit.json"
    audit = read(audit_path) if audit_path.exists() else dataset["duplicate_audit"]
    report = {
        "status": "training_completed_candidate_not_promoted",
        "source": dataset["source"],
        "license_scope": "CC0-1.0 is the mirror uploader's declaration; original CK+ authorization was not independently established. This is an experimental local research run, not a redistribution of CK+ images in the deployment ZIP.",
        "subject_split": dataset["split"],
        "duplicate_audit": audit,
        "training": {
            "architecture": "single seven-class pure ViT-S/16",
            "initial_checkpoint": "training/runs/vit-small-seven-glasses-v1/best_target.pt",
            "rounds": len(history),
            "max_train_batches_per_round": 500,
            "batch_size": 32,
            "training_images_in_folder": 44810,
            "backbone_learning_rate": 3e-6,
            "head_learning_rate": 2e-5,
            "trainable_last_transformer_blocks": 4,
            "distillation_weight": 0.20,
            "label_smoothing": 0.06,
            "extra_source_class_weighting": "none",
            "count_metadata_note": "The training-time extra-source counter omitted the ckplus_ prefix; the sample-level loss correctly recognized it. With weighting=none this does not change the loss. The shared-prefix counter has been fixed for future runs.",
        },
        "baseline": baseline,
        "candidates": candidates,
        "decision": {
            "promoted": False,
            "most_targeted_improvement": "e5-a075",
            "reason": "CK+ emphasizes posed peak expressions, while the requested product target is subtle expression. The candidate improves CK+ and synthetic eyewear but regresses natural-video preservation metrics, so it was withdrawn.",
            "formal_int8_sha256": hashlib.sha256(
                (ROOT / "models/vit_emotion_seven_int8.onnx").read_bytes()
            ).hexdigest(),
            "deployment_package_rebuilt": False,
        },
        "evaluation_limitations": [
            "These evaluation sets were inspected repeatedly for candidate exploration and are not untouched final holdouts for this run.",
            "CK+ contains adjacent sequence frames. Its 138 test frames correspond to 46 expression sequences and 16 subjects; frames are not independent examples.",
            "CK+ has only 12 fear test frames from four expression sequences and no neutral images; it is not a seven-class webcam-accuracy measurement.",
            "Synthetic eyewear is not a substitute for a fresh recording session with real glasses on the laboratory camera.",
            "CC0 is the mirror repository's declared license, not independent proof of the original dataset owner's redistribution terms.",
        ],
    }
    (RUN / "experiment-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    targeted = candidates["e5-a075"]
    lines = [
        "# CK+ 新数据微调实验结果",
        "",
        "已完成 6 轮微调（每轮最多 500 批、16,000 张随机样本），并比较每轮检查点、权重融合及部分层组合。CK+ 偏向刻意峰值表情，不符合细微表情目标，候选已撤回。",
        "",
        "新增 CK+：927 张六类可用帧；74 人训练、16 人验证、16 人测试。轻蔑排除，中性沿用原有数据。",
        "",
        "下表为改进新数据和合成眼镜最明显的 e5-a075 INT8 候选与当前正式 INT8 模型的对比。",
        "",
        "| 数据/场景 | 当前准确率 | 候选准确率 | 当前宏 F1 | 候选宏 F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, old in baseline.items():
        new = targeted[label]
        old_f1 = f"{old['macro_f1']:.4f}" if "macro_f1" in old else "—"
        new_f1 = f"{new['macro_f1']:.4f}" if "macro_f1" in new else "—"
        lines.append(f"| {label} | {old['accuracy']:.2%} | {new['accuracy']:.2%} | {old_f1} | {new_f1} |")
    lines += [
        "",
        "| 数据/场景 | 当前愤怒召回 | 候选愤怒召回 | 当前恐惧召回 | 候选恐惧召回 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, old in baseline.items():
        new = targeted[label]
        lines.append(f"| {label} | {old['anger_recall']:.2%} | {new['anger_recall']:.2%} | {old['fear_recall']:.2%} | {new['fear_recall']:.2%} |")
    lines += [
        "",
        "结论：CK+ 候选已撤回。正式系统恢复自然视频表现较好的上一模型，后续训练改用低强度、连续帧数据。",
        "",
        "评估限制：本轮反复查看了这些评测集，属于探索性比较，不能称为未触碰的最终测试。CK+ 测试只有 16 人、46 段表情序列、138 张相邻帧，恐惧只有 4 段序列/12 张帧，且不包含中性。戴眼镜指标来自合成眼镜，并非实验室真实摄像头。",
        "",
        "数据授权：镜像仓库标注 CC0-1.0，原 CK+ 的授权未独立核实。原始数据只保留在本地实验目录，没有加入部署 ZIP。",
        "",
        f"正式 INT8 SHA256：`{report['decision']['formal_int8_sha256']}`。",
        "",
        "全部候选的逐类成绩、训练参数、去重记录见 experiment-report.json；训练记录见 history.json。",
    ]
    (RUN / "experiment-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "formal_sha256": report["decision"]["formal_int8_sha256"], "baseline": baseline, "targeted_candidate": targeted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
