from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import prepare_feedback_data


class PrepareFeedbackDataTest(unittest.TestCase):
    def save_pattern(self, path: Path, seed: int) -> None:
        randomizer = random.Random(seed)
        pixels = bytes(randomizer.randrange(256) for _ in range(9 * 8))
        image = Image.frombytes("L", (9, 8), pixels).resize((224, 224))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    def test_prepares_session_isolated_data_and_ignores_smoke_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "feedback"
            output = root / "prepared"
            for label_index, label in enumerate(("happiness", "neutral")):
                for index in range(12):
                    self.save_pattern(
                        source / "natural_day1" / label / f"sample_{index:02d}.png",
                        label_index * 1000 + index,
                    )
                for index in range(8):
                    self.save_pattern(
                        source / "natural_day2" / label / f"sample_{index:02d}.png",
                        label_index * 1000 + 100 + index,
                    )
            shutil.copy2(
                source / "natural_day1" / "happiness" / "sample_00.png",
                source / "natural_day1" / "happiness" / "duplicate.png",
            )
            self.save_pattern(
                source / "_smoke" / "happiness" / "ignored.png", 9999
            )

            arguments = [
                "prepare_feedback_data.py",
                "--input",
                str(source),
                "--output",
                str(output),
                "--hash-distance",
                "0",
            ]
            with patch.object(sys, "argv", arguments):
                prepare_feedback_data.main()

            report = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(report["discovery"]["ignored_sessions"], ["_smoke"])
            self.assertEqual(len(report["discovery"]["near_duplicates"]), 1)
            self.assertEqual(report["classes"]["happiness"]["total"], 20)
            self.assertEqual(report["classes"]["neutral"]["total"], 20)
            self.assertTrue(report["classes"]["happiness"]["eligible"])
            self.assertTrue(report["classes"]["neutral"]["eligible"])
            self.assertEqual(
                set(report["train_sessions"]) & set(report["validation_sessions"]),
                set(),
            )

    def test_rejects_a_single_natural_session(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two separate sessions"):
            prepare_feedback_data.split_sessions(["natural_day1"], 0.25, 42)


if __name__ == "__main__":
    unittest.main()
