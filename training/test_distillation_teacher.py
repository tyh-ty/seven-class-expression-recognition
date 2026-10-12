"""Input-contract checks for cross-architecture distillation (no GPU required)."""
import unittest
import cv2
import numpy as np
import torch
from torch import nn
from train_transformer import (DistillationTeacher, IMAGENET_MEAN, IMAGENET_STD,
                               run_epoch, SampleWeightedLabelSmoothingLoss)


class TeacherInputTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.tensor = torch.from_numpy(self.rgb.transpose(2, 0, 1).copy()).float()[None] / 255
        self.normalized = (self.tensor - self.mean) / self.std

    def test_rgb_identity_normalization(self):
        teacher = DistillationTeacher(nn.Identity(), False, IMAGENET_MEAN, IMAGENET_STD)
        torch.testing.assert_close(teacher(self.normalized), self.normalized, atol=3e-7, rtol=1e-6)

    def test_grayscale_matches_camera_up_to_byte_rounding(self):
        teacher = DistillationTeacher(nn.Identity(), True, IMAGENET_MEAN, IMAGENET_STD)
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        expected = torch.from_numpy(gray.copy()).float()[None, None] / 255
        expected = (expected.expand(-1, 3, -1, -1) - self.mean) / self.std
        self.assertLess(float((teacher(self.normalized) - expected).abs().max()), .51 / (255 * min(IMAGENET_STD)))

    def test_teacher_specific_normalization(self):
        teacher = DistillationTeacher(nn.Identity(), False, (.5, .5, .5), (.5, .5, .5))
        torch.testing.assert_close(teacher(self.normalized), self.tensor * 2 - 1, atol=2e-7, rtol=1e-6)

    def test_training_distills_only_base_source_with_different_architectures(self):
        class RecordingTeacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 7))
                self.seen = 0
            def forward(self, inputs):
                self.seen += len(inputs)
                return self.network(inputs)
        torch.manual_seed(7)
        student = nn.Sequential(nn.Conv2d(3, 5, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(5, 7))
        teacher_net = RecordingTeacher()
        teacher = DistillationTeacher(teacher_net, True, IMAGENET_MEAN, IMAGENET_STD).eval()
        teacher.requires_grad_(False)
        images = self.normalized.repeat(2, 1, 1, 1)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            images, torch.tensor([0, 1]), torch.tensor([False, True])), batch_size=2)
        criterion = SampleWeightedLabelSmoothingLoss(torch.ones(7), .1)
        before = [p.clone() for p in student.parameters()]
        metrics, _, steps = run_epoch(student, loader, criterion, torch.device('cpu'),
            optimizer=torch.optim.AdamW(student.parameters(), lr=.001),
            scaler=torch.amp.GradScaler('cuda', enabled=False), teacher_model=teacher,
            distillation_weight=1.)
        self.assertEqual(teacher_net.seen, 1)
        self.assertEqual(steps, 1)
        self.assertTrue(np.isfinite(metrics.loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, student.parameters())))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_correct_extra_distillation_excludes_wrong_teacher_targets(self):
        class LogitModel(nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = nn.Parameter(logits.clone())
            def forward(self, inputs):
                return self.logits[inputs[:, 0].long()]
        student_logits = torch.tensor([[1., 0, 0, 0, 0, 0, 0],
                                       [0., 1, 0, 0, 0, 0, 0],
                                       [0., 0, 1, 0, 0, 0, 0]])
        teacher_logits = torch.tensor([[0., 0, 0, 4, 0, 0, 0],
                                       [0., 5, 0, 0, 0, 0, 0],
                                       [0., 0, 0, 0, 6, 0, 0]])
        student, teacher = LogitModel(student_logits), LogitModel(teacher_logits)
        teacher.requires_grad_(False)
        images, targets = torch.arange(3).float()[:, None], torch.tensor([0, 1, 2])
        sources = torch.tensor([False, True, True])
        criterion = SampleWeightedLabelSmoothingLoss(torch.ones(7), .1)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, targets, sources), batch_size=3)
        expected = criterion(student_logits, targets, sources)
        selected = torch.tensor([True, True, False])
        expected += torch.nn.functional.kl_div(
            torch.log_softmax(student_logits[selected] / 2, dim=1),
            torch.softmax(teacher_logits[selected] / 2, dim=1), reduction='batchmean') * 4
        metrics, _, _ = run_epoch(student, loader, criterion, torch.device('cpu'),
            optimizer=torch.optim.SGD(student.parameters(), lr=.01),
            scaler=torch.amp.GradScaler('cuda', enabled=False), teacher_model=teacher,
            distillation_weight=1., distillation_scope='base-and-correct-extra')
        self.assertAlmostEqual(metrics.loss, float(expected), places=5)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_correct_extra_empty_mask_is_finite(self):
        student = nn.Linear(3, 7)
        teacher = nn.Linear(3, 7)
        teacher.requires_grad_(False)
        images = torch.randn(2, 3)
        targets = (teacher(images).argmax(dim=1) + 1) % 7
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            images, targets, torch.ones(2, dtype=torch.bool)), batch_size=2)
        criterion = SampleWeightedLabelSmoothingLoss(torch.ones(7), .1)
        expected = float(criterion(student(images), targets, torch.ones(2, dtype=torch.bool)).detach())
        metrics, _, _ = run_epoch(student, loader, criterion, torch.device('cpu'),
            optimizer=torch.optim.SGD(student.parameters(), lr=.01),
            scaler=torch.amp.GradScaler('cuda', enabled=False), teacher_model=teacher,
            distillation_weight=1., distillation_scope='base-and-correct-extra')
        self.assertAlmostEqual(metrics.loss, expected, places=5)


if __name__ == '__main__':
    unittest.main()
