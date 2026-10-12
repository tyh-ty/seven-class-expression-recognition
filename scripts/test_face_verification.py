"""Regression cases for false nearest faces, low light and eyewear sensitivity."""
from pathlib import Path
import sys
from threading import Lock
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import CLASSIFIER_SPECS, EmotionRecognizer


def box(x, y, width, height, score):
    return [x, y, width, height, *([0.] * 10), score]


class FakeDetector:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.sizes = []
        self.calls = 0

    def setInputSize(self, size):
        self.sizes.append(size)

    def detect(self, image):
        if not self.outputs:
            raise AssertionError('Unexpected additional detection pass')
        self.calls += 1
        result = self.outputs.pop(0)
        return None, np.asarray(result, dtype=np.float32) if result else None


class VerificationTests(unittest.TestCase):
    def recognizer(self, *outputs):
        result = EmotionRecognizer.__new__(EmotionRecognizer)
        result.face_detector = FakeDetector(*outputs)
        result.inference_lock = Lock()
        return result

    def setUp(self):
        self.frame = np.zeros((384, 512, 3), dtype=np.uint8)
        self.strong = box(360, 30, 65, 70, .95)
        self.large_weak = box(80, 100, 120, 150, .78)

    def test_single_dim_or_eyeglass_face_keeps_low_threshold(self):
        recognizer = self.recognizer([box(150, 80, 100, 100, .51)])
        self.assertEqual(len(recognizer._detect_faces(self.frame)), 1)
        self.assertEqual(recognizer.face_detector.calls, 1)

    def test_all_low_confidence_faces_are_retained(self):
        recognizer = self.recognizer([box(20, 40, 80, 80, .7), box(240, 40, 80, 80, .6)])
        self.assertEqual(len(recognizer._detect_faces(self.frame)), 2)
        self.assertEqual(recognizer.face_detector.calls, 1)

    def test_large_body_false_positive_does_not_become_nearest(self):
        recognizer = self.recognizer([self.strong, self.large_weak], None)
        faces = recognizer._detect_faces(self.frame)
        self.assertEqual([face[:4] for face in faces], [tuple(self.strong[:4])])

    def test_near_eyeglass_face_passes_local_verification(self):
        recognizer = self.recognizer([self.strong, self.large_weak], [box(60, 50, 100, 120, .88)])
        faces = recognizer._detect_faces(self.frame)
        self.assertEqual(len(faces), 2)
        self.assertEqual(max(faces, key=lambda f: f[2] * f[3])[:4], tuple(self.large_weak[:4]))

    def test_incidental_face_at_crop_edge_does_not_validate_body(self):
        recognizer = self.recognizer([self.strong, self.large_weak], [box(0, 0, 55, 55, .95)])
        self.assertEqual(len(recognizer._detect_faces(self.frame)), 1)

    def test_distant_weak_face_remains_and_detector_size_resets(self):
        small = box(20, 20, 30, 35, .56)
        recognizer = self.recognizer([self.strong, self.large_weak, small], None, [self.strong])
        self.assertEqual(len(recognizer._detect_faces(self.frame)), 2)
        self.assertEqual(len(recognizer._detect_faces(self.frame)), 1)
        self.assertEqual(recognizer.face_detector.sizes, [(512, 384), (224, 224), (512, 384)])

    def test_feedback_uses_same_verified_nearest_face(self):
        recognizer = self.recognizer([self.strong, self.large_weak], None)
        feedback = recognizer.feedback_face(self.frame)
        self.assertIsNotNone(feedback)
        self.assertGreater(feedback['faceConfidence'], .9)
        self.assertEqual(feedback['image'].shape, (224, 224, 3))

    def test_analyze_classifies_verified_faces_and_keeps_seven_class_contract(self):
        class Session:
            def run(self, names, inputs):
                self.tensor = inputs['image']
                return [np.asarray([[0., 0, 0, 2, 0, 0, 0]], dtype=np.float32)]
        recognizer = self.recognizer([self.strong, self.large_weak], None)
        session = Session()
        recognizer.classifiers = {'transformer': {'spec': CLASSIFIER_SPECS[0],
            'session': session, 'input_name': 'image', 'output_name': 'scores'}}
        result = recognizer.analyze(self.frame)
        self.assertEqual(len(result['faces']), 1)
        self.assertEqual(result['faces'][0]['bbox']['x'], self.strong[0])
        self.assertEqual(len(result['faces'][0]['scores']), 7)
        self.assertEqual(result['faces'][0]['emotion']['id'], 'happiness')
        self.assertEqual(session.tensor.shape, (1, 3, 224, 224))


if __name__ == '__main__':
    unittest.main()
