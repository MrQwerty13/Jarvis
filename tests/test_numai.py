import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from numai.model import MLP
from numai.vision import extract_digit, normalize_digit
from numai.camera import StablePrediction, center_square


class ModelTests(unittest.TestCase):
    def test_backprop_matches_numerical_derivative(self):
        model = MLP((4, 5, 3), seed=12)
        # Double precision avoids cancellation in this independent gradient check.
        model.params = {k: v.astype(np.float64) for k, v in model.params.items()}
        x = np.random.default_rng(20).normal(size=(6, 4))
        y = np.array([0, 1, 2, 2, 1, 0])
        _, gradients = model.loss_and_gradients(x, y)
        eps = 1e-5
        for name, param in model.params.items():
            for index in np.ndindex(param.shape):
                original = param[index]
                param[index] = original + eps
                plus = model.loss_and_gradients(x, y)[0]
                param[index] = original - eps
                minus = model.loss_and_gradients(x, y)[0]
                param[index] = original
                self.assertAlmostEqual(gradients[name][index], (plus-minus)/(2*eps), places=6)

    def test_network_learns_and_checkpoint_roundtrips(self):
        model = MLP((2, 12, 2), seed=7)
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0])
        before = model.loss_and_gradients(x, y)[0]
        for _ in range(200):
            model.train_batch(x, y, lr=0.03)
        self.assertLess(model.loss_and_gradients(x, y)[0], before * 0.1)
        np.testing.assert_array_equal(model.predict_proba(x).argmax(1), y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.npz'
            model.save(path)
            restored = MLP.load(path)
            np.testing.assert_allclose(restored.predict_proba(x), model.predict_proba(x))


class VisionTests(unittest.TestCase):
    def test_blank_and_low_contrast_are_rejected(self):
        for value in (0, 127, 255):
            self.assertIsNone(extract_digit(np.full((240, 240), value, np.uint8)))
        faint = np.full((240, 240), 240, np.uint8)
        cv2.line(faint, (120, 60), (120, 180), 234, 5)
        self.assertIsNone(extract_digit(faint))

    def test_single_stroke_is_centered_and_scaled(self):
        card = np.full((240, 240), 255, np.uint8)
        cv2.line(card, (90, 60), (90, 180), 0, 9)
        digit = extract_digit(card)
        self.assertIsNotNone(digit)
        self.assertEqual(digit.shape, (28, 28))
        self.assertGreater(digit.max(), 0.9)
        ys, xs = np.indices(digit.shape)
        self.assertAlmostEqual(float((xs*digit).sum()/digit.sum()), 13.5, delta=0.6)
        self.assertAlmostEqual(float((ys*digit).sum()/digit.sum()), 13.5, delta=0.6)

    def test_two_digits_and_border_are_rejected(self):
        card = np.full((240, 240), 255, np.uint8)
        cv2.line(card, (60, 60), (60, 180), 0, 9)
        cv2.line(card, (160, 60), (160, 180), 0, 9)
        self.assertIsNone(extract_digit(card))
        card[:] = 255
        cv2.rectangle(card, (0, 0), (239, 239), 0, 9)
        self.assertIsNone(extract_digit(card))

    def test_normalization_preserves_hole(self):
        image = np.zeros((60, 60), np.uint8)
        cv2.ellipse(image, (30, 30), (15, 23), 0, 0, 360, 255, 4)
        digit = normalize_digit(image)
        self.assertLess(digit[14, 14], 0.1)
        self.assertGreater(digit.sum(), 25)


class StabilityTests(unittest.TestCase):
    def test_camera_region_is_centered_square(self):
        self.assertEqual(center_square((480, 640)), (224, 144, 192, 192))

    def test_requires_consecutive_agreement_and_rearms_after_blank(self):
        stable = StablePrediction(frames=3)
        self.assertIsNone(stable.update(4))
        self.assertIsNone(stable.update(2))
        self.assertIsNone(stable.update(4))
        self.assertIsNone(stable.update(4))
        self.assertEqual(stable.update(4), 4)
        self.assertIsNone(stable.update(4))
        for _ in range(3):
            stable.update(None)
        self.assertIsNone(stable.update(4))
        self.assertIsNone(stable.update(4))
        self.assertEqual(stable.update(4), 4)


if __name__ == '__main__':
    unittest.main()
