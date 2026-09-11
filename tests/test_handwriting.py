import unittest

import cv2
import numpy as np

from numai.camera import Reading, StableTracks, draw_preview
from numai.handwriting import STRESS_VARIANTS, stress_images
from numai.vision import augment, normalize_digit


class HandwritingTests(unittest.TestCase):
    def test_augmentation_is_seeded_bounded_and_keeps_ink(self):
        ink = np.zeros((60, 60), np.uint8)
        cv2.ellipse(ink, (30, 30), (12, 22), 0, 0, 360, 255, 3)
        original = np.repeat(normalize_digit(ink)[None], 100, axis=0)
        before = original.copy()
        augmented = augment(original, np.random.default_rng(42))
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(augmented, augment(original, np.random.default_rng(42)))
        self.assertTrue(np.isfinite(augmented).all())
        self.assertGreaterEqual(augmented.min(), 0)
        self.assertLessEqual(augmented.max(), 1)
        self.assertTrue((augmented.sum(axis=(1, 2)) > 10).all())
        self.assertGreater(np.max(augmented.sum(axis=(1, 2))), original[0].sum()*1.4)
        self.assertGreater(np.count_nonzero(np.any(augmented != original, axis=(1, 2))), 60)
        stressed = stress_images(original)
        self.assertEqual(stressed.shape, original.shape)
        self.assertEqual(len(STRESS_VARIANTS), 6)
        self.assertGreater(stressed[0].sum(), original[0].sum())
        self.assertGreater(stressed[4].sum(), original[0].sum() * 0.8)
        self.assertGreater(stressed[5].sum(), 10)


class TrackingTests(unittest.TestCase):
    def reading(self, x, number=7, y=50):
        return Reading(np.zeros((28, 28), np.float32), (x, y, 80), number, 0.99)

    def test_same_digit_in_two_places_is_reported_independently(self):
        tracks = StableTracks(frames=3)
        for _ in range(2):
            self.assertEqual(tracks.update([self.reading(30), self.reading(400)])[0], [])
        events, _ = tracks.update([self.reading(34), self.reading(396)])
        self.assertEqual(len(events), 2)
        self.assertNotEqual(events[0][0], events[1][0])
        self.assertEqual(tracks.update([self.reading(35), self.reading(397)])[0], [])
        for _ in range(3):
            events, removed = tracks.update([])
        self.assertTrue(removed)
        self.assertEqual(tracks.tracks, {})
        for _ in range(3):
            events, _ = tracks.update([self.reading(35)])
        self.assertEqual(len(events), 1)

    def test_uncertain_neighbor_does_not_block_and_gap_resets_agreement(self):
        tracks = StableTracks(frames=3)
        tracks.update([self.reading(30)])
        tracks.update([])
        for _ in range(2):
            self.assertEqual(tracks.update([self.reading(30), self.reading(400, None)])[0], [])
        events, _ = tracks.update([self.reading(30), self.reading(400, None)])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1].number, 7)

    def test_preview_does_not_cover_corner_digit_or_mutate_input(self):
        frame = np.full((480, 640, 3), 255, np.uint8)
        cv2.line(frame, (30, 365), (30, 450), (0, 0, 0), 9)
        original = frame.copy()
        shown = draw_preview(frame, [self.reading(10, y=370)])
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(shown[390:440, 25:36], original[390:440, 25:36])
        self.assertEqual(shown.shape, (480, 790, 3))


if __name__ == '__main__':
    unittest.main()
