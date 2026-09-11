"""Automatic localization must work beyond the old center-only camera crop."""

import unittest

import cv2
import numpy as np

from numai.vision import locate_digit, locate_digits


class DetectionTests(unittest.TestCase):
    def test_finds_all_digits_including_close_neighbors(self):
        frame = np.full((480, 640), 255, np.uint8)
        points = [(25, 30), (60, 30), (550, 320)]
        for x, y in points:
            cv2.line(frame, (x, y), (x, y+90), 0, 9)
        detections = locate_digits(frame)
        self.assertEqual(len(detections), 3)
        for (digit, box), (x, y) in zip(detections, points):
            self.assert_square_contains(box, [(x, y), (x, y+90)], frame.shape)
            self.assertGreater(digit.sum(), 0)

    def test_shadowed_paper_and_broken_marker_stroke(self):
        frame = np.tile(np.linspace(150, 245, 640, dtype=np.uint8), (480, 1))
        cv2.line(frame, (80, 70), (80, 120), 10, 9)
        cv2.line(frame, (80, 137), (80, 190), 10, 9)
        detections = locate_digits(frame)
        self.assertEqual(len(detections), 1)
        digit, box = detections[0]
        self.assert_square_contains(box, [(80, 70), (80, 190)], frame.shape)
        # Both pieces survive normalization, not just the larger piece.
        self.assertGreater(np.count_nonzero(digit.sum(axis=1)), 16)

    def test_dashed_messy_marker_is_still_one_digit(self):
        frame = np.full((480, 640), 255, np.uint8)
        for y0, y1 in ((80, 110), (113, 145), (148, 185)):
            cv2.line(frame, (120, y0), (120, y1), 0, 11)
        detections = locate_digits(frame)
        self.assertEqual(len(detections), 1)
        digit, box = detections[0]
        self.assert_square_contains(box, [(120, 80), (120, 185)], frame.shape)
        self.assertGreater(digit.sum(), 0)

    def test_shadowed_and_bright_digits_are_found_together(self):
        frame = np.tile(np.linspace(150, 245, 640, dtype=np.uint8), (480, 1))
        for x in (30, 550):
            cv2.line(frame, (x, 70), (x, 190), 10, 9)
        self.assertEqual(len(locate_digits(frame)), 2)

    def assert_square_contains(self, box, points, shape):
        x, y, w, h = box
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, shape[1])
        self.assertLessEqual(y + h, shape[0])
        for px, py in points:
            self.assertTrue(x <= px < x + w)
            self.assertTrue(y <= py < y + h)

    def test_square_follows_position_and_scale(self):
        sides = []
        for x, y, length in ((40, 30, 40), (550, 330, 90), (290, 110, 180)):
            with self.subTest(x=x, y=y, length=length):
                frame = np.full((480, 640, 3), 255, np.uint8)
                cv2.line(frame, (x, y), (x, y + length), (0, 0, 0), 7)
                digit, box = locate_digit(frame)
                self.assertIsNotNone(digit)
                self.assertEqual(digit.shape, (28, 28))
                self.assert_square_contains(box, [(x, y), (x, y + length)], frame.shape)
                sides.append(max(box[2], box[3]))
        self.assertLess(sides[0], sides[1])
        self.assertLess(sides[1], sides[2])

    def test_paper_on_dark_background_and_noise(self):
        frame = np.full((480, 640), 35, np.uint8)
        frame[40:300, 350:610] = 245
        cv2.line(frame, (480, 100), (480, 220), 0, 9)
        cv2.circle(frame, (550, 80), 1, 0, -1)
        digit, box = locate_digit(frame)
        self.assertIsNotNone(digit)
        self.assert_square_contains(box, [(480, 100), (480, 220)], frame.shape)

    def test_blank_low_contrast_clipped_and_multiple_digits(self):
        frames = [None, np.empty((0, 0), np.uint8), np.zeros((20, 20), np.uint8)]
        frames += [np.full((480, 640), value, np.uint8) for value in (0, 127, 255)]
        faint = np.full((480, 640), 245, np.uint8)
        cv2.line(faint, (50, 50), (50, 150), 235, 7)
        frames.append(faint)
        clipped = np.full((480, 640), 255, np.uint8)
        cv2.line(clipped, (50, 0), (50, 100), 0, 7)
        frames.append(clipped)
        multiple = np.full((480, 640), 255, np.uint8)
        for x in (70, 530):
            cv2.line(multiple, (x, 50), (x, 150), 0, 7)
        frames.append(multiple)
        for frame in frames:
            with self.subTest(shape=None if frame is None else frame.shape):
                digit, box = locate_digit(frame)
                self.assertIsNone(digit)
                self.assertIsNone(box)

    def test_square_stays_inside_frame_near_edge(self):
        frame = np.full((240, 320), 255, np.uint8)
        cv2.line(frame, (8, 70), (8, 160), 0, 5)
        digit, box = locate_digit(frame)
        self.assertIsNotNone(digit)
        self.assert_square_contains(box, [(8, 70), (8, 160)], frame.shape)

    def test_tshirt_folds_are_not_digits(self):
        rng = np.random.default_rng(0)
        # Mid-tone shirt with hard seams and soft wrinkles — no paper.
        for base, soft in ((118, False), (165, True)):
            with self.subTest(base=base, soft=soft):
                frame = np.full((480, 640), base, np.uint8)
                frame = np.clip(frame.astype(np.int16) + rng.integers(-18, 19, frame.shape),
                                0, 255).astype(np.uint8)
                for x in (180, 320, 460):
                    if soft:
                        band = np.zeros_like(frame)
                        cv2.line(band, (x, 90), (x, 360), 255, 25)
                        band = cv2.GaussianBlur(band, (21, 21), 0)
                        frame = np.clip(frame.astype(np.int16) -
                                        (band.astype(np.int16) * 0.55).astype(np.int16),
                                        0, 255).astype(np.uint8)
                    else:
                        cv2.line(frame, (x, 90), (x, 360), int(max(25, base - 80)), 14)
                        cv2.line(frame, (x + 18, 120), (x + 8, 340), int(max(35, base - 60)), 7)
                self.assertEqual(locate_digits(frame), [])

    def test_digit_on_paper_ignored_next_to_clothing(self):
        rng = np.random.default_rng(1)
        frame = np.full((480, 640), 120, np.uint8)
        frame = np.clip(frame.astype(np.int16) + rng.integers(-18, 19, frame.shape), 0, 255).astype(np.uint8)
        cv2.line(frame, (80, 100), (80, 320), 45, 16)
        frame[60:280, 360:600] = 245
        cv2.line(frame, (480, 100), (480, 220), 0, 9)
        detections = locate_digits(frame)
        self.assertEqual(len(detections), 1)
        self.assert_square_contains(detections[0][1], [(480, 100), (480, 220)], frame.shape)


if __name__ == '__main__':
    unittest.main()
