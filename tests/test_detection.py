"""Automatic localization must work beyond the old center-only camera crop."""

import unittest

import cv2
import numpy as np

from numai.vision import locate_digit


class DetectionTests(unittest.TestCase):
    def assert_square_contains(self, box, points, shape):
        x, y, side = box
        self.assertGreater(side, 0)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + side, shape[1])
        self.assertLessEqual(y + side, shape[0])
        for px, py in points:
            self.assertTrue(x <= px < x + side)
            self.assertTrue(y <= py < y + side)

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
                sides.append(box[2])
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


if __name__ == '__main__':
    unittest.main()
