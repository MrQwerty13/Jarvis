import unittest

import cv2
import numpy as np

from numai.numbers import (
    MINUS,
    Symbol,
    group_symbols,
    looks_like_minus,
    parse_number,
)
from numai.vision import locate_digits


def symbol(value, x, y=40, w=30, h=70, score=0.99):
    return Symbol(np.zeros((28, 28), np.float32), (x, y, w, h), value, score)


class NumberParseTests(unittest.TestCase):
    def test_parses_signed_range_and_rejects_invalid(self):
        self.assertEqual(parse_number([symbol(7, 0)]), 7)
        self.assertEqual(parse_number([symbol(1, 0), symbol(0, 40), symbol(0, 80)]), 100)
        self.assertEqual(parse_number([symbol(MINUS, 0, w=40, h=12), symbol(4, 50), symbol(2, 90)]), -42)
        self.assertEqual(parse_number([symbol(MINUS, 0, w=40, h=12), symbol(1, 50),
                                       symbol(0, 90), symbol(0, 130)]), -100)
        self.assertIsNone(parse_number([symbol(1, 0), symbol(0, 40), symbol(1, 80)]))
        self.assertIsNone(parse_number([symbol(0, 0), symbol(7, 40)]))
        self.assertIsNone(parse_number([symbol(MINUS, 0, w=40, h=12)]))
        self.assertIsNone(parse_number([symbol(4, 0), symbol(MINUS, 50, w=40, h=12), symbol(2, 100)]))

    def test_groups_nearby_glyphs_and_keeps_distant_numbers_apart(self):
        nearby = [symbol(4, 20), symbol(2, 55)]
        groups = group_symbols(nearby)
        self.assertEqual(len(groups), 1)
        self.assertEqual(parse_number(groups[0]), 42)
        distant = [symbol(7, 20), symbol(7, 400)]
        groups = group_symbols(distant)
        self.assertEqual(len(groups), 2)
        self.assertEqual([parse_number(group) for group in groups], [7, 7])

    def test_minus_shape_and_locate(self):
        ink = np.zeros((28, 28), np.float32)
        ink[13:16, 4:24] = 1
        self.assertTrue(looks_like_minus((100, 100, 50, 12), ink))
        tall = np.zeros((28, 28), np.float32)
        tall[4:24, 13:16] = 1
        self.assertFalse(looks_like_minus((100, 100, 12, 50), tall))
        frame = np.full((480, 640), 255, np.uint8)
        cv2.line(frame, (120, 160), (200, 160), 0, 8)
        cv2.line(frame, (230, 110), (230, 210), 0, 9)
        cv2.line(frame, (270, 110), (270, 210), 0, 9)
        detections = locate_digits(frame)
        self.assertGreaterEqual(len(detections), 3)
        minus = min(detections, key=lambda item: item[1][3])
        self.assertTrue(looks_like_minus(minus[1], minus[0]))


if __name__ == '__main__':
    unittest.main()
