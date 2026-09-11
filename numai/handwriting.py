"""Fixed handwriting stress cases, separate from random training augmentation."""

import cv2
import numpy as np

from .vision import normalize_digit

# Keep in sync with numai.benchmark variant names.
STRESS_VARIANTS = (
    'thick_marker',
    'slant',
    'rotation',
    'stretch_marker',
    'messy_marker',
    'broken_stroke',
)


def stress_images(images):
    """Deterministic marker/messy/slant variants; not a real-camera benchmark."""
    result = np.empty_like(images)
    for i, image in enumerate(images):
        variant = i % len(STRESS_VARIANTS)
        ink = image.copy()
        if variant == 0:
            ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
            ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
        elif variant == 1:
            shear = 0.35 if i % 12 == 1 else -0.35
            ink = cv2.warpAffine(ink, np.float32([[1, shear, -13.5*shear], [0, 1, 0]]), (28, 28))
        elif variant == 2:
            matrix = cv2.getRotationMatrix2D((13.5, 13.5), 22 if i % 12 == 2 else -22, 1)
            ink = cv2.warpAffine(ink, matrix, (28, 28))
        elif variant == 3:
            ink = cv2.warpAffine(ink, np.float32([[0.7, 0.2, 1.5], [0, 1.15, -1.5]]), (28, 28))
            ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
        elif variant == 4:
            # Thick, slanted, slightly soft — closer to hurried marker on paper.
            ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
            shear = 0.25 if i % 12 == 4 else -0.25
            ink = cv2.warpAffine(ink, np.float32([[1, shear, -13.5*shear], [0, 1, 0]]), (28, 28))
            ink = cv2.GaussianBlur(ink, (3, 3), 0.8)
        else:
            # Broken / lifted strokes with leftover marker thickness.
            y = 10 + (i % 5)
            ink[y:y+2, :] = 0
            x = 9 + ((i // 6) % 5)
            ink[:, x:x+2] = 0
            ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
        result[i] = normalize_digit(np.clip(ink*255, 0, 255).astype(np.uint8))
    return result
