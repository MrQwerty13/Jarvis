"""Выделение тёмной цифры на светлом листе и нормализация к 28×28."""

import cv2
import numpy as np


def normalize_digit(ink):
    """Вход: светлый штрих на чёрном фоне, uint8. Выход: float32 [0, 1]."""
    points = cv2.findNonZero((ink > 20).astype(np.uint8))
    if points is None:
        return np.zeros((28, 28), np.float32)
    x, y, width, height = cv2.boundingRect(points)
    crop = ink[y:y+height, x:x+width]
    scale = 20 / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.float32)
    left, top = (28-w)//2, (28-h)//2
    canvas[top:top+h, left:left+w] = resized.astype(np.float32) / 255
    moments = cv2.moments(canvas)
    if moments['m00']:
        dx = 13.5 - moments['m10'] / moments['m00']
        dy = 13.5 - moments['m01'] / moments['m00']
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return canvas


def extract_digit(image):
    """Возвращает одну цифру либо None для пустого/неподходящего кадра.

    Это геометрический фильтр, а не обученный детектор произвольных объектов.
    """
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if min(gray.shape) < 28:
        return None
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if int(gray.max()) - int(gray.min()) < 25:
        return None
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = gray.shape
    minimum_area = max(15, height * width * 0.0008)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < minimum_area:
            continue
        # Края листа, обрезанный символ и фон не должны становиться цифрой.
        if x <= 1 or y <= 1 or x+w >= width-1 or y+h >= height-1:
            return None
        candidates.append((int(area), label))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    area, label = candidates[0]
    x, y, w, h, _ = stats[label]
    if h < height * 0.15 or max(w, h) > min(height, width) * 0.9:
        return None
    if area > height * width * 0.4:
        return None
    if len(candidates) > 1 and candidates[1][0] > area * 0.2:
        return None
    ink = np.where(labels == label, 255, 0).astype(np.uint8)
    return normalize_digit(ink)


def locate_digit(image):
    """Find one dark digit anywhere in a frame; return (28×28 ink, (x, y, side)).

    Localization is geometric; the existing MLP determines the digit afterwards.
    Border-connected background is ignored, and competing symbols are rejected.
    """
    if image is None or image.size == 0:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    if min(height, width) < 28:
        return None, None
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if int(blurred.max()) - int(blurred.min()) < 25:
        return None, None
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    minimum_height = max(18, min(height, width) * 0.035)
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if x <= 1 or y <= 1 or x+w >= width-1 or y+h >= height-1:
            continue
        if h < minimum_height or max(w, h) > min(height, width) * 0.85:
            continue
        if area < 25 or not 0.04 <= w/h <= 1.4 or area/(w*h) < 0.04:
            continue
        # A thin solid stroke may be a 1; a broad filled block is background.
        if w/h > 0.3 and area/(w*h) > 0.9:
            continue
        candidates.append((area, label))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    area, label = candidates[0]
    if len(candidates) > 1 and candidates[1][0] > area * 0.2:
        return None, None
    x, y, w, h, _ = map(int, stats[label])
    side = min(min(height, width), max(28, int(np.ceil(max(w, h) * 1.5))))
    left = min(max(0, x + w//2 - side//2), width - side)
    top = min(max(0, y + h//2 - side//2), height - side)
    # Recheck the local crop: paper edges, shadows, and nearby ink must not
    # silently enter the network as part of the selected symbol.
    digit = extract_digit(gray[top:top+side, left:left+side])
    if digit is None:
        return None, None
    return digit, (left, top, side)


def augment(images, rng):
    """Небольшие повороты, сдвиги и изменения толщины штриха только при обучении."""
    result = np.empty_like(images)
    for i, image in enumerate(images):
        matrix = cv2.getRotationMatrix2D((13.5, 13.5), rng.uniform(-12, 12), rng.uniform(0.9, 1.1))
        matrix[:, 2] += rng.uniform(-1.5, 1.5, size=2)
        transformed = cv2.warpAffine(image, matrix, (28, 28))
        choice = rng.random()
        if choice < 0.12:
            transformed = cv2.dilate(transformed, np.ones((2, 2), np.uint8))
        elif choice < 0.2:
            thinner = cv2.erode(transformed, np.ones((2, 2), np.uint8))
            if thinner.sum() > transformed.sum() * 0.3:
                transformed = thinner
        result[i] = transformed
    return result
