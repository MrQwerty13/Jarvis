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


def _stroke_edge_score(gray, ink_full):
    """Marker ink has a sharp edge; clothing folds are usually softer."""
    boundary = cv2.dilate(ink_full.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary &= ~ink_full
    if not boundary.any():
        return 0.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(cv2.magnitude(grad_x, grad_y)[boundary].mean())


def _on_paper(gray, background, x, y, right, bottom, ink):
    """Accept only dark ink sitting on bright, smooth paper — not clothing folds."""
    height, width = gray.shape
    box_w, box_h = right - x, bottom - y
    pad = max(12, int(0.45 * max(box_w, box_h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, right + pad), min(height, bottom + pad)
    local = gray[y0:y1, x0:x1]
    local_bg = background[y0:y1, x0:x1]
    ink_full = np.zeros(local.shape, dtype=bool)
    ink_full[y - y0:bottom - y0, x - x0:right - x0] = ink > 0
    if ink_full.sum() < 8:
        return False
    exclude = cv2.dilate(ink_full.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    support = ~exclude
    if support.sum() < 40:
        return False
    # Sheet paper is bright and close to the local background estimate.
    paper_mask = (local_bg >= 150) & (np.abs(local.astype(np.int16) - local_bg.astype(np.int16)) < 30)
    paper_mask &= support
    if paper_mask.mean() < 0.5:
        return False
    paper_vals = local[paper_mask]
    paper_mean = float(paper_vals.mean())
    paper_std = float(paper_vals.std())
    ink_mean = float(local[ink_full].mean())
    contrast = paper_mean - ink_mean
    if contrast < 45 or paper_mean < 150:
        return False
    if paper_std > max(14.0, 0.35 * contrast):
        return False
    edge = _stroke_edge_score(local, ink_full)
    # White paper tolerates camera blur and gray anti-aliased ink.
    # Dimmer paper needs darker, sharper marker strokes than clothing folds.
    if paper_mean >= 190:
        if edge < 70 or ink_mean > 145 or ink_mean > 0.58 * paper_mean:
            return False
    elif edge < 130 or ink_mean > 45 or ink_mean > 0.35 * paper_mean:
        return False
    bg_mean = float(local_bg[paper_mask].mean())
    return bg_mean >= 150 and bg_mean - ink_mean >= 40


def locate_digits(image):
    """Return separate (normalized ink, square) candidates across the full frame.

    Estimate the local paper brightness before thresholding so a shadow does not
    hide a digit. This is geometric localization, not an object classifier.
    """
    if image is None or image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    if min(height, width) < 28:
        return []
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel_size = max(31, int(min(height, width) * 0.15) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                                np.ones((kernel_size, kernel_size), np.uint8))
    contrast = cv2.subtract(background, gray)
    # No single threshold is reliable for camera frames: exposure changes across
    # the sheet and glare can split a marker stroke. Keep several complementary
    # masks and merge their geometric detections below.
    masks = []
    for cutoff in (10, 16, 24):
        mask = np.where(contrast > cutoff, 255, 0).astype(np.uint8)
        masks.append(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    masks.append(cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))

    candidates = []
    for mask in masks:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        for label in range(1, count):
            x, y, w, h, area = map(int, stats[label])
            if area < 12 or x <= 1 or y <= 1 or x+w >= width-1 or y+h >= height-1:
                continue
            components.append([x, y, x+w, y+h, area, [label]])
        components = sorted(components, key=lambda item: item[4], reverse=True)[:128]
        # Join vertically broken strokes, while preserving side-by-side digits.
        changed = True
        while changed:
            changed = False
            for i, first in enumerate(components):
                for j in range(i+1, len(components)):
                    second = components[j]
                    overlap = min(first[2], second[2]) - max(first[0], second[0])
                    gap = max(first[1], second[1]) - min(first[3], second[3])
                    total_height = max(first[3], second[3]) - min(first[1], second[1])
                    if overlap >= 0.5 * min(first[2]-first[0], second[2]-second[0]) and 0 <= gap <= 0.12 * total_height:
                        components[i] = [min(first[0], second[0]), min(first[1], second[1]),
                                         max(first[2], second[2]), max(first[3], second[3]),
                                         first[4]+second[4], first[5]+second[5]]
                        components.pop(j)
                        changed = True
                        break
                if changed:
                    break
        for x, y, right, bottom, area, group in components:
            w, h = right-x, bottom-y
            aspect = w / max(h, 1)
            density = area / max(w * h, 1)
            minus_shape = 1.7 <= aspect <= 10.0 and density >= 0.15
            min_height = max(4, int(min(height, width) * 0.008)) if minus_shape else max(18, int(min(height, width) * 0.025))
            if h < min_height or max(w, h) > min(height, width) * 0.9 or area < 20:
                continue
            if minus_shape:
                if h > max(28, int(min(height, width) * 0.12)):
                    continue
            elif not 0.035 <= aspect <= 1.5 or density < 0.04:
                continue
            if not minus_shape and aspect > 0.3 and density > 0.9:
                continue
            ink = np.where(np.isin(labels[y:bottom, x:right], group), 255, 0).astype(np.uint8)
            if not _on_paper(gray, background, x, y, right, bottom, ink):
                continue
            candidates.append((x, y, right, bottom, area, ink))

    # The same glyph appears in several masks. Deduplicate only strongly
    # overlapping boxes so neighboring digits remain separate.
    candidates.sort(key=lambda item: item[4], reverse=True)
    unique = []
    for candidate in candidates:
        x, y, right, bottom, area, ink = candidate
        duplicate = False
        for other in unique:
            ox, oy, oright, obottom = other[:4]
            intersection = max(0, min(right, oright) - max(x, ox)) * max(0, min(bottom, obottom) - max(y, oy))
            if intersection / max(min((right-x)*(bottom-y), (oright-ox)*(obottom-oy)), 1) > 0.5:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    detections = [(normalize_digit(ink), (x, y, right-x, bottom-y))
                  for x, y, right, bottom, _, ink in unique]
    return sorted(detections, key=lambda item: (item[1][1], item[1][0]))


def locate_digit(image):
    """Compatibility helper for callers requiring exactly one candidate."""
    detections = locate_digits(image)
    return detections[0] if len(detections) == 1 else (None, None)


def _stroke_gaps(ink, rng):
    """Simulate pen lifts and broken/uneven marker strokes."""
    result = ink.copy()
    for _ in range(int(rng.integers(1, 3))):
        if rng.random() < 0.5:
            y = int(rng.integers(5, 23))
            result[y:y + int(rng.integers(1, 3)), :] *= 0
        else:
            x = int(rng.integers(5, 23))
            result[:, x:x + int(rng.integers(1, 3))] *= 0
    return result


def augment(images, rng):
    """Marker thickness, messy strokes, slant, shape variation and elastic warp."""
    result = np.empty_like(images)
    for i, image in enumerate(images):
        # Keep some clean examples in every epoch.
        if rng.random() < 0.15:
            result[i] = image
            continue
        matrix = cv2.getRotationMatrix2D((13.5, 13.5), rng.uniform(-25, 25), 1)
        shape = np.array([[rng.uniform(0.75, 1.2), rng.uniform(-0.4, 0.4)],
                          [0, rng.uniform(0.8, 1.2)]], dtype=np.float32)
        matrix[:, :2] = matrix[:, :2] @ shape
        matrix[:, 2] = 13.5 - matrix[:, :2] @ np.array([13.5, 13.5])
        transformed = cv2.warpAffine(image, matrix, (28, 28))
        if rng.random() < 0.4:
            displacement = rng.normal(0, 1.1, (4, 4, 2)).astype(np.float32)
            displacement = cv2.resize(displacement, (28, 28), interpolation=cv2.INTER_CUBIC)
            yy, xx = np.indices((28, 28), dtype=np.float32)
            transformed = cv2.remap(transformed, xx+displacement[:, :, 0],
                                    yy+displacement[:, :, 1], cv2.INTER_LINEAR)
        choice = rng.random()
        if choice < 0.28:
            # Thick marker: often heavier than MNIST pen strokes.
            size = 3 if choice < 0.14 else 2
            transformed = cv2.dilate(transformed, np.ones((size, size), np.uint8))
            if choice < 0.07:
                transformed = cv2.dilate(transformed, np.ones((2, 2), np.uint8))
        elif choice < 0.4:
            # Patchy marker: thick in places, thinner elsewhere.
            thick = cv2.dilate(transformed, np.ones((3, 3), np.uint8))
            patch = rng.random((28, 28)) < 0.45
            transformed = np.where(patch, thick, transformed)
        elif choice < 0.52:
            gapped = _stroke_gaps(transformed, rng)
            if gapped.sum() > transformed.sum() * 0.35:
                transformed = gapped
                if rng.random() < 0.5:
                    transformed = cv2.dilate(transformed, np.ones((2, 2), np.uint8))
        elif choice < 0.6:
            thinner = cv2.erode(transformed, np.ones((2, 2), np.uint8))
            if thinner.sum() > transformed.sum() * 0.3:
                transformed = thinner
        if rng.random() < 0.28:
            transformed = cv2.GaussianBlur(transformed, (3, 3), 0.7)
        if rng.random() < 0.12:
            blot = (rng.random((28, 28)) < 0.02).astype(np.float32)
            transformed = np.clip(transformed + blot * rng.uniform(0.4, 1.0), 0, 1)
        transformed = normalize_digit(np.clip(transformed*255, 0, 255).astype(np.uint8))
        shift = rng.uniform(-2, 2, size=2)
        result[i] = cv2.warpAffine(transformed, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (28, 28))
    return result
