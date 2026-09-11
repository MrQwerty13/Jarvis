"""Group digit/minus glyphs into integers from -100 to 100."""

from dataclasses import dataclass

import numpy as np


MINUS = 'minus'


@dataclass
class Symbol:
    ink: object
    box: tuple  # x, y, w, h in the frame
    value: object  # 0-9, MINUS, or None
    score: float


def looks_like_minus(box, ink):
    """Geometric minus: wide short stroke, not a vertical digit."""
    _, _, width, height = box
    if height < 3 or width < 10 or width / max(height, 1) < 1.7:
        return False
    if width / max(height, 1) > 12:
        return False
    rows = ink.sum(axis=1)
    cols = ink.sum(axis=0)
    if cols.sum() <= 0:
        return False
    active_rows = int(np.count_nonzero(rows > rows.max() * 0.2))
    active_cols = int(np.count_nonzero(cols > cols.max() * 0.2))
    return active_cols >= 12 and active_rows <= 10 and active_cols > active_rows * 1.4


def parse_number(symbols):
    """Return an int in [-100, 100], or None if the glyph sequence is invalid."""
    if not symbols or any(symbol.value is None for symbol in symbols):
        return None
    tokens = [symbol.value for symbol in symbols]
    sign = 1
    if tokens[0] == MINUS:
        sign = -1
        tokens = tokens[1:]
    if not tokens or any(token == MINUS for token in tokens):
        return None
    if not all(isinstance(token, int) and 0 <= token <= 9 for token in tokens):
        return None
    if len(tokens) > 3 or (len(tokens) > 1 and tokens[0] == 0):
        return None
    value = 0
    for token in tokens:
        value = value * 10 + token
    value *= sign
    if value < -100 or value > 100:
        return None
    return value


def _same_number(left, right):
    ax, ay, aw, ah = left.box
    bx, by, bw, bh = right.box
    gap = bx - (ax + aw)
    height = max(ah, bh, 1)
    if gap > height * 1.15 or gap < -0.25 * height:
        return False
    if abs((ay + ah / 2) - (by + bh / 2)) > height * 0.55:
        return False
    ratio = max(ah, bh) / max(min(ah, bh), 1)
    # A minus is naturally much shorter than digits; allow that mismatch.
    if MINUS in (left.value, right.value):
        return ratio <= 8.0
    return ratio <= 2.8


def group_symbols(symbols):
    """Cluster left-to-right glyphs that belong to the same written number."""
    if not symbols:
        return []
    ordered = sorted(symbols, key=lambda symbol: (symbol.box[1] // max(symbol.box[3], 1), symbol.box[0]))
    lines = [[ordered[0]]]
    for symbol in ordered[1:]:
        previous = lines[-1][-1]
        py = previous.box[1] + previous.box[3] / 2
        sy = symbol.box[1] + symbol.box[3] / 2
        height = max(previous.box[3], symbol.box[3], 1)
        if abs(py - sy) > height * 0.8:
            lines.append([symbol])
        else:
            lines[-1].append(symbol)
    groups = []
    for line in lines:
        line = sorted(line, key=lambda symbol: symbol.box[0])
        group = [line[0]]
        for symbol in line[1:]:
            if _same_number(group[-1], symbol):
                group.append(symbol)
            else:
                groups.append(group)
                group = [symbol]
        groups.append(group)
    return groups


def number_box(symbols):
    x0 = min(symbol.box[0] for symbol in symbols)
    y0 = min(symbol.box[1] for symbol in symbols)
    x1 = max(symbol.box[0] + symbol.box[2] for symbol in symbols)
    y1 = max(symbol.box[1] + symbol.box[3] for symbol in symbols)
    return x0, y0, x1 - x0, y1 - y0


def number_score(symbols):
    return float(min(symbol.score for symbol in symbols)) if symbols else 0.0
