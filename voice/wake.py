"""Разбор обращения по имени Джарвис / Jarvis."""

import re

# Без \\b: для кириллицы граница слова в ряде движков ломается.
WAKE_RE = re.compile(
    r'^\s*(?:джарвис|jarvis)(?:[\s,.:!\-]+|$)(.*)$',
    re.IGNORECASE,
)


def extract_command(text):
    """
    Вернуть текст команды после имени или None, если обращения нет.
    Пустая строка значит, что сказали только имя.
    """
    normalized = (text or '').lower().replace('ё', 'е').strip()
    match = WAKE_RE.match(normalized)
    if not match:
        return None
    return match.group(1).strip()
