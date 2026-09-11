"""Запуск NumAI-камеры и озвучка распознанных чисел."""

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NUMBER_LINE_RE = re.compile(r'Вижу число:\s*(-?\d+)')


def run_numai_camera_session(
    camera_index=0,
    speak_fn=None,
    on_number=None,
    preview=True,
    threshold=0.85,
):
    """
    Запускает тот же режим, что и `python -m numai camera`.
    Парсит stdout и вызывает speak/on_number для каждого нового числа.
    """
    command = [
        sys.executable,
        '-m',
        'numai',
        'camera',
        '--camera',
        str(camera_index),
        '--threshold',
        str(threshold),
    ]
    if not preview:
        command.append('--no-preview')

    print('Запуск: ' + ' '.join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    spoken = []
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if line:
                print(line, flush=True)
            match = NUMBER_LINE_RE.search(line)
            if not match:
                continue
            value = match.group(1)
            spoken.append(value)
            phrase = f'Вижу число {value}'
            if on_number:
                on_number(value, phrase)
            if speak_fn:
                speak_fn(phrase)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return spoken
