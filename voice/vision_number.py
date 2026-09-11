"""Голосовой режим: одно стабильное число с камеры NumAI."""

import sys
import time
from pathlib import Path

import cv2

from numai.camera import StableTracks, recognize_frame
from numai.model import MLP


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / 'models' / 'arabic.npz'


def read_number_from_camera(
    model_path=None,
    camera_index=0,
    threshold=0.85,
    stable_frames=5,
    timeout_sec=25,
    preview=False,
):
    path = Path(model_path or DEFAULT_MODEL)
    if not path.exists():
        raise FileNotFoundError('Нет весов NumAI. Сначала: python -m numai train')
    model = MLP.load(path)
    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    camera = cv2.VideoCapture(camera_index, backend)
    try:
        if not camera.isOpened():
            raise RuntimeError('Не удалось открыть камеру для распознавания чисел.')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        stable = StableTracks(stable_frames)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            readings = recognize_frame(model, frame, threshold)
            events, _removed = stable.update(readings)
            if preview:
                cv2.imshow('Jarvis numbers - Q to quit', frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            for _track_id, reading in events:
                if reading.value is not None:
                    return {
                        'value': reading.value,
                        'score': float(reading.score),
                    }
            time.sleep(0.08)
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()
    return None
