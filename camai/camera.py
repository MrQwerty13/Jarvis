"""camAI camera for number recognition."""

import sys
import time

import cv2

from numai.camera import StableTracks, draw_preview, recognize_frame
from numai.model import MLP


def run_camera(model, index=0, threshold=0.65, stable_frames=4, preview=True, max_frames=0,
               speak_fn=None):
    """Run number recognition and return spoken number events.

    Face and finger recognition remain in detectors.py for later development,
    but are intentionally disabled in the camera loop for now.
    """
    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError('Не удалось открыть камеру camAI.')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    number_tracks = StableTracks(stable_frames)
    events = []
    frames = 0
    window = 'camAI - camera - Q to quit'

    def say(text):
        print(text, flush=True)
        events.append(text)
        if speak_fn:
            speak_fn(text)

    print('camAI включён: распознавание чисел. Q/Esc для выхода.', flush=True)
    try:
        while True:
            started = time.monotonic()
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError('camAI не получил кадр с камеры.')
            readings = recognize_frame(model, frame, threshold)
            number_events, removed = number_tracks.update(readings)
            for _, reading in number_events:
                say(f'Вижу число: {reading.value}')
            if removed:
                print('Число убрано или не распознано.', flush=True)

            # Распознавание лиц временно отключено.
            # faces = detect_faces(frame)
            # Распознавание пальцев временно отключено.
            # fingers, finger_box = count_fingers(frame)

            if preview:
                shown = draw_preview(frame, readings)
                cv2.imshow(window, shown)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            frames += 1
            if max_frames and frames >= max_frames:
                break
            time.sleep(max(0, 0.05 - (time.monotonic() - started)))
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()
    return events
