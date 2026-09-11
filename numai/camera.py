"""Камера → область листа → собственная MLP → терминал."""

from dataclasses import dataclass
import sys
import time

import cv2
import numpy as np

from .vision import locate_digits


class StablePrediction:
    def __init__(self, frames=5):
        if frames < 1:
            raise ValueError('Число кадров должно быть положительным.')
        self.frames = frames
        self.candidate = None
        self.count = 0
        self.last_output = None

    def update(self, digit):
        self.count = self.count + 1 if digit == self.candidate else 1
        self.candidate = digit
        if self.count < self.frames:
            return None
        if digit is None:
            self.last_output = None
        elif digit != self.last_output:
            self.last_output = digit
            return digit
        return None


def classify(model, digit, threshold=0.85):
    if digit is None:
        return None, 0.0
    probabilities = model.predict_proba(digit.reshape(1, -1))[0]
    order = np.argsort(probabilities)
    best, second = int(order[-1]), int(order[-2])
    score = float(probabilities[best])
    if score < threshold or score-float(probabilities[second]) < 0.25:
        return None, score
    return best, score


@dataclass
class Reading:
    digit: np.ndarray
    box: tuple
    number: object
    score: float


def recognize_frame(model, frame, threshold=0.85):
    return [Reading(digit, box, *classify(model, digit, threshold))
            for digit, box in locate_digits(frame)]


class StableTracks:
    """Associate nearby boxes and require independent agreement per digit."""
    def __init__(self, frames=5):
        self.frames = frames
        self.tracks = {}
        self.next_id = 1

    def update(self, readings):
        pairs = []
        for index, reading in enumerate(readings):
            x, y, side = reading.box
            for track_id, track in self.tracks.items():
                tx, ty, ts = track['box']
                distance = np.hypot(x+side/2-tx-ts/2, y+side/2-ty-ts/2)
                if distance <= max(side, ts)*0.8 and max(side, ts)/min(side, ts) <= 2.5:
                    pairs.append((distance/max(side, ts), index, track_id))
        matched, used = {}, set()
        for _, index, track_id in sorted(pairs):
            if index not in matched and track_id not in used:
                matched[index] = track_id
                used.add(track_id)
        events, removed = [], False
        for track_id in list(self.tracks):
            if track_id in used:
                continue
            track = self.tracks[track_id]
            previous = track['stable'].last_output
            track['stable'].update(None)
            track['missed'] += 1
            if track['missed'] >= self.frames:
                removed |= previous is not None
                del self.tracks[track_id]
        for index, reading in enumerate(readings):
            track_id = matched.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {'stable': StablePrediction(self.frames)}
            track = self.tracks[track_id]
            track.update(box=reading.box, missed=0)
            previous = track['stable'].last_output
            output = track['stable'].update(reading.number)
            removed |= previous is not None and track['stable'].last_output is None
            if output is not None:
                events.append((track_id, reading))
        return events, removed


def draw_preview(frame, readings):
    """Keep diagnostics outside the camera image so no digit is covered."""
    height, width = frame.shape[:2]
    preview = np.full((height, width+150, 3), 35, np.uint8)
    preview[:, :width] = frame
    for index, reading in enumerate(readings):
        x, y, side = reading.box
        color = (0, 200, 0) if reading.number is not None else (0, 180, 255)
        cv2.rectangle(preview, (x, y), (x+side-1, y+side-1), color, 2)
        label = f'{reading.number} {reading.score:.0%}' if reading.number is not None else '?'
        cv2.putText(preview, label, (x, max(18, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        top = 35+index*140
        if top+112 <= height:
            thumbnail = cv2.resize((reading.digit*255).astype(np.uint8), (112, 112), interpolation=cv2.INTER_NEAREST)
            preview[top:top+112, width+19:width+131] = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)
            cv2.putText(preview, label, (width+19, top-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    if not readings:
        cv2.putText(preview, 'Searching...', (width+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return preview


def run_camera(model, index=0, threshold=0.85, stable_frames=5, preview=True, max_frames=0):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                'Не удалось открыть камеру. Проверьте --camera, закройте другие приложения с камерой. '
                'В macOS разрешите доступ приложению, из которого запущен Python: '
                'Системные настройки → Конфиденциальность и безопасность → Камера.'
            )
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        stable = StableTracks(stable_frames)
        frames = 0
        print('Камера включена. Покажите цифры 0–9 на светлом листе в любой части кадра.', flush=True)
        print('Выход: Ctrl+C в терминале или Q / Esc в окне камеры.', flush=True)
        while True:
            started = time.monotonic()
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError('Камера открылась, но кадр не получен. Проверьте устройство и разрешения.')
            readings = recognize_frame(model, frame, threshold)
            events, removed = stable.update(readings)
            for track_id, reading in events:
                print(f'Вижу цифру: {reading.number} (оценка модели: {reading.score:.0%}; объект {track_id})', flush=True)
            if removed:
                print('Цифра убрана или не распознана.', flush=True)
            if preview:
                cv2.imshow('NumAI - automatic digit detection - Q to quit', draw_preview(frame, readings))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                if cv2.getWindowProperty('NumAI - automatic digit detection - Q to quit', cv2.WND_PROP_VISIBLE) < 1:
                    break
            frames += 1
            if max_frames and frames >= max_frames:
                break
            time.sleep(max(0, 0.1 - (time.monotonic()-started)))
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()
