"""Камера → область листа → собственная MLP → терминал."""

import sys
import time

import cv2
import numpy as np

from .vision import locate_digit


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
        stable = StablePrediction(stable_frames)
        frames = 0
        print('Камера включена. Покажите одну цифру 0–9 на светлом листе в любой части кадра.', flush=True)
        print('Выход: Ctrl+C в терминале или Q / Esc в окне камеры.', flush=True)
        while True:
            started = time.monotonic()
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError('Камера открылась, но кадр не получен. Проверьте устройство и разрешения.')
            height, width = frame.shape[:2]
            digit, box = locate_digit(frame)
            number, score = classify(model, digit, threshold)
            previous = stable.last_output
            output = stable.update(number)
            if output is not None:
                print(f'Вижу цифру: {output} (оценка модели: {score:.0%})', flush=True)
            elif previous is not None and stable.last_output is None:
                print('Цифра убрана или не распознана.', flush=True)
            if preview:
                color = (0, 200, 0) if number is not None else (0, 180, 255)
                if box is not None:
                    x, y, side = box
                    cv2.rectangle(frame, (x, y), (x+side-1, y+side-1), color, 2)
                label = f'{number} ({score:.0%})' if number is not None else (
                    'Uncertain digit' if box is not None else 'Searching for one digit...')
                cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                processed = np.zeros((28, 28), np.float32) if digit is None else digit
                thumbnail = cv2.resize((processed*255).astype(np.uint8), (112, 112), interpolation=cv2.INTER_NEAREST)
                if height >= 162 and width >= 127:
                    frame[height-127:height-15, 15:127] = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)
                cv2.imshow('NumAI - automatic digit detection - Q to quit', frame)
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
