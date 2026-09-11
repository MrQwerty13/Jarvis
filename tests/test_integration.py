"""Проверки реальной модели; пропускаются, если локальные веса ещё не обучены."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from numai.camera import run_camera
from numai.data import read_idx
from numai.model import MLP


ROOT = Path(__file__).resolve().parent.parent


class CameraFailureTests(unittest.TestCase):
    def test_unavailable_camera_is_released_with_actionable_error(self):
        camera = MagicMock()
        camera.isOpened.return_value = False
        with patch('numai.camera.cv2.VideoCapture', return_value=camera):
            with self.assertRaisesRegex(RuntimeError, 'Не удалось открыть камеру'):
                run_camera(MLP(), preview=False)
        camera.release.assert_called_once()

    def test_failed_frame_is_reported_and_camera_released(self):
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        with patch('numai.camera.cv2.VideoCapture', return_value=camera), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'кадр не получен'):
                run_camera(MLP(), preview=False)
        camera.release.assert_called_once()


@unittest.skipUnless((ROOT/'models/arabic.npz').exists() and
                     (ROOT/'data/mnist/t10k-images-idx3-ubyte.gz').exists(),
                     'Сначала обучите локальную модель: python -m numai train')
class TrainedPipelineTests(unittest.TestCase):
    def test_camera_finds_moving_off_center_digit_and_resizes_preview(self):
        model = MLP.load(ROOT/'models/arabic.npz')
        images = read_idx(ROOT/'data/mnist/t10k-images-idx3-ubyte.gz')
        labels = read_idx(ROOT/'data/mnist/t10k-labels-idx1-ubyte.gz')
        raw = images[np.flatnonzero(labels == 7)[0]]
        frames = []
        for left, top, size in ((5, 10, 84), (475, 300, 140), (210, 100, 196)):
            for _ in range(5):
                frame = np.full((480, 640, 3), 255, np.uint8)
                ink = 255-cv2.resize(raw, (size, size))
                frame[top:top+size, left:left+size] = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
                frames.append(frame)
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame) for frame in frames]
        output = io.StringIO()
        # Mock only camera/GUI boundaries; use real detection, model, and drawing.
        with patch('numai.camera.cv2.VideoCapture', return_value=camera), \
                patch('numai.camera.time.sleep'), \
                patch('numai.camera.cv2.imshow') as show, \
                patch('numai.camera.cv2.waitKey', return_value=-1), \
                patch('numai.camera.cv2.getWindowProperty', return_value=1), \
                patch('numai.camera.cv2.destroyAllWindows'), \
                contextlib.redirect_stdout(output):
            run_camera(model, preview=True, max_frames=len(frames))
        self.assertEqual(output.getvalue().count('Вижу цифру: 7'), 1)
        sides = []
        for index in (0, 5, 10):
            shown = show.call_args_list[index].args[1]
            ys, xs = np.where(np.all(shown == (0, 200, 0), axis=2))
            # The status label is above y=40; remaining green pixels are the box.
            xs, ys = xs[ys > 40], ys[ys > 40]
            self.assertGreater(len(xs), 0)
            sides.append(int(xs.max() - xs.min()))
        self.assertLess(sides[0], sides[1])
        self.assertLess(sides[1], sides[2])
        camera.release.assert_called_once()

    def test_camera_replay_recognizes_seven_and_resets_after_blank(self):
        model = MLP.load(ROOT/'models/arabic.npz')
        images = read_idx(ROOT/'data/mnist/t10k-images-idx3-ubyte.gz')
        labels = read_idx(ROOT/'data/mnist/t10k-labels-idx1-ubyte.gz')
        raw = images[np.flatnonzero(labels == 7)[0]]
        white = np.full((480, 640, 3), 255, np.uint8)
        card = white.copy()
        ink = 255-cv2.resize(raw, (196, 196))
        card[142:338, 222:418] = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
        frames = [card]*7 + [white]*7 + [card]*7
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame) for frame in frames]
        output = io.StringIO()
        with patch('numai.camera.cv2.VideoCapture', return_value=camera), \
                patch('numai.camera.time.sleep'), contextlib.redirect_stdout(output):
            run_camera(model, preview=False, max_frames=len(frames))
        self.assertEqual(output.getvalue().count('Вижу цифру: 7'), 2)
        self.assertIn('Цифра убрана или не распознана.', output.getvalue())
        camera.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
