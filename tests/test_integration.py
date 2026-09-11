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
    def test_camera_reports_two_simultaneous_digits_independently(self):
        model = MLP.load(ROOT/'models/arabic.npz')
        images = read_idx(ROOT/'data/mnist/t10k-images-idx3-ubyte.gz')
        labels = read_idx(ROOT/'data/mnist/t10k-labels-idx1-ubyte.gz')
        raw = images[np.flatnonzero(labels == 7)[0]]
        frame = np.full((480, 640, 3), 255, np.uint8)
        for x, y in ((230, 155), (320, 235)):
            frame[y:y+90, x:x+90] = cv2.cvtColor(255-cv2.resize(raw, (90, 90)), cv2.COLOR_GRAY2BGR)
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame.copy()) for _ in range(8)]
        output = io.StringIO()
        with patch('numai.camera.cv2.VideoCapture', return_value=camera), \
                patch('numai.camera.time.sleep'), contextlib.redirect_stdout(output):
            run_camera(model, preview=False, max_frames=8)
        self.assertEqual(output.getvalue().count('Вижу число: 7'), 2)
        self.assertIn('объект 1', output.getvalue())
        self.assertIn('объект 2', output.getvalue())

    def test_camera_finds_moving_off_center_digit_and_resizes_preview(self):
        model = MLP.load(ROOT/'models/arabic.npz')
        images = read_idx(ROOT/'data/mnist/t10k-images-idx3-ubyte.gz')
        labels = read_idx(ROOT/'data/mnist/t10k-labels-idx1-ubyte.gz')
        raw = images[np.flatnonzero(labels == 7)[0]]
        frames = []
        for left, top, size in ((230, 150, 84), (300, 150, 100), (240, 220, 110)):
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
        # Large jumps create new tracks; each location must recognize the number.
        self.assertEqual(output.getvalue().count('Вижу число: 7'), 3)
        sides = []
        for index in (0, 5, 10):
            shown = show.call_args_list[index].args[1]
            ys, xs = np.where(np.all(shown == (0, 200, 0), axis=2))
            # The status label is above y=40; remaining green pixels are the box.
            xs, ys = xs[ys > 40], ys[ys > 40]
            self.assertGreater(len(xs), 0)
            sides.append(int(max(xs.max() - xs.min(), ys.max() - ys.min())))
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
        self.assertEqual(output.getvalue().count('Вижу число: 7'), 2)
        self.assertIn('Число убрано или не распознано.', output.getvalue())
        camera.release.assert_called_once()

    def test_camera_reads_negative_two_digit_number(self):
        model = MLP.load(ROOT/'models/arabic.npz')
        images = read_idx(ROOT/'data/mnist/t10k-images-idx3-ubyte.gz')
        labels = read_idx(ROOT/'data/mnist/t10k-labels-idx1-ubyte.gz')
        ones = images[np.flatnonzero(labels == 1)]
        zeros = images[np.flatnonzero(labels == 0)]
        frame = np.full((480, 640, 3), 255, np.uint8)
        cv2.line(frame, (245, 235), (270, 235), (0, 0, 0), 6)
        for left, raw in ((278, ones[0]), (316, zeros[1] if len(zeros) > 1 else zeros[0])):
            ink = 255 - cv2.resize(raw, (32, 32))
            frame[219:251, left:left+32] = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame.copy()) for _ in range(8)]
        output = io.StringIO()
        with patch('numai.camera.cv2.VideoCapture', return_value=camera), \
                patch('numai.camera.time.sleep'), contextlib.redirect_stdout(output):
            run_camera(model, preview=False, max_frames=8)
        self.assertIn('Вижу число: -10', output.getvalue())


if __name__ == '__main__':
    unittest.main()
