import unittest
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

from camai.detectors import count_fingers, detect_faces
from voice.actions import ActionRouter
from voice.memory import JsonMemory


class FakeBrain:
    def answer(self, text):
        return 'ответ', 'test', 0.99


class CamAICommandTests(unittest.TestCase):
    def setUp(self):
        self.memory_dir = tempfile.TemporaryDirectory()
        self.router = ActionRouter(backend='mini', brain=FakeBrain(), mute_tts=True,
                                   memory=JsonMemory(self.memory_dir.name + '/memory.json'))

    def tearDown(self):
        self.memory_dir.cleanup()

    def test_dialogue_result_is_recalled_without_calling_brain(self):
        result = self.router.handle('что нового')
        self.assertEqual(result['reply'], 'ответ')
        self.router.brain.answer = lambda text: (_ for _ in ()).throw(AssertionError('brain called'))
        recalled = self.router.handle('  Что   нового  ')
        self.assertTrue(recalled['memory_hit'])
        self.assertEqual(recalled['reply'], 'ответ')

    def test_open_commands_use_safe_application_allowlist(self):
        with patch('voice.actions.subprocess.Popen') as start:
            result = self.router.handle('джарвис открой терминал')
        self.assertEqual(result['tag'], 'open')
        start.assert_called_once_with(['open', '-a', 'Ghostty'])

        result = self.router.handle('джарвис открой неизвестное')
        self.assertEqual(result['tag'], 'open-unknown')

    def test_vpn_command_opens_vpnka(self):
        with patch('voice.actions.subprocess.Popen') as start:
            result = self.router.handle('джарвис включи впн')
        self.assertEqual(result['tag'], 'open')
        start.assert_called_once_with(['open', '-a', 'VPNKa.PRO.app'])

    def test_close_all_quits_every_supported_application(self):
        with patch('voice.actions.subprocess.run') as run:
            result = self.router.handle('джарвис закрой все приложения')
        self.assertEqual(result['tag'], 'close-all')
        self.assertEqual(run.call_count, 4)

    def test_empty_frame_has_no_face_or_finger_false_positive(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(detect_faces(frame), [])
        self.assertEqual(count_fingers(frame), (None, None))


if __name__ == '__main__':
    unittest.main()
