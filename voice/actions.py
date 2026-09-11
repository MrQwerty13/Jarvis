"""Маршрутизация голосовых команд Jarvis."""

import re
import subprocess

from .ollama_chat import ensure_ollama_chat
from .vision_number import run_camai_camera_session


FREE_RE = re.compile(r'(ты\s+свобод(ен|на)|я\s+свободен|отбой|выключайся)', re.IGNORECASE)
# Vosk часто даёт «распознавания чисел» / «распознование чисел».
NUMBERS_RE = re.compile(
    r'('
    r'распозн[ао]ван\w*\s+чисел'
    r'|распознай\s+числ\w*'
    r'|режим\s+чисел'
    r'|смотри\s+числ\w*'
    r'|камера\s+чисел'
    r')',
    re.IGNORECASE,
)
CAMERA_RE = re.compile(r'(?:джарвис\s+)?(умная\s+камера|смотри)\b', re.IGNORECASE)
OPEN_RE = re.compile(r'(?:джарвис\s+)?открой\s+(.+)$', re.IGNORECASE)
VPN_RE = re.compile(r'(?:джарвис\s+)?включи\s+впн\b', re.IGNORECASE)
CLOSE_ALL_RE = re.compile(r'(?:джарвис\s+)?закрой\s+все(?:\s+приложения)?\b', re.IGNORECASE)
CLOSE_RE = re.compile(r'(?:джарвис\s+)?закрой\s+(.+)$', re.IGNORECASE)
APPLICATIONS = {
    'терминал': 'Ghostty',
    'ghostty': 'Ghostty',
    'чаты': 'Telegram',
    'чат': 'Telegram',
    'телеграм': 'Telegram',
    'интернет': 'Safari',
    'сафари': 'Safari',
    'впн': 'VPNKa.PRO.app',
    'vpn': 'VPNKa.PRO.app',
}


def normalize_command(text):
    return ' '.join((text or '').lower().replace('ё', 'е').split())


def is_free_command(text):
    return bool(FREE_RE.search(normalize_command(text)))


def is_numbers_command(text):
    normalized = normalize_command(text)
    if not NUMBERS_RE.search(normalized):
        return False
    # Вопросы вроде «что такое распознавание чисел» оставляем Ollama.
    if re.search(r'\b(что такое|расскажи|объясни|зачем|означает)\b', normalized):
        return False
    return True


def _command_matches(pattern, text):
    return bool(pattern.search(normalize_command(text)))


class ActionRouter:
    def __init__(
        self,
        language='ru',
        backend='ollama',
        ollama_model=None,
        ollama_host=None,
        camera_index=0,
        brain=None,
        speak_fn=None,
        mute_tts=False,
    ):
        self.language = language
        self.backend = backend
        self.camera_index = camera_index
        self.speak_fn = speak_fn
        self.mute_tts = mute_tts
        if brain is not None:
            self.brain = brain
        elif backend == 'ollama':
            kwargs = {'language': language}
            if ollama_model:
                kwargs['model'] = ollama_model
            if ollama_host:
                kwargs['host'] = ollama_host
            self.brain = ensure_ollama_chat(**kwargs)
        else:
            from .brain import ensure_brain
            self.brain = ensure_brain(language=language)

    def _say(self, text):
        if self.mute_tts or not text:
            return
        if self.speak_fn:
            self.speak_fn(text)

    def handle(self, text):
        text = (text or '').strip()
        if not text:
            return {'reply': 'Слушаю.', 'done': False, 'tag': 'empty'}

        if is_free_command(text):
            reply = 'Хорошо. Выключаюсь.' if self.language == 'ru' else 'Shutting down.'
            return {'reply': reply, 'done': True, 'tag': 'free'}

        if _command_matches(CAMERA_RE, text):
            return self._handle_camera()

        if _command_matches(VPN_RE, text):
            return self._open_application('впн')

        if _command_matches(CLOSE_ALL_RE, text):
            return self._close_applications()

        closed = CLOSE_RE.search(normalize_command(text))
        if closed:
            return self._close_applications(closed.group(1))

        opened = OPEN_RE.search(normalize_command(text))
        if opened:
            return self._open_application(opened.group(1))

        if is_numbers_command(text):
            return self._handle_numbers()

        if self.backend == 'ollama':
            reply, done = self.brain.answer(text)
            return {'reply': reply, 'done': done or is_free_command(text), 'tag': 'ollama'}

        reply, label, confidence = self.brain.answer(text)
        return {
            'reply': reply,
            'done': label == 'bye' or is_free_command(text),
            'tag': f'{label or "fallback"} {confidence:.0%}',
        }

    def _handle_numbers(self):
        intro = (
            'Включаю распознавание чисел. Покажи маркер на белом листе. '
            'Выход из окна камеры — Q.'
        )
        print('Режим camAI: распознавание чисел', flush=True)
        self._say(intro)
        try:
            spoken = run_camai_camera_session(
                camera_index=self.camera_index,
                speak_fn=None if self.mute_tts else self.speak_fn,
            )
        except Exception as error:
            return {
                'reply': 'Не удалось запустить камеру NumAI: ' + str(error),
                'done': False,
                'tag': 'numbers-error',
                'skip_tts': False,
            }
        if spoken:
            reply = 'Режим чисел завершён. Назвал: ' + ', '.join(spoken) + '.'
        else:
            reply = 'Режим чисел завершён. Чисел не увидел.'
        # Числа уже озвучены по ходу; финальную фразу тоже скажем коротко.
        return {'reply': reply, 'done': False, 'tag': 'numbers', 'skip_tts': False}

    def _handle_camera(self):
        self._say('Включаю умную камеру camAI.')
        try:
            from camai.camera import run_camera
            from numai.model import MLP
            from pathlib import Path
            root = Path(__file__).resolve().parent.parent
            model = MLP.load(root / 'models' / 'arabic.npz')
            run_camera(model, index=self.camera_index, threshold=0.65,
                       speak_fn=None if self.mute_tts else self.speak_fn)
        except Exception as error:
            return {'reply': 'Не удалось запустить camAI: ' + str(error),
                    'done': False, 'tag': 'camera-error'}
        return {'reply': 'Умная камера camAI выключена.', 'done': False, 'tag': 'camera'}

    def _open_application(self, requested):
        app = APPLICATIONS.get(normalize_command(requested).strip())
        if app is None:
            return {'reply': 'Я умею открыть Терминал, Чаты, Интернет или VPN.',
                    'done': False, 'tag': 'open-unknown'}
        try:
            subprocess.Popen(['open', '-a', app])
        except OSError as error:
            return {'reply': f'Не удалось открыть {app}: {error}',
                    'done': False, 'tag': 'open-error'}
        return {'reply': f'Открываю {app}.', 'done': False, 'tag': 'open'}

    def _close_applications(self, requested=None):
        if requested is None:
            apps = list(dict.fromkeys(APPLICATIONS.values()))
        else:
            app = APPLICATIONS.get(normalize_command(requested).strip())
            if app is None:
                return {'reply': 'Я умею закрыть Терминал, Чаты, Интернет или VPN.',
                        'done': False, 'tag': 'close-unknown'}
            apps = [app]
        failures = []
        for app in apps:
            script = f'tell application "{app}" to quit'
            try:
                subprocess.run(['osascript', '-e', script], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                failures.append(app)
        if failures:
            return {'reply': 'Не удалось закрыть: ' + ', '.join(failures) + '.',
                    'done': False, 'tag': 'close-error'}
        if requested is None:
            return {'reply': 'Закрываю все приложения, которыми могу управлять.',
                    'done': False, 'tag': 'close-all'}
        return {'reply': f'Закрываю {apps[0]}.', 'done': False, 'tag': 'close'}
