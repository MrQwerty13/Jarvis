"""Маршрутизация голосовых команд Jarvis."""

import re

from .ollama_chat import ensure_ollama_chat
from .vision_number import run_numai_camera_session


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
        print('Режим чисел: python -m numai camera', flush=True)
        self._say(intro)
        try:
            spoken = run_numai_camera_session(
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
