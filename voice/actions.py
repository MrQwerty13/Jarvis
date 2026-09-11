"""Маршрутизация голосовых команд Jarvis."""

import re

from .ollama_chat import ensure_ollama_chat
from .vision_number import read_number_from_camera


FREE_RE = re.compile(r'(ты\s+свобод(ен|на)|я\s+свободен|отбой|выключайся)', re.IGNORECASE)
NUMBERS_RE = re.compile(
    r'(распозн[ао]вание\s+чисел|распознай\s+числ|режим\s+чисел|смотри\s+числ)',
    re.IGNORECASE,
)


def normalize_command(text):
    return ' '.join((text or '').lower().replace('ё', 'е').split())


def is_free_command(text):
    return bool(FREE_RE.search(normalize_command(text)))


def is_numbers_command(text):
    return bool(NUMBERS_RE.search(normalize_command(text)))


class ActionRouter:
    def __init__(
        self,
        language='ru',
        backend='ollama',
        ollama_model=None,
        ollama_host=None,
        camera_index=0,
        brain=None,
    ):
        self.language = language
        self.backend = backend
        self.camera_index = camera_index
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
            # «Ты свободен» мог прийти внутри обычного ответа-запроса — уже проверили выше.
            return {'reply': reply, 'done': done or is_free_command(text), 'tag': 'ollama'}

        reply, label, confidence = self.brain.answer(text)
        return {
            'reply': reply,
            'done': label == 'bye' or is_free_command(text),
            'tag': f'{label or "fallback"} {confidence:.0%}',
        }

    def _handle_numbers(self):
        print('Режим чисел: смотрю в камеру…', flush=True)
        try:
            found = read_number_from_camera(camera_index=self.camera_index)
        except Exception as error:
            return {
                'reply': 'Не удалось открыть камеру: ' + str(error),
                'done': False,
                'tag': 'numbers-error',
            }
        if not found:
            return {
                'reply': 'Число не увидел. Покажи маркер на белом листе ближе к камере.',
                'done': False,
                'tag': 'numbers-miss',
            }
        value = found['value']
        score = found['score']
        reply = f'Вижу число {value}. Оценка модели {score:.0%}.'
        return {'reply': reply, 'done': False, 'tag': 'numbers'}
